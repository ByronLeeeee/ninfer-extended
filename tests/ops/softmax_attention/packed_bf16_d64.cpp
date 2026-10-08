#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ninfer/ops/softmax_attention.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::test;
constexpr int kDim = 64, kQueryPatterns = 7;
// Retain the established D64 BF16 attention relative-L2 and RMS-scaled
// gross-error limits from tools/xiaomi_ocr/qualify_attention.cu. The D72
// suite has a different accumulator profile and its own tighter criterion.
constexpr ReductionCriterion kBf16Criterion{6e-3, 1e-3, 2.8e-3};
constexpr double kGrossErrorOverReferenceRms = 4.5e-2;

int run_case(int tokens, int heads, bool interleaved, int entry, bool replay = false,
             int segments = 1) {
    const int segment_length = tokens / segments;
    const int plane = heads * kDim, stride = plane * (interleaved ? 3 : 1);
    const std::size_t count = std::size_t(tokens) * stride;
    std::vector<std::uint16_t> q(count, 0), k(count, 0), v(count, 0);
    std::vector<std::uint16_t> query(kQueryPatterns * plane), values(tokens * heads);
    std::mt19937 rng(5070 + heads);
    std::uniform_real_distribution<float> random(-1.f, 1.f);
    for (auto& x : query) x = f32_to_bf16(random(rng));
    for (auto& x : values) x = f32_to_bf16(random(rng));
    for (int t = 0; t < tokens; ++t) for (int h = 0; h < heads; ++h) for (int d = 0; d < kDim; ++d) {
        const auto i = std::size_t(t) * stride + h * kDim + d;
        q[i] = query[(t % kQueryPatterns) * plane + h * kDim + d];
        k[i] = f32_to_bf16(random(rng));
        // An exact power-of-two value basis permits a complete FP64 oracle
        // without a quadratic 64-column value reduction for every query.
        v[i] = f32_to_bf16(std::ldexp(bf16_to_f32(values[t * heads + h]), d % 5 - 2));
    }
    std::vector<std::uint16_t> storage;
    if (interleaved) {
        storage.resize(count);
        for (int t = 0; t < tokens; ++t) {
            const auto start = std::size_t(t) * stride;
            std::copy_n(q.data() + start, plane, storage.data() + start);
            std::copy_n(k.data() + start, plane, storage.data() + start + plane);
            std::copy_n(v.data() + start, plane, storage.data() + start + 2 * plane);
        }
    }
    DeviceBuffer dq = to_device(interleaved ? storage : q);
    DeviceBuffer dk = interleaved ? DeviceBuffer() : to_device(k);
    DeviceBuffer dv = interleaved ? DeviceBuffer() : to_device(v);
    Tensor tq(dq.p, DType::BF16, {kDim, heads, tokens});
    tq.nb[2] = stride * 2;
    Tensor tk = tq, tv = tq;
    tk.data = interleaved ? static_cast<std::uint8_t*>(dq.p) + plane * 2 : dk.p;
    tv.data = interleaved ? static_cast<std::uint8_t*>(dq.p) + plane * 4 : dv.p;
    GuardedDeviceBuffer output(std::size_t(tokens) * plane * 2);
    output.fill(0xff);
    Tensor out(output.data(), DType::BF16, {kDim, heads, tokens});
    std::vector<int> offsets(segments + 1);
    for (int s = 0; s <= segments; ++s) offsets[s] = s * segment_length;
    DeviceBuffer boundaries = to_device(offsets);
    Tensor cu(boundaries.p, DType::I32, {segments + 1});
    const ops::AttentionHeadGeometry geometry{kDim, heads, heads};
    const auto capacity = ops::packed_softmax_attention_workspace_capacity_bytes(
        geometry, tokens, tokens, segments, segments);
    WorkspaceArena workspace(std::max(std::size_t{1}, capacity));
    DeviceContext context;
    auto compute = [&] {
        if (entry == 0) ops::softmax_attention(tq, tk, tv, geometry, .125f, workspace, out, context.stream);
        else if (entry == 1) ops::packed_softmax_attention(tq, tk, tv, geometry, .125f, segment_length, out, context.stream);
        else ops::packed_softmax_attention(tq, tk, tv, geometry, .125f, cu, workspace, out, context.stream);
    };
    cuda_synchronize();
    compute();
    cuda_synchronize();
    if (replay) {
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        definition.capture(context.stream, compute);
        graph.instantiate(definition);
        graph.launch(context.stream);
        cuda_synchronize();
        for (auto& x : query) x ^= 0x8000U;
        for (int t = 0; t < tokens; ++t) for (int x = 0; x < plane; ++x) {
            q[std::size_t(t) * stride + x] = query[(t % kQueryPatterns) * plane + x];
            if (interleaved) storage[std::size_t(t) * stride + x] = q[std::size_t(t) * stride + x];
        }
        const auto& updated = interleaved ? storage : q;
        dq.copy_from_host(updated.data(), updated.size() * 2);
        output.fill(0xff);
        cuda_synchronize();
        graph.launch(context.stream);
        cuda_synchronize();
    }
    std::vector<double> mean(segments * kQueryPatterns * heads);
    std::vector<std::thread> workers;
    for (int s = 0; s < segments; ++s) for (int h = 0; h < heads; ++h) workers.emplace_back([&, s, h] {
        std::vector<double> scores(segment_length);
        for (int pattern = 0; pattern < kQueryPatterns; ++pattern) {
            double maximum = -INFINITY;
            for (int local = 0; local < segment_length; ++local) {
                const int t = s * segment_length + local;
                double dot = 0;
                for (int d = 0; d < kDim; ++d)
                    dot += double(bf16_to_f32(query[pattern * plane + h * kDim + d])) *
                           double(bf16_to_f32(k[std::size_t(t) * stride + h * kDim + d]));
                scores[local] = dot * .125;
                maximum = std::max(maximum, scores[local]);
            }
            double denominator = 0, numerator = 0;
            for (int local = 0; local < segment_length; ++local) {
                const int t = s * segment_length + local;
                const double probability = std::exp(scores[local] - maximum);
                denominator += probability;
                numerator += probability * bf16_to_f32(values[t * heads + h]);
            }
            mean[(s * kQueryPatterns + pattern) * heads + h] = numerator / denominator;
        }
    });
    for (auto& worker : workers) worker.join();
    auto actual = from_device_bf16(output.data(), std::size_t(tokens) * plane);
    std::vector<double> reference(actual.size());
    for (int t = 0; t < tokens; ++t) for (int h = 0; h < heads; ++h) for (int d = 0; d < kDim; ++d)
        reference[(std::size_t(t) * heads + h) * kDim + d] =
            std::ldexp(mean[((t / segment_length) * kQueryPatterns + t % kQueryPatterns) * heads + h], d % 5 - 2);
    const std::string label = "PackedAttention BF16 D64 H=" + std::to_string(heads) +
        " T=" + std::to_string(tokens) + " S=" + std::to_string(segments) + " entry=" + std::to_string(entry) +
        (interleaved ? " interleaved" : " contiguous") + (replay ? " graph" : " eager");
    int failures = output.verify_guards(label) + verify_reduction(label, actual, reference, kBf16Criterion);
    const auto stats = compute_reduction_stats(actual.data(), reference.data(), actual.size());
    if (stats.maximum_absolute_error > kGrossErrorOverReferenceRms * stats.reference_root_mean_square) {
        std::cerr << label << ": RMS-scaled gross-error limit exceeded\n";
        ++failures;
    }
    if (from_device<std::uint16_t>(dq, count) != (interleaved ? storage : q) ||
        (!interleaved && (from_device<std::uint16_t>(dk, count) != k || from_device<std::uint16_t>(dv, count) != v)))
        ++failures;
    if (from_device<int>(boundaries, offsets.size()) != offsets ||
        workspace.peak_used() > capacity || ((entry != 2 || segments == 1) && workspace.peak_used() != 0))
        ++failures;
    return failures;
}
} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) return 77;
    try {
        int failures = 0;
        for (int heads : {12, 16}) {
            for (int tokens : {63, 64, 65, 4095, 4096, 4097, 6655, 6656, 6657,
                               7144, 7167, 7168, 7169, 9184, 9216})
                failures += run_case(tokens, heads, tokens % 2 != 0, 1);
            failures += run_case(6656, heads, true, 0);
            failures += run_case(6657, heads, true, 2, true);
            failures += run_case(2 * 6656, heads, true, 1, true, 2);
            failures += run_case(2 * 6657, heads, false, 2, false, 2);
        }
        std::cout << (failures ? "FAIL" : "OK") << " PackedAttention BF16 D64\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
