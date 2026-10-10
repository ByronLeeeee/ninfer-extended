#include "ninfer/ops/rmsnorm_rope.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {
constexpr int kD = 256;
constexpr float kEpsilon = 1.0e-6F;
constexpr float kTheta = 1.0e7F;
constexpr ReductionCriterion kCriterion{1.0 / 256.0, 1.0e-4, 0.035};

// The oracle uses naive FP64 math, no CUDA helper or production coefficient table.
// BF16 normalization is part of this Op's specified formula.
std::vector<double> oracle(const std::vector<std::uint16_t>& input,
                            const std::vector<std::uint16_t>& weights,
                            const std::vector<std::int32_t>& positions, int heads, int tokens,
                            int axes, bool offset, float epsilon, float theta) {
    std::vector<double> result(input.size());
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < heads; ++h) {
            const auto base = (static_cast<std::size_t>(t) * heads + h) * kD;
            double sum = 0.0;
            for (int d = 0; d < kD; ++d) {
                const double v = bf16_to_f32(input[base + d]);
                sum += v * v;
            }
            const double inverse = 1.0 / std::sqrt(sum / kD + epsilon);
            double normalized[kD];
            for (int d = 0; d < kD; ++d) {
                const double n = static_cast<double>(bf16_to_f32(input[base + d])) * inverse *
                    (bf16_to_f32(weights[d]) + (offset ? 1.0 : 0.0));
                normalized[d] = bf16_to_f32(f32_to_bf16(static_cast<float>(n)));
                result[base + d] = normalized[d];
            }
            for (int i = 0; i < 32; ++i) {
                const int axis = axes == 3 ? i % 3 : 0;
                const double phase = static_cast<double>(positions[axis * tokens + t]) *
                    std::pow(static_cast<double>(theta), -2.0 * i / 64.0);
                const double c = std::cos(phase), s = std::sin(phase);
                result[base + i] = normalized[i] * c - normalized[i + 32] * s;
                result[base + i + 32] = normalized[i + 32] * c + normalized[i] * s;
            }
        }
    }
    return result;
}

int run_case(int q_heads, int k_heads, int tokens, int axes, bool offset, int first_position,
              bool replay, float magnitude = 2.0F, float epsilon = kEpsilon,
              float theta = kTheta) {
    std::mt19937 generator(1731U + tokens + q_heads * 37);
    std::uniform_real_distribution<float> values(-magnitude, magnitude), gains(-0.25F, 0.25F);
    const auto make = [&](std::size_t size, bool weight) {
        std::vector<std::uint16_t> bits(size);
        for (auto& b : bits) b = f32_to_bf16(weight ? gains(generator) + (offset ? 0.0F : 1.0F)
                                                   : values(generator));
        return bits;
    };
    auto q_input = make(static_cast<std::size_t>(q_heads) * tokens * kD, false);
    auto k_input = make(static_cast<std::size_t>(k_heads) * tokens * kD, false);
    auto q_weight = make(kD, true), k_weight = make(kD, true);
    std::vector<std::int32_t> position(static_cast<std::size_t>(tokens) * axes);
    const auto set_positions = [&](int first) {
        for (int a = 0; a < axes; ++a)
            for (int t = 0; t < tokens; ++t) position[a * tokens + t] = first + t + a * 17;
    };
    set_positions(first_position);
    GuardedDeviceBuffer q_storage(q_input.size() * 2), k_storage(k_input.size() * 2);
    auto wq = to_device(q_weight), wk = to_device(k_weight), pos = to_device(position);
    Tensor q(q_storage.data(), DType::BF16, {kD, q_heads, tokens});
    Tensor k(k_storage.data(), DType::BF16, {kD, k_heads, tokens});
    Tensor nq(wq.p, DType::BF16, {kD}), nk(wk.p, DType::BF16, {kD});
    Tensor positions(pos.p, DType::I32, {tokens, axes});
    DeviceContext device;
    cuda_synchronize();
    const auto upload = [&] {
        cuda_check(cudaMemcpyAsync(q_storage.data(), q_input.data(), q_input.size() * 2,
                                    cudaMemcpyHostToDevice, device.stream), "upload q");
        cuda_check(cudaMemcpyAsync(k_storage.data(), k_input.data(), k_input.size() * 2,
                                    cudaMemcpyHostToDevice, device.stream), "upload k");
        cuda_synchronize(device.stream);
    };
    const auto launch = [&] {
        ops::rmsnorm_rope_partial(positions, nq, nk, epsilon, theta, offset, q, k, device.stream);
    };
    upload();
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (replay) { definition.capture(device.stream, launch); graph.instantiate(definition); }
    int failures = 0;
    for (int phase = 0; phase < (replay ? 2 : 1); ++phase) {
        if (phase) {
            for (auto& b : q_input) b ^= 0x8000;
            for (auto& b : k_input) b ^= 0x8000;
            set_positions(first_position + 113);
            for (auto& b : q_weight) b = f32_to_bf16(bf16_to_f32(b) * 0.75F);
            for (auto& b : k_weight) b = f32_to_bf16(bf16_to_f32(b) * 0.875F);
            pos.copy_from_host(position.data(), pos.bytes);
            wq.copy_from_host(q_weight.data(), wq.bytes);
            wk.copy_from_host(k_weight.data(), wk.bytes);
            cuda_synchronize();
            upload();
        }
        if (replay) graph.launch(device.stream); else launch();
        cuda_synchronize(device.stream);
        const std::string label = "partial RMSNorm/RoPE T=" + std::to_string(tokens) +
            " axes=" + std::to_string(axes) + " phase=" + std::to_string(phase);
        failures += q_storage.verify_guards(label);
        failures += k_storage.verify_guards(label);
        for (int which = 0; which < 2; ++which) {
            auto got_bits = from_device<std::uint16_t>(which ? k_storage.data() : q_storage.data(),
                                                        which ? k_input.size() : q_input.size());
            std::vector<double> got(got_bits.size());
            for (std::size_t i = 0; i < got.size(); ++i) got[i] = bf16_to_f32(got_bits[i]);
            const auto expected = oracle(which ? k_input : q_input, which ? k_weight : q_weight,
                                         position, which ? k_heads : q_heads, tokens, axes,
                                         offset, epsilon, theta);
            const int current_failures = verify_reduction(label, got, expected, kCriterion);
            failures += current_failures;
            if (current_failures) {
                const auto stats = compute_reduction_stats(got.data(), expected.data(), got.size());
                const auto index = static_cast<std::size_t>(stats.maximum_error_index);
                const auto& input = which ? k_input : q_input;
                const int heads = which ? k_heads : q_heads;
                const auto base = index / kD * kD;
                const int d = index % kD, t = index / kD / heads;
                std::cerr << "diagnostic which=" << which << " token=" << t << " dim=" << d
                          << " position=" << position[t] << " input=" << bf16_to_f32(input[index])
                          << " partner=" << bf16_to_f32(input[base + (d < 32 ? d + 32 : d - 32)])
                          << " normalized tail got=" << got[base + 128]
                          << " expected=" << expected[base + 128] << '\n';
            }
        }
        failures += verify_exact("q norm unchanged", from_device<std::uint16_t>(wq, kD), q_weight);
        failures += verify_exact("k norm unchanged", from_device<std::uint16_t>(wk, kD), k_weight);
        failures += verify_exact("positions unchanged", from_device<std::int32_t>(pos, position.size()), position);
    }
    return failures;
}

int rejections() {
    int failures = 0;
    DeviceBuffer data(8192), other(8192), weights(512), pos_data(12);
    Tensor q(data.p, DType::BF16, {256, 8, 1}), k(other.p, DType::BF16, {256, 2, 1});
    Tensor norm(weights.p, DType::BF16, {256}), positions(pos_data.p, DType::I32, {1});
    const auto invalid = [&](auto action) {
        try { action(); ++failures; } catch (const std::invalid_argument&) {}
    };
    const auto call = [&](Tensor p, Tensor qn, Tensor qq, Tensor kk, float eps, float theta) {
        ops::rmsnorm_rope_partial(p, qn, norm, eps, theta, true, qq, kk, nullptr);
    };
    invalid([&] { call(positions, norm, q, q, kEpsilon, kTheta); });
    invalid([&] { call(positions, Tensor(data.p, DType::BF16, {256}), q, k, kEpsilon, kTheta); });
    invalid([&] { call(Tensor(pos_data.p, DType::I32, {1, 2}), norm, q, k, kEpsilon, kTheta); });
    invalid([&] { call(positions, norm, Tensor(static_cast<char*>(data.p) + 2, DType::BF16, {256, 8}), k, kEpsilon, kTheta); });
    invalid([&] { call(positions, norm, q, k, 0.0F, kTheta); });
    invalid([&] { call(positions, norm, q, k, kEpsilon, std::numeric_limits<float>::infinity()); });
    invalid([&] { call(positions, norm, Tensor(data.p, DType::BF16, {256, 8, 0}), k, kEpsilon, kTheta); });
    invalid([&] { call(positions, norm, q, Tensor(other.p, DType::BF16, {256, 2, 2}), kEpsilon, kTheta); });
    return failures;
}
} // namespace

int main() {
    if (cuda_unavailable()) return 77;
    try {
        int failures = rejections();
        for (int tokens : {1, 2, 4, 8, 65, 257, 1807, 2048})
            for (int axes : {1, 3})
                failures += run_case(8, 2, tokens, axes, true, 30000, tokens <= 8);
        failures += run_case(3, 1, 17, 3, false, 131000, true);
        failures += run_case(64, 64, 2, 1, true, 260000, true);
        failures += run_case(1, 7, 1, 1, false, 0, true, 1.0e-5F, 1.0e-5F, 10000.0F);
        std::cout << (failures ? "FAIL" : "OK") << " rmsnorm_rope_partial\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "rmsnorm_rope_partial: " << error.what() << '\n';
        return 1;
    }
}
