#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::direct_bf16_weight;

// The same A16 compute criterion used by the other LinearSwiGLU suites.
constexpr ReductionCriterion kA16Tolerance{3.3e-3, 5.0e-3, 6.3e-3};

std::vector<std::uint16_t> activation(int hidden, int tokens) {
    std::vector<std::uint16_t> bits(std::size_t(hidden) * tokens, 0);
    // A dense first token checks every represented K. Later tokens rotate four
    // nonzero columns so a complete FP64 oracle remains practical at large T.
    for (int k = 0; k < hidden; ++k)
        bits[k] = f32_to_bf16(float(((k * 29 + 17) & 255) - 128) / 512.f);
    for (int t = 1; t < tokens; ++t) for (int lane = 0; lane < 4; ++lane) {
        const int k = (t * 71 + lane * 251) % hidden;
        bits[std::size_t(t) * hidden + k] =
            f32_to_bf16(float(((t * 19 + lane * 31) & 127) - 64) / 128.f);
    }
    return bits;
}

std::vector<double> oracle(const HostWeight& weight,
                           const std::vector<std::uint16_t>& bits, int tokens) {
    const int rows = weight.n / 2;
    std::vector<std::vector<std::pair<int, double>>> nonzero(tokens);
    for (int t = 0; t < tokens; ++t) for (int k = 0; k < weight.k; ++k) {
        const auto value = bits[std::size_t(t) * weight.k + k];
        if ((value & 0x7fffU) != 0) nonzero[t].emplace_back(k, bf16_to_f32(value));
    }
    std::vector<double> reference(std::size_t(rows) * tokens);
    const int threads = std::min(16U, std::max(1U, std::thread::hardware_concurrency()));
    std::vector<std::thread> workers;
    for (int thread = 0; thread < threads; ++thread) workers.emplace_back([&, thread] {
        for (int row = rows * thread / threads; row < rows * (thread + 1) / threads; ++row) {
            for (int t = 0; t < tokens; ++t) {
                double gate = 0, up = 0;
                for (const auto [k, x] : nonzero[t]) {
                    gate += bf16_to_f32(weight.bits[std::size_t(row) * weight.k + k]) * x;
                    up += bf16_to_f32(weight.bits[std::size_t(rows + row) * weight.k + k]) * x;
                }
                const double exponential = std::exp(gate >= 0 ? -gate : gate);
                const double silu = gate >= 0 ? gate / (1 + exponential)
                                              : gate * exponential / (1 + exponential);
                reference[std::size_t(t) * rows + row] = silu * up;
            }
        }
    });
    for (auto& worker : workers) worker.join();
    return reference;
}

int run_case(DeviceWeight& weight, int tokens, bool replay = false) {
    auto bits = activation(weight.host.k, tokens);
    DeviceBuffer input = to_device(bits);
    GuardedDeviceBuffer output(std::size_t(weight.host.n / 2) * tokens * sizeof(std::uint16_t));
    output.fill(0xff);
    Tensor x(input.p, DType::BF16, {weight.host.k, tokens});
    Tensor y(output.data(), DType::BF16, {weight.host.n / 2, tokens});
    WorkspaceArena workspace(256);
    const auto capacity = ops::linear_swiglu_workspace_capacity_bytes(
        QType::BF16, weight.host.n, weight.host.k, tokens, tokens);
    if (capacity != 0) throw std::runtime_error("paired BF16 LinearSwiGLU requires no workspace");
    DeviceContext context;
    auto compute = [&] {
        ops::linear_swiglu(x, weight.view(), y, ops::LinearPolicy::A16Only,
                          workspace, context.stream, context.multiprocessor_count());
    };
    // Fixture uploads and output poisoning use the default stream; the Op is
    // exercised on a nonblocking stream after those transfers have completed.
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
        for (auto& value : bits) value ^= 0x8000U;
        input.copy_from_host(bits.data(), input.bytes);
        output.fill(0xff);
        cuda_synchronize();
        graph.launch(context.stream);
        cuda_synchronize();
    }
    const std::string label = "LinearSwiGLU BF16 N=" + std::to_string(weight.host.n) +
        " T=" + std::to_string(tokens) + (replay ? " graph" : " eager");
    int failures = output.verify_guards(label);
    const auto actual_bits = from_device<std::uint16_t>(output.data(),
                                                       std::size_t(weight.host.n / 2) * tokens);
    std::vector<double> actual(actual_bits.size());
    std::transform(actual_bits.begin(), actual_bits.end(), actual.begin(), bf16_to_f32);
    failures += verify_reduction(label, actual, oracle(weight.host, bits, tokens), kA16Tolerance);
    if (from_device<std::uint16_t>(input.p, bits.size()) != bits) {
        std::cerr << label << ": input was modified\n";
        ++failures;
    }
    if (workspace.used() != 0 || workspace.peak_used() != 0) {
        std::cerr << label << ": unexpected transient workspace\n";
        ++failures;
    }
    return failures;
}
} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) return 77;
    try {
        int failures = 0;
        DeviceWeight weight(make_patterned(7168, 1024, 1607));
        for (int t : {1, 2, 3, 4, 5, 6, 7, 8, 9, 31, 32, 33, 63, 64, 65,
                      127, 128, 129, 255, 256, 257, 319, 320, 321, 383, 384, 385, 783,
                      447, 448, 449, 511, 512, 513,
                      1023, 1024, 1025, 2047, 2048, 2049}) failures += run_case(weight, t);
        for (int t : {4, 33, 65, 256, 321, 1024}) failures += run_case(weight, t, true);
        failures += weight.verify_preserved("LinearSwiGLU BF16 7168");
        DeviceWeight paired(make_patterned(6144, 1024, 1609));
        for (int t : {1, 4, 5, 6, 8, 9, 31, 32, 33, 63, 64, 65, 129, 159, 160, 161, 1024})
            failures += run_case(paired, t);
        failures += run_case(paired, 159, true);
        failures += paired.verify_preserved("LinearSwiGLU BF16 6144");
        std::cout << (failures ? "FAIL" : "OK") << " LinearSwiGLU BF16_A16 correctness\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
