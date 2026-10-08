#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ninfer/ops/add_bias.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_bias.h"
#include "ninfer/ops/residual_add.h"
#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::direct_bf16_weight;
constexpr ReductionCriterion kCriterion{3.3e-3, 5.0e-3, 6.3e-3};

double round_bf16(double value) {
    // Evaluate the semantic BF16 seam directly in FP64, without an extra
    // FP32 cast that could double-round a value close to a BF16 midpoint.
    if (value == 0) return value;
    int exponent = 0;
    const double magnitude = std::frexp(std::abs(value), &exponent);
    (void)magnitude;
    const double step = std::ldexp(1.0, std::max(exponent - 8, -133));
    const double units = std::abs(value) / step;
    const double lower = std::floor(units);
    const double fraction = units - lower;
    const bool increment = fraction > 0.5 ||
                           (fraction == 0.5 && std::fmod(lower, 2.0) != 0);
    return std::copysign((lower + (increment ? 1.0 : 0.0)) * step, value);
}

std::vector<std::uint16_t> activation(int k, int t) {
    std::vector<std::uint16_t> bits(std::size_t(k) * t);
    for (int i = 0; i < k; ++i)
        bits[i] = f32_to_bf16(float(((i * 29 + 17) & 255) - 128) / 512.f);
    for (int col = 1; col < t; ++col) for (int lane = 0; lane < 4; ++lane)
        bits[std::size_t(col) * k + (col * 71 + lane * 251) % k] =
            f32_to_bf16(float(((col * 19 + lane * 31) & 127) - 64) / 128.f);
    return bits;
}

std::vector<double> oracle(const HostWeight& w, const std::vector<std::uint16_t>& x,
                           const std::vector<std::uint16_t>& bias,
                           const std::vector<std::uint16_t>& residual, int t, bool add,
                           ops::GeluMode mode) {
    std::vector<std::vector<std::pair<int, double>>> nonzero(t);
    for (int col = 0; col < t; ++col) for (int k = 0; k < w.k; ++k) {
        const auto value = x[std::size_t(col) * w.k + k];
        if ((value & 0x7fffU) != 0) nonzero[col].emplace_back(k, bf16_to_f32(value));
    }
    std::vector<double> reference(std::size_t(w.n) * t);
    const int count = std::min(16U, std::max(1U, std::thread::hardware_concurrency()));
    std::vector<std::thread> workers;
    for (int thread = 0; thread < count; ++thread) workers.emplace_back([&, thread] {
        for (int row = w.n * thread / count; row < w.n * (thread + 1) / count; ++row) {
            for (int col = 0; col < t; ++col) {
                double sum = 0;
                for (auto [k, value] : nonzero[col])
                    sum += bf16_to_f32(w.bits[std::size_t(row) * w.k + k]) * value;
                // These two casts are explicit semantic seams in linear_bias.h.
                const double projected = round_bf16(sum);
                const double z = round_bf16(projected + bf16_to_f32(bias[row]));
                const auto index = std::size_t(col) * w.n + row;
                reference[index] = add ? z + bf16_to_f32(residual[index]) :
                    (mode == ops::GeluMode::Exact ?
                        0.5 * z * (1 + std::erf(z / std::sqrt(2.0))) :
                        0.5 * z * (1 + std::tanh(std::sqrt(2.0 / std::acos(-1.0)) *
                                                    (z + 0.044715 * z * z * z))));
            }
        }
    });
    for (auto& worker : workers) worker.join();
    return reference;
}

int run_case(DeviceWeight& weight, int tokens, bool add, ops::GeluMode mode, bool graph) {
    auto bits = activation(weight.host.k, tokens);
    std::vector<std::uint16_t> bias(weight.host.n), residual(std::size_t(weight.host.n) * tokens);
    for (int i = 0; i < weight.host.n; ++i)
        bias[i] = f32_to_bf16(float((i * 37 % 127) - 63) / 64.f);
    for (std::size_t i = 0; i < residual.size(); ++i)
        residual[i] = f32_to_bf16(float((i * 31 % 127) - 63) / 128.f);
    auto input = to_device(bits);
    auto bias_device = to_device(bias);
    GuardedDeviceBuffer output(residual.size() * 2), staged(residual.size() * 2);
    output.copy_from_host(residual.data(), output.bytes());
    Tensor x(input.p, DType::BF16, {weight.host.k, tokens});
    Tensor b(bias_device.p, DType::BF16, {weight.host.n});
    Tensor y(output.data(), DType::BF16, {weight.host.n, tokens});
    Tensor temporary(staged.data(), DType::BF16, {weight.host.n, tokens});
    WorkspaceArena workspace(256);
    DeviceContext context;
    const auto compute = [&] {
        if (add) ops::linear_bias_add(x, weight.view(), b, y, context.stream);
        else ops::linear_bias_gelu(x, weight.view(), b, mode, y, context.stream);
    };
    cuda_synchronize();
    if (graph) {
        DecodeGraphDefinition definition;
        DecodeGraphExecutable executable;
        definition.capture(context.stream, compute);
        executable.instantiate(definition);
        executable.launch(context.stream);
        cuda_synchronize();
        for (auto& v : bits) v ^= 0x8000U;
        input.copy_from_host(bits.data(), input.bytes);
        output.copy_from_host(residual.data(), output.bytes());
        cuda_synchronize();
        executable.launch(context.stream);
    } else compute();
    cuda_synchronize();
    const auto actual_bits = from_device<std::uint16_t>(output.data(), residual.size());
    std::vector<double> actual(actual_bits.size());
    std::transform(actual_bits.begin(), actual_bits.end(), actual.begin(), bf16_to_f32);
    const std::string label = std::string(add ? "LinearBiasAdd" : "LinearBiasGelu") +
        " N=" + std::to_string(weight.host.n) + " K=" + std::to_string(weight.host.k) +
        " T=" + std::to_string(tokens) + (graph ? " graph" : " eager");
    int failures = verify_reduction(label, actual,
        oracle(weight.host, bits, bias, residual, tokens, add, mode), kCriterion);
    failures += output.verify_guards(label);

    // Independent oracle above is authoritative; exact staged parity also checks
    // the BF16 seams and use of fresh operands on graph replay.
    ops::linear(x, weight.view(), temporary, ops::LinearPolicy::A16Only, workspace, context.stream);
    ops::add_bias(b, temporary, context.stream);
    if (add) {
        output.copy_from_host(residual.data(), output.bytes());
        cuda_synchronize();
        ops::residual_add(temporary, y, context.stream);
    } else ops::gelu(temporary, mode, context.stream);
    cuda_synchronize();
    failures += verify_exact((label + " staged parity").c_str(), actual_bits,
        from_device<std::uint16_t>(add ? output.data() : staged.data(), residual.size()));
    failures += staged.verify_guards(label + " staged");
    failures += verify_exact((label + " preserved input").c_str(), from_device<std::uint16_t>(input.p, bits.size()), bits);
    failures += verify_exact((label + " preserved bias").c_str(), from_device<std::uint16_t>(bias_device.p, bias.size()), bias);
    if (workspace.used() != 0 || workspace.peak_used() != 0) ++failures;
    return failures;
}
} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) return 77;
    try {
        int failures = 0;
        for (int k : {768, 1536, 3072}) {
            DeviceWeight weight(make_patterned(768, k, 1701 + k));
            for (int t : {1, 4, 9, 31, 32, 33, 63, 64, 65, 127, 128, 129, 6144})
                failures += run_case(weight, t, true, ops::GeluMode::Tanh, false);
            failures += run_case(weight, 65, true, ops::GeluMode::Tanh, true);
            failures += weight.verify_preserved("LinearBiasAdd");
        }
        for (int k : {768, 3072}) {
            DeviceWeight weight(make_patterned(3072, k, 1721 + k));
            for (auto mode : {ops::GeluMode::Exact, ops::GeluMode::Tanh}) {
                for (int t : {1, 4, 9, 31, 32, 33, 63, 64, 65, 129, 1536, 6144})
                    failures += run_case(weight, t, false, mode, false);
                failures += run_case(weight, 65, false, mode, true);
            }
            failures += weight.verify_preserved("LinearBiasGelu");
        }
        std::cout << (failures ? "FAIL" : "OK") << " LinearBias BF16 correctness\n";
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
