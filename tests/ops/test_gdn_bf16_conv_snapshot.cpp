#include "ninfer/ops/gdn_input_proj.h"
#include "ops/direct_bf16_weight.h"
#include "ops/input_projection_test_common.h"
#include "core/device.h"
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::direct_bf16_weight;
using namespace ninfer::test::input_projection;

namespace {
// Independent FP64 projection/conv/SiLU oracle; the criterion covers private BF16 staging.
constexpr ReductionCriterion kTolerance{0.006, 0.008, 0.008};
constexpr int C = 6144, H = 1024, R = 2048;

int run(DeviceWeight& weight, int width, int batch, bool graph) {
    const int columns = width * batch, slots = columns + batch + 1;
    auto activation = make_bf16_activation(H, columns, 702 + columns);
    auto activation_bits = bf16_bits(activation);
    std::vector<float> cw(C * 4), sh(C * 3 * slots);
    fill_uniform(cw, 715, -0.3F, 0.3F); round_to_bf16(cw);
    fill_uniform(sh, 717, -0.2F, 0.2F); round_to_bf16(sh);
    auto cb = bf16_bits(cw), sb = bf16_bits(sh);
    auto dx = to_device(activation_bits), dc = to_device(cb);
    GuardedBf16Tensor state(C * 3, slots), q(R, columns), k(R, columns), v(R, columns), z(R, columns);
    std::vector<int> initial(batch), base(batch), valid(batch, width);
    for (int b = 0; b < batch; ++b) { initial[b] = columns + b; base[b] = width * b; }
    // Same-row initial/destination alias is part of the public contract.
    initial[0] = base[0];
    if (batch > 1) valid.back() = 1;
    auto di = to_device(initial), db = to_device(base), dv = to_device(valid);
    Tensor tx(dx.p, DType::BF16, {H, width, batch}), tc(dc.p, DType::BF16, {C, 4});
    Tensor ts(state.data(), DType::BF16, {C, 3, slots});
    Tensor ti(di.p, DType::I32, {batch}), tb(db.p, DType::I32, {batch}), tv(dv.p, DType::I32, {batch});
    Tensor tq(q.data(), DType::BF16, {R, width, batch}), tk(k.data(), DType::BF16, {R, width, batch});
    Tensor tout(v.data(), DType::BF16, {R, width, batch}), tz(z.data(), DType::BF16, {R, width, batch});
    const auto cap = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        QType::BF16, 8192, H, ops::LinearPolicy::A16Only, batch, width, width);
    WorkspaceArena ws(std::max<std::size_t>(1, cap));
    cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));
    auto launch = [&] { ops::gdn_input_proj_conv_snapshot(tx, weight.view(), tc, ts, tv, ti, tb,
        tq, tk, tout, tz, ops::LinearPolicy::A16Only, ws, stream); };
    state.copy_from_bits(sb); launch(); CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaGraph_t g{}; cudaGraphExec_t exec{};
    if (graph) {
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        launch(); CUDA_CHECK(cudaStreamEndCapture(stream, &g));
        CUDA_CHECK(cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0));
    }
    int failures = 0;
    auto invalid = weight.view();
    invalid.layout = QuantLayout::RowSplit;
    try {
        ops::gdn_input_proj_conv_snapshot(tx, invalid, tc, ts, tv, ti, tb,
            tq, tk, tout, tz, ops::LinearPolicy::A16Only, ws, stream);
        ++failures;
    } catch (const std::invalid_argument&) {}
    for (int replay = 0; replay < (graph ? 3 : 1); ++replay) {
        if (replay) {
            activation = make_bf16_activation(H, columns, 730 + replay);
            activation_bits = bf16_bits(activation);
            dx.copy_from_host(activation_bits.data(), activation_bits.size() * 2);
            // Change slot selectors without recapturing the graph.
            initial[0] = replay == 1 ? columns : 0;
            di.copy_from_host(initial.data(), initial.size() * sizeof(int));
            if (batch > 1) {
                valid.back() = replay == 1 ? 0 : width;
                dv.copy_from_host(valid.data(), valid.size() * sizeof(int));
            }
        }
        // Invalid columns must leave query/key/value untouched, even after
        // changing the valid-column selector without recapturing the graph.
        const std::vector<std::uint16_t> zeros(static_cast<std::size_t>(columns) * R, 0);
        q.copy_from_bits(zeros);
        k.copy_from_bits(zeros);
        v.copy_from_bits(zeros);
        state.copy_from_bits(sb);
        if (graph) CUDA_CHECK(cudaGraphLaunch(exec, stream)); else launch();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto aq = q.values(), ak = k.values(), av = v.values(), az = z.values();
        const auto actual_state = state.bits();
        std::vector<double> eq(columns * R), ek(columns * R), ev(columns * R), ez(columns * R);
        std::vector<double> es(sh.begin(), sh.end()), as(es.size());
        for (std::size_t i = 0; i < as.size(); ++i) as[i] = bf16_to_f32(actual_state[i]);
        for (int b = 0; b < batch; ++b) {
            for (int row = 0; row < 8192; ++row) {
                double s0 = 0, s1 = 0, s2 = 0;
                if (row < C) {
                    const auto offset = static_cast<std::size_t>(initial[b]) * C * 3 + row;
                    s0 = sh[offset]; s1 = sh[offset + C]; s2 = sh[offset + 2 * C];
                }
                for (int t = 0; t < width; ++t) {
                    const int col = b * width + t;
                    const double dot = dot_fp64(weight.host, row,
                        std::span<const float>(activation.data() + col * H, H));
                    if (row >= C) { ez[col * R + row - C] = dot; continue; }
                    if (t >= valid[b]) continue;
                    double conv = cw[row] * s0 + cw[C + row] * s1 + cw[2 * C + row] * s2 + cw[3 * C + row] * dot;
                    auto& expected = row < R ? eq : row < 2 * R ? ek : ev;
                    expected[col * R + row % R] = conv / (1.0 + std::exp(-conv));
                    const auto offset = static_cast<std::size_t>(base[b] + t) * C * 3 + row;
                    es[offset] = s1; es[offset + C] = s2; es[offset + 2 * C] = dot;
                    s0 = s1; s1 = s2; s2 = dot;
                }
            }
        }
        failures += compare("BF16 snapshot q", aq, eq, kTolerance);
        failures += compare("BF16 snapshot k", ak, ek, kTolerance);
        failures += compare("BF16 snapshot v", av, ev, kTolerance);
        failures += compare("BF16 snapshot z", az, ez, kTolerance);
        failures += compare("BF16 snapshot state", as, es, kTolerance);
        for (int slot = 0; slot < slots; ++slot) {
            bool changed = false;
            for (int b = 0; b < batch; ++b) changed |= slot >= base[b] && slot < base[b] + valid[b];
            if (!changed && !std::equal(sb.begin() + slot * C * 3, sb.begin() + (slot + 1) * C * 3,
                                       actual_state.begin() + slot * C * 3)) ++failures;
        }
        failures += q.verify_guards("q") + k.verify_guards("k") + v.verify_guards("v") + z.verify_guards("z") + state.verify_guards("state");
        failures += verify_preserved("input", dx, activation_bits) + verify_preserved("conv", dc, cb);
    }
    if (ws.used() != 0 || ws.peak_used() != cap) ++failures;
    if (exec) CUDA_CHECK(cudaGraphExecDestroy(exec));
    if (g) CUDA_CHECK(cudaGraphDestroy(g));
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::cout << "BF16 snapshot W=" << width << " B=" << batch << " graph=" << graph << " failures=" << failures << '\n';
    return failures;
}
} // namespace

int main() {
    if (cuda_unavailable()) return 77;
    DeviceWeight weight(make_patterned(8192, 1024, 709));
    int failures = run(weight, 1, 1, true);
    failures += run(weight, 2, 1, false);
    for (int batch : {2, 3, 4, 5, 8}) failures += run(weight, 1, batch, true);
    failures += run(weight, 2, 4, false);
    failures += weight.verify_preserved("parent");
    return failures ? 1 : 0;
}
