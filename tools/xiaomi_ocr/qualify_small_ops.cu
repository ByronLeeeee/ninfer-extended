#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/softmax_attention.h"
#include "core/device.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;

namespace {
std::mt19937 rng(5070);
std::normal_distribution<float> normal(0.0f, 0.4f);

double value(__nv_bfloat16 x) { return __bfloat162float(x); }

std::vector<__nv_bfloat16> random_values(std::size_t count) {
    std::vector<__nv_bfloat16> result(count);
    for (auto& x : result) x = __float2bfloat16_rn(normal(rng));
    return result;
}

void upload(DeviceBuffer& buffer, const auto& values) {
    buffer.copy_from_host(values.data(), values.size() * sizeof(values[0]));
}

// Criteria belong to the mathematical Op; every reported output must be finite.
struct Error {
    double squared_error = 0.0;
    double squared_reference = 0.0;
    double maximum_error = 0.0;
    double maximum_reference = 0.0;
    std::size_t count = 0;
    bool finite = true;

    void add(double actual, double expected) {
        finite = finite && std::isfinite(actual) && std::isfinite(expected);
        const double difference = actual - expected;
        squared_error += difference * difference;
        squared_reference += expected * expected;
        maximum_error = std::max(maximum_error, std::abs(difference));
        maximum_reference = std::max(maximum_reference, std::abs(expected));
        ++count;
    }

    bool report(const std::string& name, bool swiglu = false) const {
        const double rms = std::sqrt(squared_reference / count);
        const double nrms = std::sqrt(squared_error / squared_reference);
        const double peak = maximum_error / rms;
        // Existing LinearSwiGLU A16 criterion from tests/ops/linear_swiglu:
        // relative L2 <= 3.3e-3, gross error <= 5e-3 + 6.3e-3 * max(abs(reference)).
        const bool pass = finite && std::isfinite(nrms) && (swiglu
            ? nrms <= 0.0033 && maximum_error <= 0.005 + 0.0063 * maximum_reference
            : nrms < 0.006 && peak < 0.045);
        std::cout << "{\"case\":\"" << name << "\",\"nrms\":" << nrms
                  << ",\"max_error_over_reference_rms\":" << peak
                  << ",\"criterion\":\"" << (swiglu ? "SwiGLU A16" : "projection/attention A16") << "\""
                  << ",\"pass\":" << (pass ? "true" : "false") << "}\n" << std::flush;
        return pass;
    }
};

bool projection(int n, int k, int tokens, bool residual) {
    const auto x = random_values(k * tokens);
    const auto weights = random_values(std::size_t(n) * k);
    const auto initial = random_values(n * tokens);
    DeviceBuffer xb(x.size() * 2), wb(weights.size() * 2), out(n * tokens * 2);
    upload(xb, x);
    upload(wb, weights);
    upload(out, initial);
    Weight w;
    w.qtype = QType::BF16;
    w.layout = QuantLayout::Contiguous;
    w.qdata = wb.p;
    w.n = n;
    w.k = k;
    Tensor tx(xb.p, DType::BF16, {k, tokens});
    Tensor to(out.p, DType::BF16, {n, tokens});
    const auto capacity = residual
        ? ops::linear_add_workspace_capacity_bytes(QType::BF16, n, k, tokens, tokens)
        : n == 8192
            ? ops::gdn_input_proj_workspace_capacity_bytes(QType::BF16, n, k,
                ops::LinearPolicy::A16Only, tokens, tokens)
            : ops::attn_input_proj_workspace_capacity_bytes(QType::BF16, n, k,
                ops::LinearPolicy::A16Only, tokens, tokens);
    if (capacity != 0) throw std::runtime_error("BF16 fused projection should require no workspace");
    WorkspaceArena workspace(std::max<std::size_t>(1, capacity));
    DeviceBuffer second((residual ? 1 : 2048) * tokens * 2);
    DeviceBuffer third((residual ? 1 : 512) * tokens * 2);
    DeviceBuffer fourth(third.bytes);
    if (residual) {
        ops::linear_add(tx, w, to, workspace, nullptr);
    } else if (n == 8192) {
        Tensor qkv(out.p, DType::BF16, {6144, tokens});
        Tensor z(second.p, DType::BF16, {2048, tokens});
        ops::gdn_input_proj(tx, w, qkv, z, ops::LinearPolicy::A16Only, workspace, nullptr);
    } else {
        Tensor q(out.p, DType::BF16, {2048, tokens});
        Tensor gate(second.p, DType::BF16, {2048, tokens});
        Tensor key(third.p, DType::BF16, {512, tokens});
        Tensor v(fourth.p, DType::BF16, {512, tokens});
        ops::attn_input_proj(tx, w, q, gate, key, v, ops::LinearPolicy::A16Only,
                             workspace, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<__nv_bfloat16> q_host(2048 * tokens), gate_host(q_host.size());
        std::vector<__nv_bfloat16> k_host(512 * tokens), v_host(k_host.size());
        out.copy_to_host(q_host.data(), q_host.size() * 2);
        second.copy_to_host(gate_host.data(), gate_host.size() * 2);
        third.copy_to_host(k_host.data(), k_host.size() * 2);
        fourth.copy_to_host(v_host.data(), v_host.size() * 2);
        Error error;
        for (int t = 0; t < tokens; ++t) for (int row = 0; row < n; ++row) {
            double expected = 0.0;
            for (int column = 0; column < k; ++column)
                expected += value(weights[std::size_t(row) * k + column]) * value(x[t * k + column]);
            const double actual = row < 2048 ? value(q_host[t * 2048 + row])
                : row < 2560 ? value(k_host[t * 512 + row - 2048])
                : row < 4608 ? value(gate_host[t * 2048 + row - 2560])
                : value(v_host[t * 512 + row - 4608]);
            error.add(actual, expected);
        }
        return error.report("attn_input_T" + std::to_string(tokens));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> actual(n * tokens);
    if (residual) {
        out.copy_to_host(actual.data(), actual.size() * 2);
    } else {
        std::vector<__nv_bfloat16> qkv(6144 * tokens), z(2048 * tokens);
        out.copy_to_host(qkv.data(), qkv.size() * 2);
        second.copy_to_host(z.data(), z.size() * 2);
        for (int t = 0; t < tokens; ++t) {
            std::copy_n(qkv.data() + t * 6144, 6144, actual.data() + t * n);
            std::copy_n(z.data() + t * 2048, 2048, actual.data() + t * n + 6144);
        }
    }
    Error error;
    for (int t = 0; t < tokens; ++t) for (int row = 0; row < n; ++row) {
        double expected = residual ? value(initial[t * n + row]) : 0.0;
        for (int column = 0; column < k; ++column)
            expected += value(weights[std::size_t(row) * k + column]) * value(x[t * k + column]);
        error.add(value(actual[t * n + row]), expected);
    }
    return error.report((residual ? "linear_add_K" + std::to_string(k) : "gdn_input")
                        + "_T" + std::to_string(tokens));
}

bool dense_projection(int n, int k, int tokens, bool swiglu) {
    const int rows = swiglu ? n / 2 : n;
    const auto x = random_values(std::size_t(k) * tokens);
    const auto weights = random_values(std::size_t(n) * k);
    DeviceBuffer xb(x.size() * 2), wb(weights.size() * 2), out(std::size_t(rows) * tokens * 2);
    upload(xb, x); upload(wb, weights);
    Weight w;
    w.qtype = QType::BF16; w.layout = QuantLayout::Contiguous; w.qdata = wb.p;
    w.n = n; w.k = k; w.ndim = 2;
    w.shape[0] = w.padded_shape[0] = n;
    w.shape[1] = w.padded_shape[1] = k;
    w.payload = static_cast<std::uint8_t*>(wb.p); w.payload_bytes = weights.size() * 2;
    Tensor tx(xb.p, DType::BF16, {k, tokens}), to(out.p, DType::BF16, {rows, tokens});
    const auto capacity = swiglu
        ? ops::linear_swiglu_workspace_capacity_bytes(QType::BF16, n, k, tokens, tokens) : 0;
    WorkspaceArena workspace(std::max<std::size_t>(1, capacity));
    if (swiglu) ops::linear_swiglu(tx, w, to, workspace, nullptr);
    else ops::linear(tx, w, to, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> actual(std::size_t(rows) * tokens);
    out.copy_to_host(actual.data(), actual.size() * 2);
    Error error;
    for (auto element : actual) error.finite = error.finite && std::isfinite(value(element));
    // Full output qualification for the hot interval; sample representative rows
    // in larger extents while retaining complete independent K reductions for every column.
    const bool full = tokens <= 8 && n < 65536;
    const int sampled_rows = full ? rows : std::min(rows, 97);
    for (int t = 0; t < tokens; ++t) for (int sample = 0; sample < sampled_rows; ++sample) {
        const int row = full ? sample : sample * (rows - 1) / (sampled_rows - 1);
        double gate = 0, up = 0;
        for (int column = 0; column < k; ++column) {
            const double activation = value(x[t * k + column]);
            gate += activation * value(weights[std::size_t(row) * k + column]);
            if (swiglu) up += activation * value(weights[std::size_t(row + rows) * k + column]);
        }
        const double expected = swiglu ? gate / (1 + std::exp(-gate)) * up : gate;
        error.add(value(actual[t * rows + row]), expected);
    }
    return error.report(std::string(swiglu ? "swiglu" : "linear") + "_N" +
        std::to_string(n) + "_K" + std::to_string(k) + "_T" + std::to_string(tokens), swiglu);
}

bool gating_projection(int tokens) {
    constexpr int k = 1024, heads = 16;
    const auto x = random_values(k * tokens), weights = random_values(2 * heads * k);
    std::vector<float> alog(heads), bias(heads);
    for (int h = 0; h < heads; ++h) {
        alog[h] = normal(rng);
        bias[h] = normal(rng);
    }
    DeviceBuffer xb(x.size() * 2), wb(weights.size() * 2);
    DeviceBuffer ab(heads * 4), db(heads * 4), gb(heads * tokens * 4), bb(gb.bytes);
    upload(xb, x); upload(wb, weights); upload(ab, alog); upload(db, bias);
    Weight w;
    w.qtype = QType::BF16; w.layout = QuantLayout::Contiguous; w.qdata = wb.p;
    w.n = 2 * heads; w.k = k;
    Tensor tx(xb.p, DType::BF16, {k, tokens}), ta(ab.p, DType::FP32, {heads});
    Tensor td(db.p, DType::FP32, {heads}), tg(gb.p, DType::FP32, {heads, tokens});
    Tensor tb(bb.p, DType::FP32, {heads, tokens});
    const auto capacity = ops::gdn_gating_proj_workspace_capacity_bytes(heads, k, tokens, tokens);
    if (capacity != 0) throw std::runtime_error("BF16 gating should require no workspace");
    WorkspaceArena workspace(1);
    ops::gdn_gating_proj(tx, w, ta, td, workspace, tg, tb, {nullptr, 70});
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> g(heads * tokens), beta(g.size());
    gb.copy_to_host(g.data(), g.size() * 4); bb.copy_to_host(beta.data(), beta.size() * 4);
    Error error;
    for (int t = 0; t < tokens; ++t) for (int h = 0; h < heads; ++h) {
        double a = 0, b = 0;
        for (int j = 0; j < k; ++j) {
            const double activation = value(x[t * k + j]);
            a += activation * value(weights[h * k + j]);
            b += activation * value(weights[(heads + h) * k + j]);
        }
        const double z = a + bias[h];
        const double softplus = std::max(z, 0.0) + std::log1p(std::exp(-std::abs(z)));
        error.add(g[t * heads + h], -std::exp(double(alog[h])) * softplus);
        error.add(beta[t * heads + h], 1.0 / (1.0 + std::exp(-b)));
    }
    return error.report("gdn_gating_T" + std::to_string(tokens));
}

bool vision_attention(const std::vector<int>& lengths, bool uniform, bool padded) {
    constexpr int d = 64, heads = 12;
    std::vector<int> offsets{0};
    for (const int length : lengths) offsets.push_back(offsets.back() + length);
    const int tokens = offsets.back();
    const int stride = d * heads + (padded ? 16 : 0);
    const auto q = random_values(std::size_t(stride) * tokens);
    const auto k = random_values(q.size()), v = random_values(q.size());
    DeviceBuffer qb(q.size() * 2), kb(k.size() * 2), vb(v.size() * 2);
    DeviceBuffer out(d * heads * tokens * 2), boundaries(offsets.size() * 4);
    upload(qb, q); upload(kb, k); upload(vb, v); upload(boundaries, offsets);
    CUDA_CHECK(cudaMemset(out.p, 0xff, out.bytes));
    Tensor tq(qb.p, DType::BF16, {d, heads, tokens});
    Tensor tk(kb.p, DType::BF16, {d, heads, tokens});
    Tensor tv(vb.p, DType::BF16, {d, heads, tokens});
    tq.nb[2] = tk.nb[2] = tv.nb[2] = stride * 2;
    Tensor to(out.p, DType::BF16, {d, heads, tokens});
    const ops::AttentionHeadGeometry geometry{d, heads, heads};
    const auto capacity = ops::packed_softmax_attention_workspace_capacity_bytes(
        geometry, tokens, tokens, lengths.size(), lengths.size());
    WorkspaceArena workspace(capacity + 256);
    if (uniform) {
        ops::packed_softmax_attention(tq, tk, tv, geometry, 0.125f, lengths.front(), to, nullptr);
    } else {
        ops::packed_softmax_attention(tq, tk, tv, geometry, 0.125f,
            Tensor(boundaries.p, DType::I32, {int(offsets.size())}), workspace, to, nullptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<__nv_bfloat16> actual(d * heads * tokens);
    out.copy_to_host(actual.data(), actual.size() * 2);
    Error error;
    for (auto x : actual) error.finite = error.finite && std::isfinite(value(x));
    // Complete FP64 reductions for the first, middle and last query of every segment.
    for (std::size_t segment = 0; segment < lengths.size(); ++segment) {
        const int begin = offsets[segment], end = offsets[segment + 1];
        for (int i : {begin, begin + (end - begin) / 2, end - 1}) for (int h = 0; h < heads; ++h) {
            std::vector<double> scores(end - begin);
            double maximum = -INFINITY;
            for (int j = begin; j < end; ++j) {
                double dot = 0.0;
                for (int z = 0; z < d; ++z) dot += value(q[i * stride + h * d + z]) * value(k[j * stride + h * d + z]);
                scores[j - begin] = dot * 0.125;
                maximum = std::max(maximum, scores[j - begin]);
            }
            double total = 0.0;
            for (auto& score : scores) { score = std::exp(score - maximum); total += score; }
            for (int z = 0; z < d; ++z) {
                double expected = 0.0;
                for (int j = begin; j < end; ++j)
                    expected += (scores[j - begin] / total) * value(v[j * stride + h * d + z]);
                error.add(value(actual[(i * heads + h) * d + z]), expected);
            }
        }
    }
    return error.report(std::string(uniform ? "vision_uniform" : "vision_packed")
        + "_T" + std::to_string(tokens) + "_S" + std::to_string(lengths.size())
        + (padded ? "_strided" : ""));
}
} // namespace

int main() {
    try {
        bool pass = true;
        for (int tokens : {1, 2, 3, 4, 5, 7, 8, 9, 17, 64, 65}) {
            for (int k : {2048, 3072, 3584}) pass = projection(1024, k, tokens, true) && pass;
            pass = projection(8192, 1024, tokens, false) && pass;
            pass = projection(5120, 1024, tokens, false) && pass;
        }
        for (int tokens : {1, 2, 3, 4, 5, 7, 8, 9, 17, 64, 65}) {
            pass = dense_projection(7168, 1024, tokens, true) && pass;
            pass = dense_projection(14336, 5120, tokens, true) && pass;
            pass = dense_projection(768, 768, tokens, false) && pass;
        }
        for (const auto shape : {std::pair{768, 1536}, std::pair{768, 3072},
             std::pair{2304, 768}, std::pair{3072, 768}, std::pair{3072, 3072},
             std::pair{7168, 1024}, std::pair{248320, 1024}})
            for (int tokens : {1, 3, 8, 9})
                pass = dense_projection(shape.first, shape.second, tokens, false) && pass;
        for (int tokens : {1, 2, 4, 65}) pass = gating_projection(tokens) && pass;
        for (int length : {1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 197, 7168, 9216})
            pass = vision_attention({length}, true, false) && pass;
        pass = vision_attention({17, 17, 17}, true, true) && pass;
        pass = vision_attention({1, 15, 17, 65, 197}, false, true) && pass;
        std::cout << "{\"oracle\":\"Independent naive FP64 over represented BF16 inputs\","
                  << "\"pass\":" << (pass ? "true" : "false") << "}\n";
        return pass ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
}
