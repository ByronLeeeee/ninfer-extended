#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear_add.h"
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

// One A16 criterion for all shapes/routes. Every reported output must be finite.
struct Error {
    double squared_error = 0.0;
    double squared_reference = 0.0;
    double maximum_error = 0.0;
    std::size_t count = 0;
    bool finite = true;

    void add(double actual, double expected) {
        finite = finite && std::isfinite(actual) && std::isfinite(expected);
        const double difference = actual - expected;
        squared_error += difference * difference;
        squared_reference += expected * expected;
        maximum_error = std::max(maximum_error, std::abs(difference));
        ++count;
    }

    bool report(const std::string& name) const {
        const double rms = std::sqrt(squared_reference / count);
        const double nrms = std::sqrt(squared_error / squared_reference);
        const double peak = maximum_error / rms;
        const bool pass = finite && std::isfinite(nrms) && nrms < 0.006 && peak < 0.045;
        std::cout << "{\"case\":\"" << name << "\",\"nrms\":" << nrms
                  << ",\"max_error_over_reference_rms\":" << peak
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
    WorkspaceArena workspace(std::size_t(n) * tokens * 2 + 256);
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
        for (int tokens : {1, 2, 4}) {
            for (int k : {2048, 3072, 3584}) pass = projection(1024, k, tokens, true) && pass;
            pass = projection(8192, 1024, tokens, false) && pass;
            pass = projection(5120, 1024, tokens, false) && pass;
        }
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
