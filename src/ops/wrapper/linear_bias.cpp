#include "ninfer/ops/linear_bias.h"
#include "ops/linear/bf16/bf16_projection.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {
bool overlaps(const void* a, std::size_t as, const void* b, std::size_t bs) {
    const auto ab = reinterpret_cast<std::uintptr_t>(a);
    const auto bb = reinterpret_cast<std::uintptr_t>(b);
    return ab < bb + bs && bb < ab + as;
}
void require_operands(const Tensor& x, const Weight& w, const Tensor& bias,
                      const Tensor& out) {
    if (w.qtype != QType::BF16 || w.layout != QuantLayout::Contiguous || !w.qdata ||
        x.dtype != DType::BF16 || !x.data || !x.is_contiguous() || x.ne[0] != w.k ||
        x.ne[1] <= 0 || x.ne[2] != 1 || x.ne[3] != 1 ||
        bias.dtype != DType::BF16 || !bias.data || !bias.is_contiguous() ||
        bias.ne[0] != w.n || bias.ne[1] != 1 || bias.ne[2] != 1 || bias.ne[3] != 1 ||
        out.dtype != DType::BF16 || !out.data || !out.is_contiguous() ||
        out.ne[0] != w.n || out.ne[1] != x.ne[1] || out.ne[2] != 1 || out.ne[3] != 1) {
        throw std::invalid_argument("linear_bias: invalid contiguous BF16 operands");
    }
    if (overlaps(out.data, out.bytes(), x.data, x.bytes()) ||
        overlaps(out.data, out.bytes(), bias.data, bias.bytes()) ||
        overlaps(out.data, out.bytes(), w.qdata, std::size_t(w.n) * w.k * 2)) {
        throw std::invalid_argument("linear_bias: output overlaps an input");
    }
}
} // namespace

void linear_bias_add(const Tensor& x, const Weight& w, const Tensor& bias,
                     Tensor& residual, cudaStream_t stream) {
    require_operands(x, w, bias, residual);
    if (w.n != 768 || (w.k != 768 && w.k != 1536 && w.k != 3072)) {
        throw std::invalid_argument("linear_bias_add: unsupported matrix geometry");
    }
    detail::bf16_projection_bias_add(x, w, bias, residual, stream);
}

void linear_bias_gelu(const Tensor& x, const Weight& w, const Tensor& bias,
                      GeluMode mode, Tensor& output, cudaStream_t stream) {
    require_operands(x, w, bias, output);
    if (w.n != 3072 || (w.k != 768 && w.k != 3072) ||
        (mode != GeluMode::Exact && mode != GeluMode::Tanh)) {
        throw std::invalid_argument("linear_bias_gelu: unsupported geometry or GELU mode");
    }
    detail::bf16_projection_bias_gelu(x, w, bias, mode, output, stream);
}
} // namespace ninfer::ops
