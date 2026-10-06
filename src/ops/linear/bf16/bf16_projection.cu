#include "ops/linear/bf16/bf16_projection.h"

#include "core/device.h"
#include "ops/linear/bf16/bf16_launch.cuh"
#include "ops/linear_swiglu/bf16/bf16_swiglu_small.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
using Gemv = Bf16GemvSchedule<4, 1, 8, 8, 4, Bf16ActivationAccess::Direct,
    Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 1, 1, 2>;
using SwiGluGemv = Bf16GemvSchedule<4, 1, 2, 8, 4, Bf16ActivationAccess::Direct,
    Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 1, 1, 2>;
using SmallT = Bf16SimtSchedule<4, 1, 2, 8, 1, 4, Bf16SimtActivationAccess::WarpPacked,
    Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 2>;
using Mma = Bf16MmaSchedule<64, 64, 64, 32, 32, 2, 2, Cache::cg, Cache::cg,
    Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
using WideMma = Bf16MmaSchedule<64, 128, 64, 32, 64, 2, 1, Cache::cg, Cache::cg,
    Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;

struct ContiguousOutput {
    __nv_bfloat16* data;
    int rows;
    __device__ __forceinline__ ContiguousOutput tile(int) const { return *this; }
    __device__ __forceinline__ void store(int row, int token, float projection) const {
        data[static_cast<std::int64_t>(token) * rows + row] = __float2bfloat16_rn(projection);
    }
};

struct ResidualOutput {
    __nv_bfloat16* data;
    int rows;
    __device__ __forceinline__ ResidualOutput tile(int) const { return *this; }
    __device__ __forceinline__ void store(int row, int token, float projection) const {
        auto* destination = data + static_cast<std::int64_t>(token) * rows + row;
        projection = __bfloat162float(__float2bfloat16_rn(projection));
        *destination = __float2bfloat16_rn(projection + __bfloat162float(*destination));
    }
};

template <int Planes>
struct SplitOutput {
    __nv_bfloat16* data[Planes];
    int rows[Planes];
    __device__ __forceinline__ SplitOutput tile(int) const { return *this; }
    __device__ __forceinline__ void store(int row, int token, float projection) const {
#pragma unroll
        for (int plane = 0; plane < Planes; ++plane) {
            if (row < rows[plane]) {
                data[plane][static_cast<std::int64_t>(token) * rows[plane] + row] =
                    __float2bfloat16_rn(projection);
                return;
            }
            row -= rows[plane];
        }
    }
};

template <class Output>
struct GemvOutput {
    Output output;
    __device__ __forceinline__ void store(int row, __nv_bfloat16 projection) const {
        output.store(row, 0, __bfloat162float(projection));
    }
};

template <class Geometry, int Capacity, class Output>
void small_projection(const Tensor& x, const Weight& w, Output output, cudaStream_t stream) {
    bf16_simt_kernel<Geometry, Capacity, SmallT, Output, true>
        <<<Geometry::kOutputRows / SmallT::kRowsPerCta, SmallT::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(w.qdata), output, x.ne[1]);
}

template <class Geometry, class Schedule, class Output>
void mma_projection_variant(const Tensor& x, const Weight& w, Output output, cudaStream_t stream) {
    const int blocks = Geometry::kOutputRows / Schedule::kBlockRows * div_up(x.ne[1], Schedule::kBlockCols);
    if (x.ne[1] % Schedule::kBlockCols == 0)
        bf16_gemm_mma_kernel<Geometry, Schedule, true>
            <<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const __nv_bfloat16*>(w.qdata), output, x.ne[1]);
    else bf16_gemm_mma_kernel<Geometry, Schedule, false>
        <<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(w.qdata), output, x.ne[1]);
}

template <class Geometry, class Output>
void mma_projection(const Tensor& x, const Weight& w, Output output, cudaStream_t stream) {
    constexpr int n = Geometry::kOutputRows, k = Geometry::kInputRows;
    if constexpr (n == 8192 && k == 1024) {
        // These columns fill the same padded extent with half as many weight tiles.
        // Retain the narrower tile when a short final column tile would add MMA work.
        const int tail = x.ne[1] % WideMma::kBlockCols;
        if (x.ne[1] >= 65 && (tail == 0 || tail > Mma::kBlockCols)) {
            mma_projection_variant<Geometry, WideMma>(x, w, output, stream);
            return;
        }
    } else if constexpr (n == 3072 && k == 768) {
        // Larger column extents provide enough CTAs for the wider accumulator tile.
        if (x.ne[1] >= 2048) {
            mma_projection_variant<Geometry, WideMma>(x, w, output, stream);
            return;
        }
    } else if constexpr ((n == 768 && k == 3072) || (n == 2304 && k == 768)) {
        if (x.ne[1] >= 4096) {
            mma_projection_variant<Geometry, WideMma>(x, w, output, stream);
            return;
        }
    }
    mma_projection_variant<Geometry, Mma>(x, w, output, stream);
}

template <int N, int K, class Output>
void projection(const Tensor& x, const Weight& w, Output output, cudaStream_t stream) {
    using Geometry = Bf16Geometry<N, K>;
    if (x.ne[1] == 1) {
        bf16_gemv_kernel<Geometry, Gemv>
            <<<N / Gemv::kRowsPerCta, Gemv::kThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const __nv_bfloat16*>(w.qdata), GemvOutput<Output>{output});
    } else {
        if constexpr (N < 65536) {
            if (x.ne[1] == 2) small_projection<Geometry, 2>(x, w, output, stream);
            else if (x.ne[1] == 3) small_projection<Geometry, 3>(x, w, output, stream);
            else if (x.ne[1] == 4) small_projection<Geometry, 4>(x, w, output, stream);
            else if (x.ne[1] <= 8) small_projection<Geometry, 8>(x, w, output, stream);
            else mma_projection<Geometry>(x, w, output, stream);
        } else {
            // Vocabulary-sized projections showed no useful small-T gain on RTX 5070 Ti.
            mma_projection<Geometry>(x, w, output, stream);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class Output>
void dispatch_projection(const Tensor& x, const Weight& w, Output output, cudaStream_t stream) {
#define SHAPE(N, K) if (w.n == N && w.k == K) { projection<N, K>(x, w, output, stream); return; }
    SHAPE(8192, 1024) SHAPE(5120, 1024) SHAPE(7168, 1024) SHAPE(248320, 1024)
    SHAPE(1024, 2048) SHAPE(1024, 3584) SHAPE(1024, 3072)
    SHAPE(768, 1536) SHAPE(768, 768) SHAPE(768, 3072)
    SHAPE(2304, 768) SHAPE(3072, 768) SHAPE(3072, 3072)
#undef SHAPE
    throw std::invalid_argument("BF16 projection: unregistered matrix shape");
}

void require_input(const Tensor& x, const Weight& w) {
    if (w.qtype != QType::BF16 || w.layout != QuantLayout::Contiguous || !w.qdata ||
        x.dtype != DType::BF16 || !x.data || !x.is_contiguous() || x.ne[0] != w.k ||
        x.ne[1] <= 0 || x.ne[2] != 1 || x.ne[3] != 1)
        throw std::invalid_argument("BF16 projection: invalid operands");
}

void require_output(const Tensor& x, const Tensor& out, int rows) {
    if (out.dtype != DType::BF16 || !out.data || !out.is_contiguous() || out.ne[0] != rows ||
        out.ne[1] != x.ne[1] || out.ne[2] != 1 || out.ne[3] != 1)
        throw std::invalid_argument("BF16 projection: invalid output");
}

__global__ void swiglu_kernel(const __nv_bfloat16* x, __nv_bfloat16* y, int rows, int tokens) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * tokens) {
        const int row = i % rows, t = i / rows;
        y[i] = bf16_swiglu_value(__bfloat162float(x[t * 2 * rows + row]),
                                 __bfloat162float(x[t * 2 * rows + rows + row]));
    }
}

template <int N, int K, int Capacity>
void small_swiglu(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    using Geometry = Bf16Geometry<N, K>;
    bf16_swiglu_simt_kernel<Geometry, Capacity, SmallT>
        <<<N / (SmallT::kRowsPerWarp * SmallT::kWarpsPerCta), SmallT::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(w.qdata), static_cast<__nv_bfloat16*>(out.data), x.ne[1]);
}

template <int N, int K>
void swiglu(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws, cudaStream_t stream) {
    using Geometry = Bf16Geometry<N, K>;
    if (x.ne[1] == 1) bf16_swiglu_gemv_kernel<Geometry, SwiGluGemv>
        <<<N / (SwiGluGemv::kRowsPerWarp * SwiGluGemv::kWarpsPerCta), SwiGluGemv::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(w.qdata), static_cast<__nv_bfloat16*>(out.data));
    else if (x.ne[1] == 2) small_swiglu<N, K, 2>(x, w, out, stream);
    else if (x.ne[1] == 3) small_swiglu<N, K, 3>(x, w, out, stream);
    else if (x.ne[1] == 4) small_swiglu<N, K, 4>(x, w, out, stream);
    else if (x.ne[1] <= bf16_swiglu_small_max_tokens(N, K)) small_swiglu<N, K, 8>(x, w, out, stream);
    else {
        auto scope = ws.scope();
        auto tmp = ws.alloc(DType::BF16, {N, x.ne[1]}, 256);
        mma_projection<Geometry>(x, w, ContiguousOutput{static_cast<__nv_bfloat16*>(tmp.data), N}, stream);
        swiglu_kernel<<<div_up(N / 2 * x.ne[1], 256), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(tmp.data), static_cast<__nv_bfloat16*>(out.data),
            N / 2, x.ne[1]);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <int InputRows, int Heads>
__global__ void gating_kernel(const __nv_bfloat16* x, const __nv_bfloat16* w,
    const float* alog, const float* dt, float* g, float* beta, int tokens) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int index = blockIdx.x * blockDim.x / 32 + warp, head = index % Heads;
    const int token = index / Heads;
    if (token >= tokens) return;
    float a = 0, b = 0;
    for (int k = lane; k < InputRows; k += 32) {
        const float value = __bfloat162float(x[token * InputRows + k]);
        a = fmaf(__bfloat162float(w[head * InputRows + k]), value, a);
        b = fmaf(__bfloat162float(w[(Heads + head) * InputRows + k]), value, b);
    }
    for (int delta = 16; delta; delta /= 2) {
        a += __shfl_down_sync(0xffffffff, a, delta);
        b += __shfl_down_sync(0xffffffff, b, delta);
    }
    if (lane == 0) {
        a = __bfloat162float(__float2bfloat16_rn(a));
        b = __bfloat162float(__float2bfloat16_rn(b));
        const float z = a + dt[head], softplus = z > 20 ? z : log1pf(expf(z));
        g[token * Heads + head] = -expf(alog[head]) * softplus;
        beta[token * Heads + head] = 1.f / (1.f + expf(-b));
    }
}
} // namespace

bool bf16_projection_shape(int n, int k) {
    return (k == 1024 && (n == 8192 || n == 5120 || n == 7168 || n == 248320)) ||
        (n == 1024 && (k == 2048 || k == 3584 || k == 3072)) ||
        (n == 768 && (k == 1536 || k == 768 || k == 3072)) ||
        (n == 2304 && k == 768) || (n == 3072 && (k == 768 || k == 3072));
}

bool bf16_swiglu_shape(int n, int k) {
    return (n == 7168 && k == 1024) || (n == 14336 && k == 5120);
}

int bf16_swiglu_small_max_tokens(int n, int k) {
    return n == 14336 && k == 5120 ? 8 : 4;
}

void bf16_projection_linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    require_input(x, w); require_output(x, out, w.n);
    dispatch_projection(x, w, ContiguousOutput{static_cast<__nv_bfloat16*>(out.data), w.n}, stream);
}

void bf16_projection_add(const Tensor& x, const Weight& w, Tensor& out,
                         WorkspaceArena&, cudaStream_t stream) {
    require_input(x, w); require_output(x, out, w.n);
    dispatch_projection(x, w, ResidualOutput{static_cast<__nv_bfloat16*>(out.data), w.n}, stream);
}

void bf16_projection_swiglu(const Tensor& x, const Weight& w, Tensor& out,
                            WorkspaceArena& ws, cudaStream_t stream) {
    require_input(x, w); require_output(x, out, w.n / 2);
    if (w.n == 7168 && w.k == 1024) swiglu<7168, 1024>(x, w, out, ws, stream);
    else if (w.n == 14336 && w.k == 5120) swiglu<14336, 5120>(x, w, out, ws, stream);
    else throw std::invalid_argument("BF16 SwiGLU: unregistered matrix shape");
}

void bf16_projection_split(const Tensor& x, const Weight& w, Tensor& first, Tensor& second,
                           cudaStream_t stream) {
    require_input(x, w); require_output(x, first, first.ne[0]); require_output(x, second, second.ne[0]);
    if (first.ne[0] + second.ne[0] != w.n) throw std::invalid_argument("BF16 split: invalid row partition");
    const SplitOutput<2> output{{static_cast<__nv_bfloat16*>(first.data),
        static_cast<__nv_bfloat16*>(second.data)}, {first.ne[0], second.ne[0]}};
    dispatch_projection(x, w, output, stream);
}

void bf16_projection_attn(const Tensor& x, const Weight& w, Tensor& q, Tensor& gate,
                          Tensor& k, Tensor& v, cudaStream_t stream) {
    require_input(x, w);
    for (const auto* plane : {&q, &k, &gate, &v}) require_output(x, *plane, plane->ne[0]);
    if (q.ne[0] + k.ne[0] + gate.ne[0] + v.ne[0] != w.n)
        throw std::invalid_argument("BF16 split: invalid four-plane partition");
    const SplitOutput<4> output{{static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data), static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data)}, {q.ne[0], k.ne[0], gate.ne[0], v.ne[0]}};
    dispatch_projection(x, w, output, stream);
}

void bf16_gating_projection(const Tensor& x, const Weight& w, const Tensor& alog, const Tensor& dt,
                            Tensor& g, Tensor& beta, cudaStream_t stream) {
    // Compile-time geometry retains constant division and loop optimization.
    if (w.n != 32 || w.k != 1024)
        throw std::invalid_argument("BF16 gating projection: unregistered matrix shape");
    gating_kernel<1024, 16><<<div_up(16 * x.ne[1], 8), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(w.qdata),
        static_cast<const float*>(alog.data), static_cast<const float*>(dt.data),
        static_cast<float*>(g.data), static_cast<float*>(beta.data), x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail
