#include "ops/linear/bf16/bf16_projection.h"

#include "core/device.h"
#include "ops/linear/bf16/bf16_launch.cuh"
#include "ops/linear_swiglu/bf16/bf16_swiglu_small.cuh"
#include "ops/kernel/gelu.cuh"

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
using NarrowRowsMma = Bf16MmaSchedule<32, 64, 64, 16, 32, 2, 2, Cache::cg, Cache::cg,
    Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
using TripleMma = Bf16MmaSchedule<64, 64, 64, 32, 32, 3, 2, Cache::cg, Cache::cg,
    Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
using CompactMma = Bf16MmaSchedule<32, 32, 64, 16, 16, 2, 2, Cache::cg, Cache::cg,
    Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
using ShortColumnsMma = Bf16MmaSchedule<64, 32, 64, 32, 16, 2, 2, Cache::cg, Cache::cg,
    Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
using PairedCompactMma = Bf16MmaSchedule<32, 32, 64, 32, 16, 2, 2, Cache::cg, Cache::cg,
    Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
using PairedTailMma = Bf16MmaSchedule<128, 32, 64, 64, 16, 2, 1, Cache::cg, Cache::cg,
    Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;

struct ContiguousOutput {
    __nv_bfloat16* data;
    int rows;
    __device__ __forceinline__ ContiguousOutput tile(int) const { return *this; }
    __device__ __forceinline__ void store(int row, int token, float projection) const {
        data[static_cast<std::int64_t>(token) * rows + row] = __float2bfloat16_rn(projection);
    }
};

struct FloatProjectionOutput {
    float* data;
    int rows;
    __device__ __forceinline__ FloatProjectionOutput tile(int) const { return *this; }
    __device__ __forceinline__ void store(int row, int token, float projection) const {
        data[static_cast<std::int64_t>(token) * rows + row] = projection;
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

struct BiasResidualOutput {
    __nv_bfloat16* data;
    const __nv_bfloat16* bias;
    int rows;
    __device__ __forceinline__ BiasResidualOutput tile(int) const { return *this; }
    __device__ __forceinline__ void store(int row, int token, float projection) const {
        auto* destination = data + static_cast<std::int64_t>(token) * rows + row;
        const float projected = __bfloat162float(__float2bfloat16_rn(projection));
        const float biased = __bfloat162float(__float2bfloat16_rn(
            projected + __bfloat162float(bias[row])));
        *destination = __float2bfloat16_rn(biased + __bfloat162float(*destination));
    }
};

template <bool TanhApprox>
struct BiasGeluOutput {
    __nv_bfloat16* data;
    const __nv_bfloat16* bias;
    int rows;
    __device__ __forceinline__ BiasGeluOutput tile(int) const { return *this; }
    __device__ __forceinline__ void store(int row, int token, float projection) const {
        const float projected = __bfloat162float(__float2bfloat16_rn(projection));
        const float biased = __bfloat162float(__float2bfloat16_rn(
            projected + __bfloat162float(bias[row])));
        data[static_cast<std::int64_t>(token) * rows + row] =
            __float2bfloat16_rn(gelu_one<TanhApprox>(biased));
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
    if constexpr(n==1024&&(k==2048||k==3072)) {
        if(x.ne[1]<=128){mma_projection_variant<Geometry,CompactMma>(x,w,output,stream);return;}
        if constexpr(k==3072) if(x.ne[1]<=256){mma_projection_variant<Geometry,NarrowRowsMma>(x,w,output,stream);return;}
        const int tail=x.ne[1]%128;
        if(x.ne[1]>=1024&&(tail==0||tail>64)){mma_projection_variant<Geometry,WideMma>(x,w,output,stream);return;}
    } else if constexpr (n == 1024 && k == 3584) {
        if (x.ne[1] <= 512) {
            mma_projection_variant<Geometry, NarrowRowsMma>(x, w, output, stream);
            return;
        }
        const int tail = x.ne[1] % WideMma::kBlockCols;
        if (x.ne[1] >= 1024 && (tail == 0 || tail > 64)) {
            mma_projection_variant<Geometry, WideMma>(x, w, output, stream);
            return;
        }
    } else if constexpr(n==4096&&k==1024) {
        const int short_tail=x.ne[1]%64,wide_tail=x.ne[1]%128;
        if(x.ne[1]<=128&&short_tail>0&&short_tail<=32){mma_projection_variant<Geometry,ShortColumnsMma>(x,w,output,stream);return;}
        if(x.ne[1]>=256&&(wide_tail==0||wide_tail>64)){mma_projection_variant<Geometry,WideMma>(x,w,output,stream);return;}
    } else if constexpr(n==6144&&k==1024) {
        if(x.ne[1]<=16){mma_projection_variant<Geometry,CompactMma>(x,w,output,stream);return;}
        if(x.ne[1]<=64){mma_projection_variant<Geometry,ShortColumnsMma>(x,w,output,stream);return;}
        const int tail=x.ne[1]%128;
        if(x.ne[1]>=1024&&(tail==0||tail>64)){mma_projection_variant<Geometry,WideMma>(x,w,output,stream);return;}
    }
    if constexpr (n==2048&&(k==2048||k==6144)) {
        // Short column extents need more row CTAs to cover the SMs. Keep the
        // established tile once longer extents provide sufficient parallelism.
        if(x.ne[1]<=256){mma_projection_variant<Geometry,NarrowRowsMma>(x,w,output,stream);return;}
    } else if constexpr(n==4096&&k==2048) {
        if(x.ne[1]>=65&&x.ne[1]<=128){mma_projection_variant<Geometry,NarrowRowsMma>(x,w,output,stream);return;}
        if(x.ne[1]>=193&&x.ne[1]<=256){mma_projection_variant<Geometry,WideMma>(x,w,output,stream);return;}
    } else if constexpr(n==12288&&k==2048) {
        if(x.ne[1]>=193&&x.ne[1]<=256){
            using NarrowWideMma=Bf16MmaSchedule<32,128,64,16,64,2,1,
                Cache::cg,Cache::cg,Bf16MmaFragmentPipeline::PingPong,Bf16MmaRaster::TokenFast>;
            mma_projection_variant<Geometry,NarrowWideMma>(x,w,output,stream);return;
        }
        if(x.ne[1]>=65&&x.ne[1]<=128){
            mma_projection_variant<Geometry,WideMma>(x,w,output,stream);return;
        }
    } else if constexpr (n == 8192 && k == 1024) {
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
    SHAPE(8192, 1024) SHAPE(5120, 1024) SHAPE(7168, 1024) SHAPE(6144, 1024) SHAPE(248320, 1024)
    SHAPE(1024, 2048) SHAPE(1024, 3584) SHAPE(1024, 3072)
    SHAPE(768, 1536) SHAPE(768, 768) SHAPE(768, 3072)
    SHAPE(2304, 768) SHAPE(3072, 768) SHAPE(3072, 3072)
    SHAPE(2048, 2048) SHAPE(4096, 2048) SHAPE(12288, 2048) SHAPE(2048, 6144) SHAPE(151936, 2048)
    SHAPE(1024, 1024) SHAPE(3072, 1024) SHAPE(4096, 1024) SHAPE(1024, 4096) SHAPE(1024, 7680) SHAPE(2048, 1024)
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

__global__ void float_projection_swiglu_kernel(const float* x, __nv_bfloat16* y,
                                             int rows, int tokens) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * tokens) {
        const int row = i % rows, t = i / rows;
        const float gate = x[static_cast<std::int64_t>(t) * 2 * rows + row];
        const float up = x[static_cast<std::int64_t>(t) * 2 * rows + rows + row];
        y[i] = __float2bfloat16_rn((gate / (1.f + expf(-gate))) * up);
    }
}

template <int N, int K, int Capacity>
void small_swiglu(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    using Geometry = Bf16Geometry<N, K>;
    bf16_swiglu_simt_kernel<Geometry, Capacity, SmallT, !(N == 6144 && K == 1024)>
        <<<N / (SmallT::kRowsPerWarp * SmallT::kWarpsPerCta), SmallT::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(w.qdata), static_cast<__nv_bfloat16*>(out.data), x.ne[1]);
}

template <int N, int K>
void swiglu(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws, cudaStream_t stream,
             std::int32_t multiprocessor_count) {
    using Geometry = Bf16Geometry<N, K>;
    if (x.ne[1] == 1) bf16_swiglu_gemv_kernel<Geometry, SwiGluGemv, !(N == 6144 && K == 1024)>
        <<<N / (SwiGluGemv::kRowsPerWarp * SwiGluGemv::kWarpsPerCta), SwiGluGemv::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(w.qdata), static_cast<__nv_bfloat16*>(out.data));
    else if (x.ne[1] == 2) small_swiglu<N, K, 2>(x, w, out, stream);
    else if (x.ne[1] == 3) small_swiglu<N, K, 3>(x, w, out, stream);
    else if (x.ne[1] == 4) small_swiglu<N, K, 4>(x, w, out, stream);
    else if (x.ne[1] <= bf16_swiglu_small_max_tokens(N, K)) small_swiglu<N, K, 8>(x, w, out, stream);
    else if constexpr ((N == 6144 || N == 7168) && K == 1024) {
        // Gate/up projection and activation share the accumulator registers.
        // No gate/up intermediate is observable or materialized in global memory.
        auto launch = [&]<class Schedule>() {
            const int blocks = N / Schedule::kBlockRows * div_up(x.ne[1], Schedule::kBlockCols);
            const ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), N / 2};
            if (x.ne[1] % Schedule::kBlockCols == 0)
                bf16_gemm_mma_kernel<Geometry, Schedule, true, ContiguousOutput, true, N == 7168>
                    <<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
                        static_cast<const __nv_bfloat16*>(x.data),
                        static_cast<const __nv_bfloat16*>(w.qdata), output, x.ne[1]);
            else bf16_gemm_mma_kernel<Geometry, Schedule, false, ContiguousOutput, true, N == 7168>
                    <<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
                        static_cast<const __nv_bfloat16*>(x.data),
                        static_cast<const __nv_bfloat16*>(w.qdata), output, x.ne[1]);
        };
        if constexpr (N == 7168) {
            const int tail = x.ne[1] % WideMma::kBlockCols;
            const auto wide_blocks = N / WideMma::kBlockRows *
                                     div_up(x.ne[1], WideMma::kBlockCols);
            // A deeper copy pipeline pays off for a partial column tile when
            // the wide route would provide fewer than three CTA waves.
            if (x.ne[1] > 256 && x.ne[1] <= 512 && tail > 0 &&
                tail <= Mma::kBlockCols && multiprocessor_count > 0 &&
                wide_blocks < std::int64_t(multiprocessor_count) * 3) {
                launch.template operator()<TripleMma>();
                CUDA_CHECK(cudaGetLastError());
                return;
            }
            if ((x.ne[1] >= 65 && x.ne[1] <= 128) ||
                (x.ne[1] >= 256 && (tail == 0 || tail > 64))) {
                launch.template operator()<WideMma>();
                CUDA_CHECK(cudaGetLastError());
                return;
            }
        }
        if (x.ne[1] < 32) launch.template operator()<PairedCompactMma>();
        else if (x.ne[1] <= 64) launch.template operator()<ShortColumnsMma>();
        // Narrow columns avoid unused tail MMA work, but the wider row tile needs
        // enough CTA waves to amortize its register pressure. Device facts come
        // from the caller; stream-only callers retain the established schedule.
        else if (N == 6144 && x.ne[1] > 2 * Mma::kBlockCols &&
                 x.ne[1] <= 2 * Mma::kBlockCols + PairedTailMma::kBlockCols &&
                 multiprocessor_count > 0 &&
                 N / PairedTailMma::kBlockRows * div_up(x.ne[1], PairedTailMma::kBlockCols) >=
                     std::int64_t(multiprocessor_count) * 3)
            launch.template operator()<PairedTailMma>();
        else launch.template operator()<Mma>();
    }
    else {
        auto scope = ws.scope();
        if constexpr (N == 12288 && K == 2048) {
            // Projection staging is private to this fused Op. Retain its FP32
            // accumulator through SiLU/multiply and round only the public output.
            auto tmp = ws.alloc(DType::FP32, {N, x.ne[1]}, 256);
            mma_projection<Geometry>(x, w, FloatProjectionOutput{static_cast<float*>(tmp.data), N}, stream);
            float_projection_swiglu_kernel<<<div_up(N / 2 * x.ne[1], 256), 256, 0, stream>>>(
                static_cast<const float*>(tmp.data), static_cast<__nv_bfloat16*>(out.data), N / 2, x.ne[1]);
        } else {
            auto tmp = ws.alloc(DType::BF16, {N, x.ne[1]}, 256);
            mma_projection<Geometry>(x, w, ContiguousOutput{static_cast<__nv_bfloat16*>(tmp.data), N}, stream);
            swiglu_kernel<<<div_up(N / 2 * x.ne[1], 256), 256, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(tmp.data), static_cast<__nv_bfloat16*>(out.data),
                N / 2, x.ne[1]);
        }
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
    return (k == 1024 && (n == 8192 || n == 5120 || n == 7168 || n == 6144 || n == 248320)) ||
        (n == 1024 && (k == 2048 || k == 3584 || k == 3072)) ||
        (n == 768 && (k == 1536 || k == 768 || k == 3072)) ||
        (n == 2304 && k == 768) || (n == 3072 && (k == 768 || k == 3072)) ||
        (k == 2048 && (n == 2048 || n == 4096 || n == 12288 || n == 151936)) ||
        (n == 2048 && k == 6144) ||
        (k == 1024 && (n == 1024 || n == 3072 || n == 4096 || n == 2048)) ||
        (n == 1024 && (k == 4096 || k == 7680));
}

bool bf16_swiglu_shape(int n, int k) {
    return ((n == 7168 || n == 6144) && k == 1024) || (n == 14336 && k == 5120) || (n == 12288 && k == 2048);
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

void bf16_projection_bias_add(const Tensor& x, const Weight& w, const Tensor& bias,
                              Tensor& out, cudaStream_t stream) {
    const BiasResidualOutput output{static_cast<__nv_bfloat16*>(out.data),
        static_cast<const __nv_bfloat16*>(bias.data), w.n};
    if (w.k == 768) projection<768, 768>(x, w, output, stream);
    else if (w.k == 1536) projection<768, 1536>(x, w, output, stream);
    else projection<768, 3072>(x, w, output, stream);
}

void bf16_projection_bias_gelu(const Tensor& x, const Weight& w, const Tensor& bias,
                               GeluMode mode, Tensor& out, cudaStream_t stream) {
    const auto launch = [&]<bool TanhApprox>() {
        const BiasGeluOutput<TanhApprox> output{static_cast<__nv_bfloat16*>(out.data),
            static_cast<const __nv_bfloat16*>(bias.data), w.n};
        if (w.k == 768) projection<3072, 768>(x, w, output, stream);
        else projection<3072, 3072>(x, w, output, stream);
    };
    if (mode == GeluMode::Tanh) launch.template operator()<true>();
    else launch.template operator()<false>();
}

void bf16_projection_swiglu(const Tensor& x, const Weight& w, Tensor& out,
                            WorkspaceArena& ws, cudaStream_t stream,
                            std::int32_t multiprocessor_count) {
    require_input(x, w); require_output(x, out, w.n / 2);
    if (w.n == 7168 && w.k == 1024) swiglu<7168, 1024>(x, w, out, ws, stream, multiprocessor_count);
    else if (w.n == 6144 && w.k == 1024) swiglu<6144, 1024>(x, w, out, ws, stream, multiprocessor_count);
    else if (w.n == 14336 && w.k == 5120) swiglu<14336, 5120>(x, w, out, ws, stream, multiprocessor_count);
    else if (w.n == 12288 && w.k == 2048) swiglu<12288, 2048>(x, w, out, ws, stream, multiprocessor_count);
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
