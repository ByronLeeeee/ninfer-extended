#include "ninfer/ops/embedding_pool.h"
#include "core/device.h"
#include <cuda_bf16.h>
#include <stdexcept>

namespace ninfer::ops {
namespace {
__global__ void output_kernel(const __nv_bfloat16* input, float* output,
                             int width, int dimensions, bool normalize) {
    const int row = blockIdx.x;
    if (!normalize) {
        for (int i=threadIdx.x; i<dimensions; i+=blockDim.x)
            output[static_cast<long long>(row)*dimensions+i]=
                __bfloat162float(input[static_cast<long long>(row)*width+i]);
        return;
    }
    float maximum = 0;
    for (int i=threadIdx.x; i<dimensions; i+=blockDim.x) {
        const float x = __bfloat162float(input[static_cast<long long>(row)*width+i]);
        maximum=fmaxf(maximum,fabsf(x));
    }
    for(int offset=16;offset;offset>>=1) maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,offset));
    __shared__ float warps[8];
    if ((threadIdx.x&31)==0) warps[threadIdx.x/32]=maximum;
    __syncthreads();
    if (threadIdx.x<32) {
        maximum=threadIdx.x<8?warps[threadIdx.x]:0;
        for(int offset=16;offset;offset>>=1) maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,offset));
        if(threadIdx.x==0)warps[0]=maximum;
    }
    __syncthreads();maximum=warps[0];
    const float divisor=maximum>0?maximum:1;
    float sum=0;
    for(int i=threadIdx.x;i<dimensions;i+=blockDim.x){
        const float x=__fdiv_rn(__bfloat162float(input[static_cast<long long>(row)*width+i]),divisor);
        sum+=x*x;
    }
    for(int offset=16;offset;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
    if((threadIdx.x&31)==0)warps[threadIdx.x/32]=sum;
    __syncthreads();
    if(threadIdx.x<32){
        sum=threadIdx.x<8?warps[threadIdx.x]:0;
        for(int offset=16;offset;offset>>=1) sum += __shfl_down_sync(0xffffffff,sum,offset);
        if(threadIdx.x==0){
            const bool epsilon_clamped=maximum<1e-12f&&maximum*sqrtf(sum)<1e-12f;
            warps[0]=epsilon_clamped?1e12f:rsqrtf(sum);
            warps[1]=epsilon_clamped?1:divisor;
        }
    }
    __syncthreads();
    const float scale=warps[0],denominator=warps[1];
    for (int i=threadIdx.x; i<dimensions; i+=blockDim.x)
        output[static_cast<long long>(row)*dimensions+i]=
            __fdiv_rn(__bfloat162float(input[static_cast<long long>(row)*width+i]),denominator)*scale;
}
}
void bf16_embedding_output(const void* input,float* output,int rows,int width,
                           int dimensions,bool normalize,cudaStream_t stream) {
    if(!input||!output||input==output||rows<1||dimensions<1||width<dimensions)
        throw std::invalid_argument("embedding output: invalid nonaliasing geometry");
    output_kernel<<<rows,256,0,stream>>>(static_cast<const __nv_bfloat16*>(input),output,width,dimensions,normalize);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops
