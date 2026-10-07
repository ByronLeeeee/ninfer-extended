#include "ninfer/ops/dense_attention.h"
#include "ops/softmax_attention/dense/context/causal_prefill_kernel.cuh"
#include "core/device.h"
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
void causal_bf16_attention(const void* q,const void* k,const void* v,void* out,
    int tokens,int qh,int kh,int qs,int ks,cudaStream_t stream){
    const bool profile=(qh==16&&(kh==4||kh==8||kh==16))||(qh==8&&kh==8);
    const auto aligned=[](const void* p){return p&&reinterpret_cast<std::uintptr_t>(p)%16==0;};
    if(!profile||tokens<=0||qs<qh*128||ks<kh*128||qs%8||ks%8||
       !aligned(q)||!aligned(k)||!aligned(v)||!aligned(out))
        throw std::invalid_argument("causal_bf16_attention: unsupported geometry or layout");
    constexpr int Br=32,Bc=32;
    constexpr int bytes=(Br+2*Bc)*128*2;
    detail::causal_attention_flash_kernel<Br,Bc><<<dim3((tokens+Br-1)/Br,qh),Br*2,bytes,stream>>>(
        static_cast<const __nv_bfloat16*>(q),static_cast<const __nv_bfloat16*>(k),
        static_cast<const __nv_bfloat16*>(v),tokens,qh,kh,static_cast<__nv_bfloat16*>(out),
        1,128,qs,1,128,ks,1,128,ks);
    CUDA_CHECK(cudaGetLastError());
}
}
