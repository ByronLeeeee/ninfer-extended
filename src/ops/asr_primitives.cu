#include "ninfer/ops/bf16_transforms.h"
#include "ninfer/ops/dense_attention.h"
#include "core/device.h"
#include "ops/softmax_attention/dense/context/float_prefill_kernel.cuh"

#include <cuda_bf16.h>
#include <cmath>
#include <stdexcept>

namespace ninfer::ops {
namespace {
using BF = __nv_bfloat16;
__device__ float f(BF x) { return __bfloat162float(x); }
__device__ BF b(float x) { return __float2bfloat16_rn(x); }
__device__ float rounded(float x) { return f(b(x)); }
__device__ float gelu(float x) { return .5f * x * (1.f + erff(x * .7071067811865475244f)); }
__device__ float warp_sum(float x) {
    for(int delta=16;delta;delta/=2) x += __shfl_xor_sync(0xffffffff,x,delta);
    return x;
}

__global__ void cast_kernel(const float* x, BF* y, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=b(x[i]);
}
__global__ void residual_kernel(BF* x,const BF* delta,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=b(f(x[i])+f(delta[i]));
}
__global__ void bias_kernel(BF* x,const BF* bias,int n,int channels,int spatial,bool activation) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){float v=rounded(f(x[i])+f(bias[(i/spatial)%channels]));x[i]=b(activation?gelu(v):v);}
}
__global__ void float_bias_kernel(const float* x,const BF* bias,BF* out,int n,int channels,bool activation) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){float v=rounded(x[i]+f(bias[i%channels]));out[i]=b(activation?gelu(v):v);}
}
__global__ void conv_transpose_kernel(const BF* x,BF* y,int n,int channels,int freq,int steps) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){int d=i%(channels*freq),t=(i/(channels*freq))%steps,c=i/(channels*freq*steps);
        y[i]=x[((c*channels+d/freq)*freq+d%freq)*steps+t];}
}
__global__ void conv_padding_kernel(BF* x,const int* widths,int n,int channels,int frequency,int steps,int stride){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){int chunk=i/(channels*frequency*steps),step=i%steps;
        if(step>=(widths[chunk]+stride-1)/stride)x[i]=b(0.f);}
}
__global__ void pos_kernel(BF* x,const BF* pos,int n,int width,int steps) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=b(f(x[i])+f(pos[i%(width*steps)]));
}
__global__ void gather_kernel(const BF* x,const int* indices,BF* y,int n,int width) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=x[indices[i/width]*width+i%width];
}
__global__ void embed_kernel(const BF* embeddings,const BF* audio,const int* ids,const int* rows,BF* y,int n,int width) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int t=i/width,d=i%width;
        y[i]=rows[t]<0?embeddings[static_cast<long long>(ids[t])*width+d]:audio[rows[t]*width+d];}
}
__global__ void split_kernel(const BF* packed,BF* q,BF* k,BF* v,int tokens,int qw,int kw) {
    int i=blockIdx.x*blockDim.x+threadIdx.x,stride=qw+2*kw;
    if(i<tokens*stride){int t=i/stride,d=i%stride;
        if(d<qw)q[t*qw+d]=packed[i];else if(d<qw+kw)k[t*kw+d-qw]=packed[i];else v[t*kw+d-qw-kw]=packed[i];}
}
template<int D>
__global__ void norm_rope_kernel(const BF* packed,const BF* qweight,const BF* kweight,const float* inv,
    const int* positions,BF* q,BF* k,BF* v,int tokens,int qheads,int kvheads,float eps) {
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    int row=blockIdx.x*(blockDim.x/32)+warp;
    int heads=qheads+kvheads,t=row/heads,h=row%heads;
    if(t>=tokens)return;
    bool isq=h<qheads;int localh=isq?h:h-qheads;
    int qw=qheads*D,kw=kvheads*D,stride=qw+2*kw;
    const BF* src=packed+t*stride+(isq?0:qw)+localh*D;
    const BF* weight=isq?qweight:kweight;
    float sum=0;
    for(int d=lane;d<D;d+=32){float x=f(src[d]);sum=fmaf(x,x,sum);}
    float r=rsqrtf(warp_sum(sum)/D+eps);
    BF* dst=(isq?q+t*qw:k+t*kw)+localh*D;
    for(int d=lane;d<D/2;d+=32){float angle=positions[t]*inv[d];
        float cosine=rounded(cosf(angle)),sine=rounded(sinf(angle));
        float x=rounded(rounded(f(src[d])*r)*f(weight[d]));
        float y=rounded(rounded(f(src[d+D/2])*r)*f(weight[d+D/2]));
        dst[d]=b(rounded(x*cosine)-rounded(y*sine));
        dst[d+D/2]=b(rounded(y*cosine)+rounded(x*sine));}
    if(!isq)for(int d=lane;d<D;d+=32)v[t*kw+localh*D+d]=packed[t*stride+qw+kw+localh*D+d];
}
__global__ void kv_kernel(const BF* k,const BF* v,BF* ck,BF* cv,int n,int width,int capacity,const int* positions,bool decode) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int t=i/width,d=i%width;
        long long dest=decode?(static_cast<long long>(t)*capacity+positions[t])*width+d:static_cast<long long>(positions[t])*width+d;
        ck[dest]=k[i];cv[dest]=v[i];}
}
__global__ void increment_kernel(int* pos,int batch){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<batch)++pos[i];}
__global__ void norm_kernel(const BF* x,const BF* weight,BF* out,int rows,int width,float eps){
    int row=blockIdx.x,lane=threadIdx.x;float sum=0;
    for(int d=lane;d<width;d+=256){float value=f(x[row*width+d]);sum=fmaf(value,value,sum);}
    __shared__ float partial[8];sum=warp_sum(sum);if(lane%32==0)partial[lane/32]=sum;__syncthreads();
    if(lane<32){sum=lane<8?partial[lane]:0;sum=warp_sum(sum);if(lane==0)partial[0]=rsqrtf(sum/width+eps);}__syncthreads();
    for(int d=lane;d<width;d+=256)out[row*width+d]=b(rounded(f(x[row*width+d])*partial[0])*f(weight[d]));
}
__global__ void rope_kernel(BF* x,const float* inv,const int* positions,int tokens,int heads,int dim){
    int i=blockIdx.x*blockDim.x+threadIdx.x,n=tokens*heads*(dim/2);
    if(i<n){int d=i%(dim/2),row=i/(dim/2),t=row/heads;float angle=positions[t]*inv[d];
        float c=rounded(cosf(angle)),s=rounded(sinf(angle));int index=row*dim+d;
        float a=f(x[index]),other=f(x[index+dim/2]);x[index]=b(rounded(a*c)-rounded(other*s));
        x[index+dim/2]=b(rounded(other*c)+rounded(a*s));}
}
__global__ void swiglu_rounded_kernel(const BF* x,BF* out,int tokens,int width){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<tokens*width){int t=i/width,d=i%width;
        float gate=f(x[t*width*2+d]),up=f(x[t*width*2+width+d]);out[i]=b(rounded(gate/(1.f+expf(-gate)))*up);}
}

template<int D,int Warps,bool Decode,bool Split=false>
__global__ void attention_kernel(const BF* q,const BF* key,const BF* value,BF* out,
    int queries,int keys,int qheads,int kvheads,int qstride,int kvstride,
    const int* begins,const int* ends,const int* positions,bool causal,int capacity,
    float* partials,int splits) {
    int qi=blockIdx.x,h=blockIdx.y,lane=threadIdx.x%32,warp=threadIdx.x/32;
    int kvh=h/(qheads/kvheads);
    int begin=Decode?0:begins[qi],end=Decode?positions[qi]+1:ends[qi];
    if(causal&&!Decode)end=min(end,positions[qi]+1);
    if constexpr(Split){int length=end-begin;int start=begin;
        begin=start+length*blockIdx.z/splits;end=start+length*(blockIdx.z+1)/splits;}
    const BF* qrow=q+static_cast<long long>(qi)*qstride+h*D;
    const BF* kbase=key+(Decode?static_cast<long long>(qi)*capacity*kvstride:0)+kvh*D;
    const BF* vbase=value+(Decode?static_cast<long long>(qi)*capacity*kvstride:0)+kvh*D;
    float qv[D/32],acc[D/32];
    for(int d=0;d<D/32;++d){qv[d]=f(qrow[lane+d*32]);acc[d]=0;}
    float maximum=-INFINITY,denom=0;
    for(int j=begin+warp;j<end;j+=Warps){float score=0;
        for(int d=0;d<D/32;++d)score=fmaf(qv[d],f(kbase[static_cast<long long>(j)*kvstride+lane+d*32]),score);
        score=warp_sum(score)*rsqrtf(static_cast<float>(D));
        float next=fmaxf(maximum,score),scale=expf(maximum-next),prob=expf(score-next);
        denom=denom*scale+prob;
        for(int d=0;d<D/32;++d)acc[d]=fmaf(prob,f(vbase[static_cast<long long>(j)*kvstride+lane+d*32]),acc[d]*scale);
        maximum=next;
    }
    __shared__ float sums[Warps*D],maxima[Warps],denoms[Warps];
    if(lane==0){maxima[warp]=maximum;denoms[warp]=denom;}
    for(int d=0;d<D/32;++d)sums[warp*D+lane+d*32]=acc[d];
    __syncthreads();
    if(warp==0){float maximum_all=-INFINITY,denom_all=0;
        for(int w=0;w<Warps;++w)maximum_all=fmaxf(maximum_all,maxima[w]);
        for(int w=0;w<Warps;++w)if(denoms[w]>0)denom_all+=denoms[w]*expf(maxima[w]-maximum_all);
        float* partial=nullptr;
        if constexpr(Split){partial=partials+((static_cast<long long>(qi)*qheads+h)*splits+blockIdx.z)*(D+2);
            if(lane==0){partial[D]=maximum_all;partial[D+1]=denom_all;}}
        for(int d=0;d<D/32;++d){float value_all=0;
            for(int w=0;w<Warps;++w)if(denoms[w]>0)value_all+=sums[w*D+lane+d*32]*expf(maxima[w]-maximum_all);
            if constexpr(Split)partial[lane+d*32]=value_all;
            else out[(static_cast<long long>(qi)*qheads+h)*D+lane+d*32]=b(value_all/denom_all);}}
}
template<int D,int Splits>
__global__ void attention_merge(const float* partials,BF* out,int heads){
    int row=blockIdx.x*heads+blockIdx.y,lane=threadIdx.x;
    const float* src=partials+static_cast<long long>(row)*Splits*(D+2);
    float maximum=-INFINITY,denominator=0,scales[Splits];
#pragma unroll
    for(int s=0;s<Splits;++s)maximum=fmaxf(maximum,src[s*(D+2)+D]);
#pragma unroll
    for(int s=0;s<Splits;++s){scales[s]=src[s*(D+2)+D+1]>0?expf(src[s*(D+2)+D]-maximum):0;
        denominator+=src[s*(D+2)+D+1]*scales[s];}
#pragma unroll
    for(int d=0;d<D/32;++d){float total=0;
#pragma unroll
        for(int s=0;s<Splits;++s)total=fmaf(src[s*(D+2)+lane+d*32],scales[s],total);
        out[static_cast<long long>(row)*D+lane+d*32]=b(total/denominator);}
}
__global__ void argmax_first(const BF* logits,float* values,int* indices,int width,int blocks,int row_stride){
    int lane=threadIdx.x,group=blockIdx.x,t=blockIdx.y;float maximum=-INFINITY;int index=0x7fffffff;
    for(int i=group*1024+lane;i<min(width,(group+1)*1024);i+=256){float v=f(logits[static_cast<long long>(t)*row_stride+i]);
        if(v>maximum||(v==maximum&&i<index)){maximum=v;index=i;}}
    __shared__ float vals[256];__shared__ int ids[256];vals[lane]=maximum;ids[lane]=index;__syncthreads();
    for(int size=128;size;size/=2){if(lane<size){float v=vals[lane+size];int i=ids[lane+size];
        if(v>vals[lane]||(v==vals[lane]&&i<ids[lane])){vals[lane]=v;ids[lane]=i;}}__syncthreads();}
    if(lane==0){values[t*blocks+group]=vals[0];indices[t*blocks+group]=ids[0];}
}
__global__ void argmax_second(const float* values,const int* indices,int* out,int blocks){
    int lane=threadIdx.x,t=blockIdx.x;float maximum=-INFINITY;int index=0x7fffffff;
    for(int i=lane;i<blocks;i+=256){float v=values[t*blocks+i];int id=indices[t*blocks+i];
        if(v>maximum||(v==maximum&&id<index)){maximum=v;index=id;}}
    __shared__ float vals[256];__shared__ int ids[256];vals[lane]=maximum;ids[lane]=index;__syncthreads();
    for(int size=128;size;size/=2){if(lane<size){float v=vals[lane+size];int id=ids[lane+size];
        if(v>vals[lane]||(v==vals[lane]&&id<ids[lane])){vals[lane]=v;ids[lane]=id;}}__syncthreads();}
    if(lane==0)out[t]=ids[0];
}
template<bool Decode>
void launch_attention(const void* q,const void* k,const void* v,void* out,int queries,int keys,int d,int qh,int kh,
    int qs,int ks,const int* begin,const int* end,const int* pos,bool causal,int warps,int capacity,cudaStream_t stream){
    if(queries<=0||qh<=0||kh<=0||qh%kh||(d!=64&&d!=128)||(warps!=4&&warps!=8))throw std::invalid_argument("dense BF16 attention geometry");
    dim3 grid(queries,qh);
#define GO(D,W) attention_kernel<D,W,Decode><<<grid,W*32,0,stream>>>(static_cast<const BF*>(q),static_cast<const BF*>(k),static_cast<const BF*>(v),static_cast<BF*>(out),queries,keys,qh,kh,qs,ks,begin,end,pos,causal,capacity,nullptr,1)
    if(d==64){if(warps==4){GO(64,4);}else{GO(64,8);}}else{if(warps==4){GO(128,4);}else{GO(128,8);}}
#undef GO
    CUDA_CHECK(cudaGetLastError());
}
int decode_splits(int batch,int capacity,int d,int qh,int kh){
    if(batch<=0||capacity<=0||qh<=0||kh<=0||qh%kh||(d!=64&&d!=128))throw std::invalid_argument("dense BF16 decode attention geometry");
    // Small query batches need more KV partitions to cover SM120's SMs.
    // Four lanes use fewer partitions to limit partial-state merge work.
    if(d==128&&qh==16&&kh==8&&capacity>=128)return batch<=2?16:batch<=4?8:1;
    return 1;
}
template<int Splits>
void split_decode(const void* q,const void* k,const void* v,void* out,int batch,int capacity,int qh,int kh,
    const int* positions,float* scratch,cudaStream_t stream){
    attention_kernel<128,8,true,true><<<dim3(batch,qh,Splits),256,0,stream>>>(
        static_cast<const BF*>(q),static_cast<const BF*>(k),static_cast<const BF*>(v),static_cast<BF*>(out),
        batch,capacity,qh,kh,qh*128,kh*128,nullptr,nullptr,positions,true,capacity,scratch,Splits);
    attention_merge<128,Splits><<<dim3(batch,qh),32,0,stream>>>(scratch,static_cast<BF*>(out),qh);
    CUDA_CHECK(cudaGetLastError());
}
}

void float_to_bf16(const float* x,void* y,int n,cudaStream_t s){cast_kernel<<<(n+255)/256,256,0,s>>>(x,static_cast<BF*>(y),n);}
void rounded_bias_gelu(void* x,const void* bias,int n,int channels,int spatial,bool a,cudaStream_t s){bias_kernel<<<(n+255)/256,256,0,s>>>(static_cast<BF*>(x),static_cast<const BF*>(bias),n,channels,spatial,a);}
void float_bias_cast(const float* x,const void* bias,void* out,int n,int channels,bool a,cudaStream_t s){float_bias_kernel<<<(n+255)/256,256,0,s>>>(x,static_cast<const BF*>(bias),static_cast<BF*>(out),n,channels,a);}
void conv_to_tokens(const void* x,void* y,int chunks,int channels,int freq,int steps,cudaStream_t s){int n=chunks*channels*freq*steps;conv_transpose_kernel<<<(n+255)/256,256,0,s>>>(static_cast<const BF*>(x),static_cast<BF*>(y),n,channels,freq,steps);}
void zero_conv_padding(void* x,const int* widths,int chunks,int channels,int freq,int steps,int stride,cudaStream_t s){
    int n=chunks*channels*freq*steps;conv_padding_kernel<<<(n+255)/256,256,0,s>>>(static_cast<BF*>(x),widths,n,channels,freq,steps,stride);}
void add_chunk_positions(void* x,const void* pos,int chunks,int steps,int width,cudaStream_t s){int n=chunks*steps*width;pos_kernel<<<(n+255)/256,256,0,s>>>(static_cast<BF*>(x),static_cast<const BF*>(pos),n,width,steps);}
void gather_rows(const void* x,const int* indices,void* y,int rows,int width,cudaStream_t s){int n=rows*width;gather_kernel<<<(n+255)/256,256,0,s>>>(static_cast<const BF*>(x),indices,static_cast<BF*>(y),n,width);}
void embed_audio_tokens(const void* x,const void* a,const int* ids,const int* rows,void* y,int tokens,int width,cudaStream_t s){int n=tokens*width;embed_kernel<<<(n+255)/256,256,0,s>>>(static_cast<const BF*>(x),static_cast<const BF*>(a),ids,rows,static_cast<BF*>(y),n,width);}
void split_qkv(const void* x,void* q,void* k,void* v,int t,int qw,int kw,cudaStream_t s){int n=t*(qw+2*kw);split_kernel<<<(n+255)/256,256,0,s>>>(static_cast<const BF*>(x),static_cast<BF*>(q),static_cast<BF*>(k),static_cast<BF*>(v),t,qw,kw);}
void qk_norm_rope(const void* x,const void* qw,const void* kw,const float* inv,const int* pos,void* q,void* k,void* v,int t,int d,int qh,int kh,float eps,cudaStream_t s){
    if(d!=128)throw std::invalid_argument("norm RoPE supports D128");
    norm_rope_kernel<128><<<(t*(qh+kh)+7)/8,256,0,s>>>(static_cast<const BF*>(x),static_cast<const BF*>(qw),static_cast<const BF*>(kw),inv,pos,static_cast<BF*>(q),static_cast<BF*>(k),static_cast<BF*>(v),t,qh,kh,eps);
}
void append_contiguous_kv(const void* k,const void* v,void* ck,void* cv,int t,int width,int cap,const int* pos,bool dec,cudaStream_t s){int n=t*width;kv_kernel<<<(n+255)/256,256,0,s>>>(static_cast<const BF*>(k),static_cast<const BF*>(v),static_cast<BF*>(ck),static_cast<BF*>(cv),n,width,cap,pos,dec);}
void advance_positions(int* pos,int n,cudaStream_t s){increment_kernel<<<(n+255)/256,256,0,s>>>(pos,n);}
void rounded_rmsnorm(const void* x,const void* w,void* out,int rows,int width,float eps,cudaStream_t s){norm_kernel<<<rows,256,0,s>>>(static_cast<const BF*>(x),static_cast<const BF*>(w),static_cast<BF*>(out),rows,width,eps);}
void rounded_rope(void* x,const float* inv,const int* positions,int t,int h,int d,cudaStream_t s){int n=t*h*(d/2);rope_kernel<<<(n+255)/256,256,0,s>>>(static_cast<BF*>(x),inv,positions,t,h,d);}
void rounded_swiglu(const void* x,void* out,int t,int width,cudaStream_t s){int n=t*width;swiglu_rounded_kernel<<<(n+255)/256,256,0,s>>>(static_cast<const BF*>(x),static_cast<BF*>(out),t,width);}
void bf16_residual_add(void* x,const void* delta,int n,cudaStream_t s){residual_kernel<<<(n+255)/256,256,0,s>>>(static_cast<BF*>(x),static_cast<const BF*>(delta),n);}
void bf16_argmax(const void* x,float* vals,int* ids,int* out,int rows,int width,cudaStream_t s){int blocks=(width+1023)/1024;
    argmax_first<<<dim3(blocks,rows),256,0,s>>>(static_cast<const BF*>(x),vals,ids,width,blocks,width);
    argmax_second<<<rows,256,0,s>>>(vals,ids,out,blocks);
}
void bf16_argmax_valid(const void* x,float* vals,int* ids,int* out,int rows,int width,int row_stride,cudaStream_t s){
    if(rows<=0||width<=0||row_stride<width)throw std::invalid_argument("argmax valid shape invalid");
    int blocks=(width+1023)/1024;
    argmax_first<<<dim3(blocks,rows),256,0,s>>>(static_cast<const BF*>(x),vals,ids,width,blocks,row_stride);
    argmax_second<<<rows,256,0,s>>>(vals,ids,out,blocks);
}
void dense_bf16_attention(const void* q,const void* k,const void* v,void* o,int t,int keys,int d,int qh,int kh,int qs,int ks,const int* begin,const int* end,const int* pos,bool causal,cudaStream_t s){
    if(d==128&&causal&&t>=193&&qh>0&&kh>0&&qh%kh==0){
        detail::dense_float_prefill_kernel<4><<<dim3((t-1)/4+1,qh),128,0,s>>>(
            static_cast<const BF*>(q),static_cast<const BF*>(k),static_cast<const BF*>(v),static_cast<BF*>(o),
            t,qh,kh,qs,ks,begin,end,pos,causal);
        CUDA_CHECK(cudaGetLastError());return;
    }
    launch_attention<false>(q,k,v,o,t,keys,d,qh,kh,qs,ks,begin,end,pos,causal,4,0,s);
}
std::size_t dense_bf16_decode_attention_workspace_capacity(int batch,int cap,int d,int qh,int kh){
    int splits=decode_splits(batch,cap,d,qh,kh);return splits==1?0:static_cast<std::size_t>(batch)*qh*splits*(d+2)*4;
}
void dense_bf16_decode_attention(const void* q,const void* k,const void* v,void* o,int batch,int cap,int d,int qh,int kh,const int* pos,WorkspaceArena& ws,cudaStream_t s){
    int splits=decode_splits(batch,cap,d,qh,kh);
    if(splits==1){launch_attention<true>(q,k,v,o,batch,cap,d,qh,kh,qh*d,kh*d,nullptr,nullptr,pos,true,4,cap,s);return;}
    auto scope=ws.scope();auto partial=ws.alloc(DType::FP32,{d+2,splits,qh,batch},256);
    if(splits==16)split_decode<16>(q,k,v,o,batch,cap,qh,kh,pos,static_cast<float*>(partial.data),s);
    else split_decode<8>(q,k,v,o,batch,cap,qh,kh,pos,static_cast<float*>(partial.data),s);
}
} // namespace ninfer::ops
