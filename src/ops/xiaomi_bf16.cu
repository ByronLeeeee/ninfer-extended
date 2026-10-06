#include "ops/xiaomi_bf16.h"
#include "core/device.h"
#include "ops/linear/bf16/bf16_launch.cuh"
#include <stdexcept>
namespace ninfer::ops::detail {
namespace {
using Gemv=Bf16GemvSchedule<4,1,8,8,4,Bf16ActivationAccess::Direct,Bf16WeightCache::Default,Bf16PhaseOrder::RowSwizzled,1,1,1,2>;
using Mma=Bf16MmaSchedule<64,64,64,32,32,2,2,Cache::cg,Cache::cg,Bf16MmaFragmentPipeline::PingPong,Bf16MmaRaster::TokenFast>;

struct ResidualOutput {
    __nv_bfloat16* data;
    __device__ __forceinline__ void store(int row, __nv_bfloat16 projection) const {
        // Preserve the existing BF16 projection boundary before residual addition.
        data[row] = __float2bfloat16_rn(__bfloat162float(projection) + __bfloat162float(data[row]));
    }
};

struct SplitOutput {
    __nv_bfloat16* first;
    __nv_bfloat16* second;
    int first_rows;
    __device__ __forceinline__ void store(int row, __nv_bfloat16 projection) const {
        if (row < first_rows) first[row] = projection;
        else second[row - first_rows] = projection;
    }
};

struct AttentionOutput {
    __nv_bfloat16* q;
    __nv_bfloat16* gate;
    __nv_bfloat16* k;
    __nv_bfloat16* v;
    __device__ __forceinline__ void store(int row, __nv_bfloat16 projection) const {
        if (row < 2048) q[row] = projection;
        else if (row < 2560) k[row - 2048] = projection;
        else if (row < 4608) gate[row - 2560] = projection;
        else v[row - 4608] = projection;
    }
};

template<int N, int K, class Output>
void launch_decode(const Tensor& x, const Weight& w, Output out, cudaStream_t stream) {
    bf16_gemv_kernel<Bf16Geometry<N, K>, Gemv>
        <<<N / Gemv::kRowsPerCta, Gemv::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(w.qdata), out);
    CUDA_CHECK(cudaGetLastError());
}

void require_input(const Tensor& x, const Weight& w) {
    if (w.qtype != QType::BF16 || w.layout != QuantLayout::Contiguous ||
        x.dtype != DType::BF16 || !x.is_contiguous() || x.ne[0] != w.k || x.ne[1] <= 0)
        throw std::invalid_argument("Xiaomi BF16 linear operand mismatch");
}

void require_output(const Tensor& x, const Tensor& out, int rows) {
    if (out.dtype != DType::BF16 || !out.is_contiguous() ||
        out.ne[0] != rows || out.ne[1] != x.ne[1])
        throw std::invalid_argument("Xiaomi BF16 linear output mismatch");
}
template<int N,int K> void launch(const Tensor& x,const Weight&w,Tensor& out,cudaStream_t s) {
    using G=Bf16Geometry<N,K>;
    if(x.ne[1]==1) launch_bf16_gemv<G,Gemv>(x,w,out,s);
    else launch_bf16_mma<G,Mma>(x,w,out,s);
}
__global__ void add_kernel(const __nv_bfloat16*x,__nv_bfloat16*y,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) y[i]=__float2bfloat16_rn(__bfloat162float(x[i])+__bfloat162float(y[i]));
}
__global__ void swiglu_kernel(const __nv_bfloat16*x,__nv_bfloat16*y,int rows,int tokens) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<rows*tokens) {int row=i%rows,t=i/rows; float gate=__bfloat162float(x[t*2*rows+row]);
        float up=__bfloat162float(x[t*2*rows+rows+row]);
        float silu=__bfloat162float(__float2bfloat16_rn(gate/(1.f+expf(-gate))));
        y[i]=__float2bfloat16_rn(silu*up);}
}
__global__ void split_kernel(const __nv_bfloat16*x,__nv_bfloat16*a,__nv_bfloat16*b,int na,int nb,int tokens) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<(na+nb)*tokens) {int row=i%(na+nb),t=i/(na+nb); if(row<na)a[t*na+row]=x[i]; else b[t*nb+row-na]=x[i];}
}
__global__ void attn_split_kernel(const __nv_bfloat16*x,__nv_bfloat16*q,__nv_bfloat16*k,__nv_bfloat16*g,__nv_bfloat16*v,int tokens) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<5120*tokens) {int row=i%5120,t=i/5120; if(row<2048)q[t*2048+row]=x[i];
        else if(row<2560)k[t*512+row-2048]=x[i]; else if(row<4608)g[t*2048+row-2560]=x[i];else v[t*512+row-4608]=x[i];}
}
__global__ void control_kernel(const __nv_bfloat16*x,const __nv_bfloat16*w,const float*alog,const float*dt,float*g,float*beta,int tokens) {
    int lane=threadIdx.x%32,warp=threadIdx.x/32; int head=(blockIdx.x*blockDim.x/32+warp)%16;
    int token=(blockIdx.x*blockDim.x/32+warp)/16; if(token>=tokens)return;
    float a=0,b=0;
    for(int k=lane;k<1024;k+=32){float value=__bfloat162float(x[token*1024+k]);a=fmaf(__bfloat162float(w[head*1024+k]),value,a);b=fmaf(__bfloat162float(w[(16+head)*1024+k]),value,b);}
    for(int delta=16;delta;delta/=2){a+=__shfl_down_sync(0xffffffff,a,delta);b+=__shfl_down_sync(0xffffffff,b,delta);}
    if(lane==0){a=__bfloat162float(__float2bfloat16_rn(a));b=__bfloat162float(__float2bfloat16_rn(b));
        float z=a+dt[head];float softplus=z>20?z:log1pf(expf(z));g[token*16+head]=-expf(alog[head])*softplus;beta[token*16+head]=1.f/(1.f+expf(-b));}
}
Tensor temporary(WorkspaceArena&ws,int rows,int tokens){return ws.alloc(DType::BF16,{rows,tokens},256);}
}
bool xiaomi_bf16_shape(int n,int k) {
    return (k==1024&&(n==8192||n==5120||n==7168||n==248320)) || (n==1024&&(k==2048||k==3584||k==3072)) || (n==768&&(k==1536||k==768||k==3072)) || (n==2304&&k==768) || (n==3072&&(k==768||k==3072));
}
void xiaomi_bf16_linear(const Tensor&x,const Weight&w,Tensor&out,cudaStream_t s){
    require_input(x,w);
    require_output(x,out,w.n);
#define SHAPE(N,K) if(w.n==N&&w.k==K){launch<N,K>(x,w,out,s);return;}
    SHAPE(8192,1024) SHAPE(5120,1024) SHAPE(7168,1024) SHAPE(248320,1024)
    SHAPE(1024,2048) SHAPE(1024,3584) SHAPE(1024,3072)
    SHAPE(768,1536) SHAPE(768,768) SHAPE(768,3072) SHAPE(2304,768) SHAPE(3072,768) SHAPE(3072,3072)
#undef SHAPE
    throw std::invalid_argument("Xiaomi BF16 unregistered matrix shape");
}
void xiaomi_bf16_add(const Tensor&x,const Weight&w,Tensor&out,WorkspaceArena&ws,cudaStream_t s){
    if (x.ne[1] == 1 && w.n == 1024) {
        require_input(x,w);
        require_output(x,out,w.n);
        const ResidualOutput output{static_cast<__nv_bfloat16*>(out.data)};
        if (w.k == 2048) { launch_decode<1024,2048>(x,w,output,s); return; }
        if (w.k == 3072) { launch_decode<1024,3072>(x,w,output,s); return; }
        if (w.k == 3584) { launch_decode<1024,3584>(x,w,output,s); return; }
    }
    auto scope=ws.scope();auto tmp=temporary(ws,w.n,x.ne[1]);xiaomi_bf16_linear(x,w,tmp,s);int n=w.n*x.ne[1];add_kernel<<<(n+255)/256,256,0,s>>>((const __nv_bfloat16*)tmp.data,(__nv_bfloat16*)out.data,n);CUDA_CHECK(cudaGetLastError());
}
void xiaomi_bf16_swiglu(const Tensor&x,const Weight&w,Tensor&out,WorkspaceArena&ws,cudaStream_t s){auto scope=ws.scope();auto tmp=temporary(ws,w.n,x.ne[1]);xiaomi_bf16_linear(x,w,tmp,s);int n=w.n/2*x.ne[1];swiglu_kernel<<<(n+255)/256,256,0,s>>>((const __nv_bfloat16*)tmp.data,(__nv_bfloat16*)out.data,w.n/2,x.ne[1]);CUDA_CHECK(cudaGetLastError());}
void xiaomi_bf16_split(const Tensor&x,const Weight&w,Tensor&a,Tensor&b,WorkspaceArena&ws,cudaStream_t s){
    if (x.ne[1] == 1 && w.n == 8192 && w.k == 1024) {
        require_input(x,w);
        require_output(x,a,6144);
        require_output(x,b,2048);
        launch_decode<8192,1024>(x,w,SplitOutput{static_cast<__nv_bfloat16*>(a.data),
            static_cast<__nv_bfloat16*>(b.data),6144},s);
        return;
    }
    auto scope=ws.scope();auto tmp=temporary(ws,w.n,x.ne[1]);xiaomi_bf16_linear(x,w,tmp,s);int n=w.n*x.ne[1];split_kernel<<<(n+255)/256,256,0,s>>>((const __nv_bfloat16*)tmp.data,(__nv_bfloat16*)a.data,(__nv_bfloat16*)b.data,a.ne[0],b.ne[0],x.ne[1]);CUDA_CHECK(cudaGetLastError());
}
void xiaomi_bf16_attn(const Tensor&x,const Weight&w,Tensor&q,Tensor&gate,Tensor&k,Tensor&v,WorkspaceArena&ws,cudaStream_t s){
    if (x.ne[1] == 1 && w.n == 5120 && w.k == 1024) {
        require_input(x,w);
        require_output(x,q,2048);
        require_output(x,gate,2048);
        require_output(x,k,512);
        require_output(x,v,512);
        launch_decode<5120,1024>(x,w,AttentionOutput{static_cast<__nv_bfloat16*>(q.data),
            static_cast<__nv_bfloat16*>(gate.data),static_cast<__nv_bfloat16*>(k.data),
            static_cast<__nv_bfloat16*>(v.data)},s);
        return;
    }
    auto scope=ws.scope();auto tmp=temporary(ws,w.n,x.ne[1]);xiaomi_bf16_linear(x,w,tmp,s);int n=w.n*x.ne[1];attn_split_kernel<<<(n+255)/256,256,0,s>>>((const __nv_bfloat16*)tmp.data,(__nv_bfloat16*)q.data,(__nv_bfloat16*)k.data,(__nv_bfloat16*)gate.data,(__nv_bfloat16*)v.data,x.ne[1]);CUDA_CHECK(cudaGetLastError());
}
void xiaomi_bf16_control(const Tensor&x,const Weight&w,const Tensor&alog,const Tensor&dt,Tensor&g,Tensor&beta,cudaStream_t s){int warps=16*x.ne[1];control_kernel<<<(warps+7)/8,256,0,s>>>((const __nv_bfloat16*)x.data,(const __nv_bfloat16*)w.qdata,(const float*)alog.data,(const float*)dt.data,(float*)g.data,(float*)beta.data,x.ne[1]);CUDA_CHECK(cudaGetLastError());}
}
