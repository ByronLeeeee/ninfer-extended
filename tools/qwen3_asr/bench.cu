#include "ninfer/ops/linear.h"
#include "core/arena.h"
#include "core/device.h"
#include "ops/linear/bf16/bf16_launch.cuh"
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

using namespace ninfer;using Json=nlohmann::ordered_json;
namespace {
using Base=ops::detail::Bf16GemvSchedule<4,1,8,8,4,ops::detail::Bf16ActivationAccess::Direct,ops::detail::Bf16WeightCache::Default,ops::detail::Bf16PhaseOrder::RowSwizzled,1,1,1,2>;
using Row4=ops::detail::Bf16GemvSchedule<4,1,4,8,4,ops::detail::Bf16ActivationAccess::Direct,ops::detail::Bf16WeightCache::Default,ops::detail::Bf16PhaseOrder::RowSwizzled,1,1,1,2>;
using Value16=ops::detail::Bf16GemvSchedule<4,1,8,16,4,ops::detail::Bf16ActivationAccess::Direct,ops::detail::Bf16WeightCache::Default,ops::detail::Bf16PhaseOrder::RowSwizzled,1,1,1,2>;
using Chains8=ops::detail::Bf16GemvSchedule<4,1,8,8,8,ops::detail::Bf16ActivationAccess::Direct,ops::detail::Bf16WeightCache::Default,ops::detail::Bf16PhaseOrder::RowSwizzled,1,1,1,2>;
using Row8Cg=ops::detail::Bf16GemvSchedule<4,1,8,8,4,ops::detail::Bf16ActivationAccess::Direct,ops::detail::Bf16WeightCache::Streaming,ops::detail::Bf16PhaseOrder::RowSwizzled,1,1,1,2>;
using Wide=ops::detail::Bf16MmaSchedule<64,128,64,32,64,2,1,ops::Cache::cg,ops::Cache::cg,ops::detail::Bf16MmaFragmentPipeline::PingPong,ops::detail::Bf16MmaRaster::TokenFast>;
struct GemvOut {__nv_bfloat16* p;__device__ void store(int row,__nv_bfloat16 value)const{p[row]=value;}};
struct MmaOut {__nv_bfloat16* p;int rows;__device__ MmaOut tile(int)const{return *this;}__device__ void store(int row,int token,float v)const{p[static_cast<long long>(token)*rows+row]=__float2bfloat16_rn(v);}};
template<int N,int K,class Schedule>void gemv(const Tensor& x,const Weight& w,Tensor& out,cudaStream_t stream){using Geometry=ops::detail::Bf16Geometry<N,K>;
    ops::detail::bf16_gemv_kernel<Geometry,Schedule><<<N/Schedule::kRowsPerCta,Schedule::kThreads,0,stream>>>(static_cast<const __nv_bfloat16*>(x.data),static_cast<const __nv_bfloat16*>(w.qdata),GemvOut{static_cast<__nv_bfloat16*>(out.data)});}
template<int N,int K>void wide(const Tensor& x,const Weight& w,Tensor& out,cudaStream_t stream){using Geometry=ops::detail::Bf16Geometry<N,K>;int blocks=N/Wide::kBlockRows*((x.ne[1]+Wide::kBlockCols-1)/Wide::kBlockCols);
    ops::detail::bf16_gemm_mma_kernel<Geometry,Wide,false><<<blocks,Wide::kThreads,Wide::kSharedBytes,stream>>>(static_cast<const __nv_bfloat16*>(x.data),static_cast<const __nv_bfloat16*>(w.qdata),MmaOut{static_cast<__nv_bfloat16*>(out.data),N},x.ne[1]);}
template<int N,int K>void candidate(int version,const Tensor& x,const Weight& w,Tensor& out,cudaStream_t stream){
    if(version==1)gemv<N,K,Row4>(x,w,out,stream);else if(version==2)gemv<N,K,Value16>(x,w,out,stream);else if(version==3)gemv<N,K,Chains8>(x,w,out,stream);else if(version==4)gemv<N,K,Row8Cg>(x,w,out,stream);else wide<N,K>(x,w,out,stream);}
void dispatch(int version,const Tensor& x,const Weight& w,Tensor& out,cudaStream_t stream){
#define S(N,K) if(w.n==N&&w.k==K){candidate<N,K>(version,x,w,out,stream);return;}
    S(2048,2048) S(4096,2048) S(12288,2048) S(2048,6144) S(151936,2048)
#undef S
    throw std::runtime_error("Unknown shape");}
}
int main(){try{DeviceContext device;cublasHandle_t handle;cublasCreate(&handle);cublasSetStream(handle,device.stream);DeviceBuffer blas_ws(16*1024*1024);cublasSetWorkspace(handle,blas_ws.p,blas_ws.bytes);
    std::mt19937 random(1705070);std::uniform_real_distribution<float> uniform(-.5,.5);
    for(auto [n,k]:std::vector<std::pair<int,int>>{{2048,2048},{4096,2048},{12288,2048},{2048,6144},{151936,2048}}){std::size_t size=static_cast<std::size_t>(n)*k*2;int copies=std::max(1,static_cast<int>((128ULL*1024*1024+size-1)/size));
        std::vector<__nv_bfloat16> values(size/2);for(auto& value:values)value=__float2bfloat16_rn(uniform(random));DeviceBuffer weights(size*copies);for(int c=0;c<copies;++c)weights.copy_from_host(values.data(),size,size*c);
        for(int t:{1,2,4,70,211,805}){if(n==151936&&t>4)continue;std::vector<__nv_bfloat16> xv(k*t);for(auto& v:xv)v=__float2bfloat16_rn(uniform(random));DeviceBuffer xb(xv.size()*2),ob(static_cast<std::size_t>(n)*t*2);xb.copy_from_host(xv.data(),xb.bytes);Tensor x(xb.p,DType::BF16,{k,t}),out(ob.p,DType::BF16,{n,t});Weight w;w.n=n;w.k=k;w.qtype=QType::BF16;w.layout=QuantLayout::Contiguous;
            for(int version=0;version<=6;++version){if(t!=1&&version>=1&&version<=4)continue;if(t<=4&&version==5)continue;
                auto launch=[&](int index){w.qdata=static_cast<std::byte*>(weights.p)+size*(index%copies);if(version==0)ops::linear(x,w,out,device.stream);else if(version==6){float alpha=1,beta=0;auto status=cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,t,k,&alpha,w.qdata,CUDA_R_16BF,k,x.data,CUDA_R_16BF,k,&beta,out.data,CUDA_R_16BF,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS failed");}else dispatch(version,x,w,out,device.stream);};
                for(int i=0;i<copies;++i)launch(i);device.synchronize();cudaGraph_t graph;CUDA_CHECK(cudaStreamBeginCapture(device.stream,cudaStreamCaptureModeGlobal));for(int i=0;i<copies;++i)launch(i);CUDA_CHECK(cudaStreamEndCapture(device.stream,&graph));cudaGraphExec_t executable;CUDA_CHECK(cudaGraphInstantiate(&executable,graph,0));cudaGraphDestroy(graph);
                std::vector<float> samples;for(int r=0;r<5;++r){CudaEventTimer timer(device);timer.start();for(int i=0;i<20;++i)CUDA_CHECK(cudaGraphLaunch(executable,device.stream));samples.push_back(timer.stop_ms()/(copies*20));}cudaGraphExecDestroy(executable);std::sort(samples.begin(),samples.end());
                const char* names[]={"production","gemv_row4","gemv_value16","gemv_chains8","gemv_streaming","wide_mma","cublas"};std::cout<<Json({{"n",n},{"k",k},{"tokens",t},{"candidate",names[version]},{"median_ms",samples[2]},{"weight_copies",copies},{"working_set_bytes",size*copies}}).dump()<<std::endl;
            }
        }
    }cublasDestroy(handle);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
