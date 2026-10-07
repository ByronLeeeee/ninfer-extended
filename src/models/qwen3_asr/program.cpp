#include "models/qwen3_asr/program.h"
#include "artifact/reader.h"
#include "core/arena.h"
#include "core/weight.h"
#include "ninfer/ops/bf16_transforms.h"
#include "ninfer/ops/dense_attention.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/layer_norm.h"
#include "ninfer/ops/softmax_attention.h"

#include <cublas_v2.h>
#include <cudnn.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_asr {
namespace {
constexpr int kConvolutionChunkBatch = 32;
void require(bool condition,const char* message){if(!condition)throw std::invalid_argument(message);}
void blas(cublasStatus_t status){if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("ASR cuBLAS operation failed: "+std::to_string(status));}
void dnn(cudnnStatus_t status){if(status!=CUDNN_STATUS_SUCCESS)throw std::runtime_error(std::string("ASR cuDNN operation failed: ")+cudnnGetErrorString(status));}
std::size_t aligned(std::size_t n){return (n+255)&~std::size_t(255);}

struct Parameter {void* data=nullptr;std::vector<std::uint64_t> shape;DType dtype=DType::BF16;};
class Model {
public:
    artifact::Json config;
    DeviceBuffer storage;
    std::map<std::string,Parameter,std::less<>> parameters;
    std::size_t weight_bytes=0;
    explicit Model(const std::filesystem::path& path){
        artifact::Reader reader(path);config=reader.directory().component("text").config;
        require(config.at("architectures")==artifact::Json::array({"Qwen3ASRForConditionalGeneration"}),"ASR artifact architecture mismatch");
        for(const auto& object:reader.directory().objects)if(const auto* tensor=std::get_if<artifact::TensorObject>(&object)){
            require(tensor->layout=="contiguous_le_v1"&&(tensor->format=="bf16"||
                (tensor->id=="text.rope_inv_freq"&&tensor->format=="fp32")),
                "ASR requires contiguous BF16 weights and FP32 RoPE constants");
            weight_bytes=aligned(weight_bytes)+tensor->bytes;
        }
        storage=DeviceBuffer(weight_bytes);std::size_t offset=0;
        std::vector<std::byte> staging(8*1024*1024);
        for(const auto& object:reader.directory().objects)if(const auto* tensor=std::get_if<artifact::TensorObject>(&object)){
            offset=aligned(offset);auto* destination=static_cast<std::byte*>(storage.p)+offset;
            for(std::size_t done=0;done<tensor->bytes;){auto count=std::min(staging.size(),static_cast<std::size_t>(tensor->bytes)-done);
                reader.read_into(tensor->offset+done,std::span(staging.data(),count));
                CUDA_CHECK(cudaMemcpy(destination+done,staging.data(),count,cudaMemcpyHostToDevice));done+=count;}
            parameters.emplace(tensor->id,Parameter{destination,tensor->shape,tensor->format=="fp32"?DType::FP32:DType::BF16});offset+=tensor->bytes;
        }
    }
    const Parameter& at(const std::string& name)const{return parameters.at(name);}
    Tensor vector(const std::string& name)const{const auto& p=at(name);return Tensor(p.data,p.dtype,{static_cast<int>(p.shape.at(0))});}
    Weight matrix(const std::string& name)const{const auto& p=at(name);require(p.shape.size()==2&&p.dtype==DType::BF16,"ASR matrix parameter invalid");
        Weight w;w.qtype=QType::BF16;w.layout=QuantLayout::Contiguous;w.qdata=w.payload=p.data;w.n=p.shape[0];w.k=p.shape[1];w.ndim=2;
        w.shape[0]=w.padded_shape[0]=w.k;w.shape[1]=w.padded_shape[1]=w.n;w.payload_bytes=static_cast<std::uint64_t>(w.n)*w.k*2;return w;}
};

struct Convolution {
    cudnnTensorDescriptor_t input=nullptr,output=nullptr;
    cudnnFilterDescriptor_t filter=nullptr;
    cudnnConvolutionDescriptor_t convolution=nullptr;
    cudnnConvolutionFwdAlgo_t algorithm=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    int out_h=0,out_w=0,out_channels=0;
    std::size_t workspace_bytes=0;
    Convolution(cudnnHandle_t handle,int chunks,int channels,int h,int w,int outputs){
        dnn(cudnnCreateTensorDescriptor(&input));dnn(cudnnCreateTensorDescriptor(&output));dnn(cudnnCreateFilterDescriptor(&filter));dnn(cudnnCreateConvolutionDescriptor(&convolution));
        dnn(cudnnSetTensor4dDescriptor(input,CUDNN_TENSOR_NCHW,CUDNN_DATA_BFLOAT16,chunks,channels,h,w));
        dnn(cudnnSetFilter4dDescriptor(filter,CUDNN_DATA_BFLOAT16,CUDNN_TENSOR_NCHW,outputs,channels,3,3));
        dnn(cudnnSetConvolution2dDescriptor(convolution,1,1,2,2,1,1,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT));
        dnn(cudnnSetConvolutionMathType(convolution,CUDNN_TENSOR_OP_MATH));
        int n,c;dnn(cudnnGetConvolution2dForwardOutputDim(convolution,input,filter,&n,&c,&out_h,&out_w));out_channels=c;
        dnn(cudnnSetTensor4dDescriptor(output,CUDNN_TENSOR_NCHW,CUDNN_DATA_BFLOAT16,n,c,out_h,out_w));
        int count=0;cudnnConvolutionFwdAlgoPerf_t results[8];
        dnn(cudnnGetConvolutionForwardAlgorithm_v7(handle,input,filter,convolution,output,8,&count,results));
        bool selected=false;
        for(int i=0;i<count;++i)if(results[i].status==CUDNN_STATUS_SUCCESS){
            // v7 returns separate plans for each math type. Query the actual
            // workspace with that plan's math type before committing it.
            dnn(cudnnSetConvolutionMathType(convolution,results[i].mathType));
            std::size_t required=0;
            if(cudnnGetConvolutionForwardWorkspaceSize(handle,input,filter,convolution,output,results[i].algo,&required)!=CUDNN_STATUS_SUCCESS||required>128ULL*1024*1024)continue;
            algorithm=results[i].algo;workspace_bytes=required;selected=true;break;
        }
        require(selected,"No ASR BF16 convolution plan fits workspace");
    }
    ~Convolution(){if(input)cudnnDestroyTensorDescriptor(input);if(output)cudnnDestroyTensorDescriptor(output);if(filter)cudnnDestroyFilterDescriptor(filter);if(convolution)cudnnDestroyConvolutionDescriptor(convolution);}
};

struct Run {
    int batch=0,chunks=0,audio_tokens=0,max_prompt=0,capacity=0;
    std::size_t bytes=0;
    std::vector<int> audio_offsets,prompt_lengths;
    DeviceBuffer input_f32,cnn_input,cnn1,cnn2,cnn3,cnn_flat,cnn_projected,audio_hidden,audio_norm,audio_qkv,audio_attn,audio_ffn,audio_delta,audio_projected;
    DeviceBuffer valid_indices,cu_seqlens,key_begin,key_end,features_out;
    DeviceBuffer hidden,norm,qkv,query,key,value,attention,delta,gate_up,ffn,logits,float_output;
    DeviceBuffer prompt_ids,audio_rows,prefill_positions,prefill_begin,prefill_end,decode_positions,decode_ids,empty_audio_rows;
    DeviceBuffer k_cache,v_cache,arg_values,arg_indices;
    std::vector<std::unique_ptr<Convolution>> conv;
    std::map<unsigned,cudaGraphExec_t> graphs;
    std::map<unsigned,cudaGraphExec_t> audio_graphs,prefill_graphs;
    ~Run(){for(auto* cache:{&graphs,&audio_graphs,&prefill_graphs})for(auto [key,graph]:*cache)cudaGraphExecDestroy(graph);}
    DeviceBuffer buffer(std::size_t n){bytes+=n;return DeviceBuffer(n);}
};
}

class Program::Impl {
public:
    EngineOptions options;
    DeviceContext& device;
    Model model;
    cublasHandle_t cublas=nullptr;
    cudnnHandle_t cudnn=nullptr;
    DeviceBuffer convolution_workspace,blas_workspace;
    WorkspaceArena workspace;
    std::unique_ptr<Run> run;
    std::string signature;
    std::mutex mutex;
    int hidden_size,layers,query_heads,kv_heads,head_dim,intermediate,vocab,audio_width,audio_layers,audio_heads,conv_channels;
    double load_seconds=0;
    Impl(const EngineOptions& o,DeviceContext& d):options(o),device(d),model(o.artifact_path),convolution_workspace(128ULL*1024*1024),blas_workspace(16ULL*1024*1024),workspace(64ULL*1024*1024){
        const auto& text=model.config.at("text_config");const auto& audio=model.config.at("audio_config");
        hidden_size=text.at("hidden_size");layers=text.at("num_hidden_layers");query_heads=text.at("num_attention_heads");kv_heads=text.at("num_key_value_heads");head_dim=text.at("head_dim");
        intermediate=text.at("intermediate_size");vocab=text.at("vocab_size");audio_width=audio.at("d_model");audio_layers=audio.at("encoder_layers");audio_heads=audio.at("encoder_attention_heads");conv_channels=audio.at("downsample_hidden_size");
        require(hidden_size==2048&&layers==28&&query_heads==16&&kv_heads==8&&head_dim==128&&intermediate==6144&&vocab==151936,
            "ASR text configuration has unqualified operator geometry");
        require(audio_width==1024&&audio_layers==24&&audio_heads==16&&conv_channels==480&&audio.at("encoder_ffn_dim")==4096&&
            audio.at("num_mel_bins")==128&&audio.at("n_window")==50&&audio.at("n_window_infer")==800&&
            audio.at("max_position_embeddings")==13&&audio.at("output_dim")==2048&&audio.at("activation_function")=="gelu"&&
            !audio.at("scale_embedding").get<bool>(),"ASR audio configuration has unqualified operator geometry");
        require(text.at("hidden_act")=="silu"&&!text.at("attention_bias").get<bool>()&&!text.at("use_sliding_window").get<bool>()&&
            text.at("rms_norm_eps")==1e-6&&text.at("rope_parameters").at("rope_type")=="default"&&
            text.at("rope_parameters").at("rope_theta")==1000000&&model.config.at("tie_word_embeddings")==true&&
            model.config.at("audio_token_id")==151676&&model.config.at("eos_token_id")==artifact::Json::array({151643,151645}),
            "ASR configuration differs from the qualified BF16 execution profile");
        require(options.kv_cache==KvCacheStorage::BFloat16,"ASR currently requires BF16 KV");
        blas(cublasCreate(&cublas));blas(cublasSetStream(cublas,device.stream));blas(cublasSetMathMode(cublas,CUBLAS_TENSOR_OP_MATH));
        blas(cublasSetWorkspace(cublas,blas_workspace.p,blas_workspace.bytes));dnn(cudnnCreate(&cudnn));dnn(cudnnSetStream(cudnn,device.stream));
    }
    ~Impl(){device.bind_to_current_thread_noexcept();run.reset();if(cudnn)cudnnDestroy(cudnn);if(cublas)cublasDestroy(cublas);}
    void linear(const void* x,int tokens,const std::string& name,void* out,const SpeechRunOptions& execution,bool activation=false){
        const auto w=model.matrix(name+".weight");auto bias=model.parameters.find(name+".bias");
        float alpha=1,beta=0;
        if(bias!=model.parameters.end()){
            blas(cublasGemmEx(cublas,CUBLAS_OP_T,CUBLAS_OP_N,w.n,tokens,w.k,&alpha,w.qdata,CUDA_R_16BF,w.k,x,CUDA_R_16BF,w.k,&beta,run->float_output.p,CUDA_R_32F,w.n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            ops::float_bias_cast(static_cast<float*>(run->float_output.p),bias->second.data,out,tokens*w.n,w.n,activation,device.stream);
        }else if(execution.linear==SpeechLinearBackend::Native){
            Tensor input(const_cast<void*>(x),DType::BF16,{w.k,tokens}),output(out,DType::BF16,{w.n,tokens});
            ops::linear(input,w,output,device.stream);
        }else{
            blas(cublasGemmEx(cublas,CUBLAS_OP_T,CUBLAS_OP_N,w.n,tokens,w.k,&alpha,w.qdata,CUDA_R_16BF,w.k,x,CUDA_R_16BF,w.k,&beta,out,CUDA_R_16BF,w.n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    }
    void norm(const void* x,const std::string& prefix,void* out,int tokens,int width,bool rms){
        if(rms)ops::rounded_rmsnorm(x,model.at(prefix+".weight").data,out,tokens,width,1e-6f,device.stream);
        else{Tensor input(const_cast<void*>(x),DType::BF16,{width,tokens}),output(out,DType::BF16,{width,tokens});
            ops::layer_norm(input,model.vector(prefix+".weight"),model.vector(prefix+".bias"),1e-5f,output,device.stream);}
    }
    void residual(void* hidden,const void* delta,int elements){ops::bf16_residual_add(hidden,delta,elements,device.stream);}
    void project_residual(const void* input,int tokens,const std::string& name,void* hidden,const SpeechRunOptions& execution){
        if(execution.linear==SpeechLinearBackend::Native&&execution.fused_projection_residual){
            auto weight=model.matrix(name+".weight");Tensor x(const_cast<void*>(input),DType::BF16,{weight.k,tokens}),out(hidden,DType::BF16,{weight.n,tokens});
            ops::linear_add(x,weight,out,ops::LinearPolicy::A16Only,workspace,device.stream);
        }else{linear(input,tokens,name,run->delta.p,execution);residual(hidden,run->delta.p,tokens*hidden_size);}
    }

    void prepare(const std::vector<SpeechFeatures>& samples){
        require(!samples.empty()&&samples.size()<=options.max_concurrency,"ASR batch exceeds configured concurrency");
        std::ostringstream key;int chunks=0,audio_tokens=0,max_prompt=0;
        std::vector<int> indices,begin,end,cu{0},offsets,lengths;
        std::vector<float> input;
        for(const auto& sample:samples){
            require(sample.mel_bins==128&&sample.frames>0&&sample.frames%100==0&&sample.mel.size()==static_cast<std::size_t>(128)*sample.frames&&sample.frame_mask.size()==static_cast<std::size_t>(sample.frames),"ASR input feature shape invalid");
            require(!sample.prompt_tokens.empty()&&sample.prompt_tokens.size()<options.max_context,"ASR prompt length invalid");
            require(std::all_of(sample.prompt_tokens.begin(),sample.prompt_tokens.end(),[&](int token){return token>=0&&token<vocab;}),"ASR token ID out of range");
            require(sample.audio_token_id==151676,"ASR audio token ID mismatch");
            bool padded=false;
            for(int mask:sample.frame_mask){require(mask==0||mask==1,"ASR frame mask must be binary");if(mask==0)padded=true;else require(!padded,"ASR frame mask must contain a contiguous valid prefix");}
            int count=0;offsets.push_back(audio_tokens);
            for(int chunk=0;chunk<sample.frames/100;++chunk){int valid=0;
                for(int t=0;t<100;++t){int mask=sample.frame_mask[chunk*100+t];require(mask==0||mask==1,"ASR frame mask must be binary");valid+=mask;}
                int post=(valid+7)/8;
                for(int t=0;t<post;++t)indices.push_back((chunks+chunk)*13+t);
                count+=post;
                for(int m=0;m<128;++m)for(int t=0;t<100;++t)input.push_back(sample.mel[m*sample.frames+chunk*100+t]);
            }
            require(count>0&&std::count(sample.prompt_tokens.begin(),sample.prompt_tokens.end(),sample.audio_token_id)==count,"ASR audio placeholder count mismatch");
            for(int start=0;start<count;start+=104){int stop=std::min(count,start+104);for(int t=start;t<stop;++t){begin.push_back(audio_tokens+start);end.push_back(audio_tokens+stop);}cu.push_back(audio_tokens+stop);}
            chunks+=sample.frames/100;audio_tokens+=count;lengths.push_back(sample.prompt_tokens.size());max_prompt=std::max(max_prompt,static_cast<int>(sample.prompt_tokens.size()));
            key<<sample.frames<<':'<<count<<':'<<sample.prompt_tokens.size()<<';';
        }
        key<<options.max_context<<':'<<samples.size();
        if(signature!=key.str()){
            device.synchronize();run.reset();signature.clear();auto pending=std::make_unique<Run>();auto& r=*pending;
            auto required=ops::linear_swiglu_workspace_capacity_bytes(QType::BF16,2*intermediate,hidden_size,
                ops::LinearPolicy::A16Only,1,max_prompt);
            required=std::max(required,ops::packed_softmax_attention_workspace_capacity_bytes({64,16,16},audio_tokens,audio_tokens,1,cu.size()-1));
            required=std::max(required,ops::dense_bf16_decode_attention_workspace_capacity(samples.size(),options.max_context,head_dim,query_heads,kv_heads));
            auto capacity=aligned(std::max<std::size_t>(64ULL*1024*1024,required));
            if(capacity!=workspace.capacity())workspace=WorkspaceArena(capacity);
            r.batch=samples.size();r.chunks=chunks;r.audio_tokens=audio_tokens;r.max_prompt=max_prompt;r.capacity=options.max_context;r.audio_offsets=offsets;r.prompt_lengths=lengths;
            auto bf=[&](std::size_t n){return r.buffer(n*2);};auto ints=[&](std::size_t n){return r.buffer(n*4);};
            r.input_f32=r.buffer(input.size()*4);r.cnn_input=bf(input.size());r.cnn1=bf(static_cast<std::size_t>(chunks)*conv_channels*64*50);r.cnn2=bf(static_cast<std::size_t>(chunks)*conv_channels*32*25);r.cnn3=bf(static_cast<std::size_t>(chunks)*conv_channels*16*13);
            r.cnn_flat=bf(static_cast<std::size_t>(chunks)*13*conv_channels*16);r.cnn_projected=bf(static_cast<std::size_t>(chunks)*13*audio_width);
            r.audio_hidden=bf(audio_tokens*audio_width);r.audio_norm=bf(audio_tokens*audio_width);r.audio_qkv=bf(audio_tokens*audio_width*3);r.audio_attn=bf(audio_tokens*audio_width);r.audio_ffn=bf(audio_tokens*4096);r.audio_delta=bf(audio_tokens*audio_width);r.audio_projected=bf(audio_tokens*audio_width);r.features_out=bf(audio_tokens*hidden_size);
            r.valid_indices=ints(indices.size());r.cu_seqlens=ints(cu.size());r.key_begin=ints(begin.size());r.key_end=ints(end.size());
            int n=std::max(max_prompt,r.batch),qw=query_heads*head_dim,kw=kv_heads*head_dim;
            r.hidden=bf(n*hidden_size);r.norm=bf(n*hidden_size);r.qkv=bf(static_cast<std::size_t>(n)*(qw+2*kw));r.query=bf(n*qw);r.key=bf(n*kw);r.value=bf(n*kw);r.attention=bf(n*qw);r.delta=bf(n*hidden_size);r.gate_up=bf(static_cast<std::size_t>(n)*2*intermediate);r.ffn=bf(n*intermediate);
            r.logits=bf(r.batch*vocab);r.float_output=r.buffer(static_cast<std::size_t>(audio_tokens)*4096*4);
            r.prompt_ids=ints(max_prompt);r.audio_rows=ints(max_prompt);r.prefill_positions=ints(max_prompt);r.prefill_begin=ints(max_prompt);r.prefill_end=ints(max_prompt);r.decode_positions=ints(r.batch);r.decode_ids=ints(r.batch);r.empty_audio_rows=ints(r.batch);
            r.k_cache=bf(static_cast<std::size_t>(layers)*r.batch*r.capacity*kw);r.v_cache=bf(static_cast<std::size_t>(layers)*r.batch*r.capacity*kw);
            r.arg_values=r.buffer(r.batch*((vocab+1023)/1024)*4);r.arg_indices=ints(r.batch*((vocab+1023)/1024));
            std::vector<int> positions(max_prompt);std::iota(positions.begin(),positions.end(),0);r.prefill_positions.copy_from_host(positions.data(),positions.size()*4);
            std::vector<int> starts(max_prompt,0),ends(max_prompt,max_prompt);r.prefill_begin.copy_from_host(starts.data(),starts.size()*4);r.prefill_end.copy_from_host(ends.data(),ends.size()*4);
            std::vector<int> empty(r.batch,-1);r.empty_audio_rows.copy_from_host(empty.data(),empty.size()*4);
            r.valid_indices.copy_from_host(indices.data(),indices.size()*4);r.cu_seqlens.copy_from_host(cu.data(),cu.size()*4);r.key_begin.copy_from_host(begin.data(),begin.size()*4);r.key_end.copy_from_host(end.data(),end.size()*4);
            // Chunk convolutions are independent. Bound the plan batch so
            // longer recordings keep tensor-op plans within the same workspace.
            int chunk_batch=std::min(chunks,kConvolutionChunkBatch);
            r.conv.emplace_back(std::make_unique<Convolution>(cudnn,chunk_batch,1,128,100,conv_channels));r.conv.emplace_back(std::make_unique<Convolution>(cudnn,chunk_batch,conv_channels,64,50,conv_channels));r.conv.emplace_back(std::make_unique<Convolution>(cudnn,chunk_batch,conv_channels,32,25,conv_channels));
            if(chunks>chunk_batch&&chunks%chunk_batch){int tail=chunks%chunk_batch;
                r.conv.emplace_back(std::make_unique<Convolution>(cudnn,tail,1,128,100,conv_channels));r.conv.emplace_back(std::make_unique<Convolution>(cudnn,tail,conv_channels,64,50,conv_channels));r.conv.emplace_back(std::make_unique<Convolution>(cudnn,tail,conv_channels,32,25,conv_channels));}
            run=std::move(pending);signature=key.str();
        }
        run->input_f32.copy_from_host(input.data(),input.size()*4);
    }

    void encode_audio(const SpeechRunOptions& execution){auto& r=*run;int t=r.audio_tokens;
        ops::float_to_bf16(static_cast<float*>(r.input_f32.p),r.cnn_input.p,r.chunks*128*100,device.stream);
        void* inputs[]={r.cnn_input.p,r.cnn1.p,r.cnn2.p};void* outputs[]={r.cnn1.p,r.cnn2.p,r.cnn3.p};
        float alpha=1,beta=0;
        const std::size_t input_chunk_elements[]={128*100,static_cast<std::size_t>(conv_channels)*64*50,static_cast<std::size_t>(conv_channels)*32*25};
        int chunk_batch=std::min(r.chunks,kConvolutionChunkBatch);
        for(int i=0;i<3;++i){std::string prefix="model.audio_tower.conv2d"+std::to_string(i+1);
            for(int start=0;start<r.chunks;start+=chunk_batch){
                auto& plan=*r.conv[i+(r.chunks-start<chunk_batch?3:0)];
                require(plan.workspace_bytes<=convolution_workspace.bytes,"ASR convolution workspace capacity mismatch");
                auto* input=static_cast<std::uint16_t*>(inputs[i])+start*input_chunk_elements[i];
                auto* output=static_cast<std::uint16_t*>(outputs[i])+static_cast<std::size_t>(start)*conv_channels*plan.out_h*plan.out_w;
                dnn(cudnnConvolutionForward(cudnn,&alpha,plan.input,input,plan.filter,model.at(prefix+".weight").data,plan.convolution,plan.algorithm,convolution_workspace.p,plan.workspace_bytes,&beta,plan.output,output));
            }
            auto& plan=*r.conv[i];
            ops::rounded_bias_gelu(outputs[i],model.at(prefix+".bias").data,r.chunks*conv_channels*plan.out_h*plan.out_w,conv_channels,plan.out_h*plan.out_w,true,device.stream);
        }
        ops::conv_to_tokens(r.cnn3.p,r.cnn_flat.p,r.chunks,conv_channels,16,13,device.stream);
        linear(r.cnn_flat.p,r.chunks*13,"model.audio_tower.conv_out",r.cnn_projected.p,execution);
        ops::add_chunk_positions(r.cnn_projected.p,model.at("audio.position_embedding").data,r.chunks,13,audio_width,device.stream);
        ops::gather_rows(r.cnn_projected.p,static_cast<int*>(r.valid_indices.p),r.audio_hidden.p,t,audio_width,device.stream);
        for(int i=0;i<audio_layers;++i){std::string prefix="model.audio_tower.layers."+std::to_string(i);
            norm(r.audio_hidden.p,prefix+".self_attn_layer_norm",r.audio_norm.p,t,audio_width,false);
            linear(r.audio_norm.p,t,prefix+".self_attn.qkv_proj",r.audio_qkv.p,execution);
            auto* packed=static_cast<std::uint16_t*>(r.audio_qkv.p);
            if(execution.audio_flash_attention){Tensor q(packed,DType::BF16,{64,16,t}),k(packed+1024,DType::BF16,{64,16,t}),v(packed+2048,DType::BF16,{64,16,t}),out(r.audio_attn.p,DType::BF16,{64,16,t});
                q.nb[2]=k.nb[2]=v.nb[2]=3072*2;
                Tensor boundaries(r.cu_seqlens.p,DType::I32,{static_cast<int>(r.cu_seqlens.bytes/4)});
                ops::packed_softmax_attention(q,k,v,{64,16,16},.125f,boundaries,workspace,out,device.stream);
            }else ops::dense_bf16_attention(packed,packed+1024,packed+2048,r.audio_attn.p,t,t,64,16,16,3072,3072,static_cast<int*>(r.key_begin.p),static_cast<int*>(r.key_end.p),nullptr,false,device.stream);
            linear(r.audio_attn.p,t,prefix+".self_attn.out_proj",r.audio_delta.p,execution);residual(r.audio_hidden.p,r.audio_delta.p,t*audio_width);
            norm(r.audio_hidden.p,prefix+".final_layer_norm",r.audio_norm.p,t,audio_width,false);
            linear(r.audio_norm.p,t,prefix+".fc1",r.audio_ffn.p,execution,true);linear(r.audio_ffn.p,t,prefix+".fc2",r.audio_delta.p,execution);residual(r.audio_hidden.p,r.audio_delta.p,t*audio_width);
        }
        norm(r.audio_hidden.p,"model.audio_tower.ln_post",r.audio_norm.p,t,audio_width,false);
        linear(r.audio_norm.p,t,"model.multi_modal_projector.linear_1",r.audio_projected.p,execution,true);
        linear(r.audio_projected.p,t,"model.multi_modal_projector.linear_2",r.features_out.p,execution);
    }

    void decoder(int tokens,int lane,bool decoding,const SpeechRunOptions& execution){auto& r=*run;int qw=query_heads*head_dim,kw=kv_heads*head_dim;
        auto* positions=static_cast<int*>(decoding?r.decode_positions.p:r.prefill_positions.p);
        for(int i=0;i<layers;++i){std::string prefix="model.language_model.layers."+std::to_string(i);
            norm(r.hidden.p,prefix+".input_layernorm",r.norm.p,tokens,hidden_size,true);
            linear(r.norm.p,tokens,prefix+".self_attn.qkv_proj",r.qkv.p,execution);
            const auto* inv=static_cast<float*>(model.at("text.rope_inv_freq").data);
            if(execution.fused_qk_norm_rope)ops::qk_norm_rope(r.qkv.p,model.at(prefix+".self_attn.q_norm.weight").data,model.at(prefix+".self_attn.k_norm.weight").data,inv,positions,r.query.p,r.key.p,r.value.p,tokens,head_dim,query_heads,kv_heads,1e-6f,device.stream);
            else{ops::split_qkv(r.qkv.p,r.query.p,r.key.p,r.value.p,tokens,qw,kw,device.stream);
                ops::rounded_rmsnorm(r.query.p,model.at(prefix+".self_attn.q_norm.weight").data,r.attention.p,tokens*query_heads,head_dim,1e-6f,device.stream);
                CUDA_CHECK(cudaMemcpyAsync(r.query.p,r.attention.p,tokens*qw*2,cudaMemcpyDeviceToDevice,device.stream));
                ops::rounded_rmsnorm(r.key.p,model.at(prefix+".self_attn.k_norm.weight").data,r.attention.p,tokens*kv_heads,head_dim,1e-6f,device.stream);
                CUDA_CHECK(cudaMemcpyAsync(r.key.p,r.attention.p,tokens*kw*2,cudaMemcpyDeviceToDevice,device.stream));
                ops::rounded_rope(r.query.p,inv,positions,tokens,query_heads,head_dim,device.stream);ops::rounded_rope(r.key.p,inv,positions,tokens,kv_heads,head_dim,device.stream);}
            std::size_t layer_offset=static_cast<std::size_t>(i)*r.batch*r.capacity*kw;
            auto* ck=static_cast<std::uint16_t*>(r.k_cache.p)+layer_offset;
            auto* cv=static_cast<std::uint16_t*>(r.v_cache.p)+layer_offset;
            if(!decoding){ck+=static_cast<std::size_t>(lane)*r.capacity*kw;cv+=static_cast<std::size_t>(lane)*r.capacity*kw;}
            ops::append_contiguous_kv(r.key.p,r.value.p,ck,cv,tokens,kw,r.capacity,positions,decoding,device.stream);
            if(decoding)ops::dense_bf16_decode_attention(r.query.p,ck,cv,r.attention.p,r.batch,r.capacity,head_dim,query_heads,kv_heads,positions,workspace,device.stream);
            else if(execution.causal_tensorcore_prefill)ops::causal_bf16_attention(r.query.p,ck,cv,r.attention.p,tokens,query_heads,kv_heads,qw,kw,device.stream,device.multiprocessor_count());
            else ops::dense_bf16_attention(r.query.p,ck,cv,r.attention.p,tokens,tokens,head_dim,query_heads,kv_heads,qw,kw,static_cast<int*>(r.prefill_begin.p),static_cast<int*>(r.prefill_end.p),positions,true,device.stream);
            project_residual(r.attention.p,tokens,prefix+".self_attn.o_proj",r.hidden.p,execution);
            norm(r.hidden.p,prefix+".post_attention_layernorm",r.norm.p,tokens,hidden_size,true);
            if(execution.linear==SpeechLinearBackend::Native){auto w=model.matrix(prefix+".mlp.gate_up_proj.weight");Tensor x(r.norm.p,DType::BF16,{hidden_size,tokens}),out(r.ffn.p,DType::BF16,{intermediate,tokens});ops::linear_swiglu(x,w,out,ops::LinearPolicy::A16Only,workspace,device.stream);}
            else{linear(r.norm.p,tokens,prefix+".mlp.gate_up_proj",r.gate_up.p,execution);ops::rounded_swiglu(r.gate_up.p,r.ffn.p,tokens,intermediate,device.stream);}
            project_residual(r.ffn.p,tokens,prefix+".mlp.down_proj",r.hidden.p,execution);
        }
        if(decoding){norm(r.hidden.p,"model.language_model.norm",r.norm.p,r.batch,hidden_size,true);linear(r.norm.p,r.batch,"model.language_model.embed_tokens",r.logits.p,execution);
            ops::bf16_argmax(r.logits.p,static_cast<float*>(r.arg_values.p),static_cast<int*>(r.arg_indices.p),static_cast<int*>(r.decode_ids.p),r.batch,vocab,device.stream);ops::advance_positions(positions,r.batch,device.stream);}
        else{auto* last=static_cast<std::uint16_t*>(r.hidden.p)+static_cast<std::size_t>(tokens-1)*hidden_size;
            norm(last,"model.language_model.norm",r.norm.p,1,hidden_size,true);linear(r.norm.p,1,"model.language_model.embed_tokens",r.logits.p,execution);
            ops::bf16_argmax(r.logits.p,static_cast<float*>(r.arg_values.p),static_cast<int*>(r.arg_indices.p),static_cast<int*>(r.decode_ids.p)+lane,1,vocab,device.stream);}
    }
    void decode_step(const SpeechRunOptions& execution){auto& r=*run;
        ops::embed_audio_tokens(model.at("model.language_model.embed_tokens.weight").data,r.features_out.p,static_cast<int*>(r.decode_ids.p),static_cast<int*>(r.empty_audio_rows.p),r.hidden.p,r.batch,hidden_size,device.stream);
        decoder(r.batch,0,true,execution);
    }
    cudaGraphExec_t captured(std::map<unsigned,cudaGraphExec_t>& cache,unsigned key,const std::function<void()>& enqueue){
        if(auto found=cache.find(key);found!=cache.end())return found->second;
        enqueue();device.synchronize();cudaGraph_t graph=nullptr;
        CUDA_CHECK(cudaStreamBeginCapture(device.stream,cudaStreamCaptureModeGlobal));
        try{enqueue();}catch(...){cudaGraph_t abandoned=nullptr;cudaStreamEndCapture(device.stream,&abandoned);if(abandoned)cudaGraphDestroy(abandoned);throw;}
        CUDA_CHECK(cudaStreamEndCapture(device.stream,&graph));cudaGraphExec_t exec=nullptr;
        auto status=cudaGraphInstantiate(&exec,graph,0);cudaGraphDestroy(graph);CUDA_CHECK(status);cache.emplace(key,exec);return exec;
    }
    SpeechResult transcribe(std::vector<SpeechFeatures> samples,const SpeechRunOptions& execution){
        std::lock_guard guard(mutex);device.bind_to_current_thread();auto wall_start=std::chrono::steady_clock::now();
        require(execution.max_new_tokens>0&&std::all_of(samples.begin(),samples.end(),[&](const auto& sample){return sample.prompt_tokens.size()+static_cast<std::uint64_t>(execution.max_new_tokens)<=options.max_context;}),"ASR output budget exceeds context");
        prepare(samples);auto& r=*run;
        require(execution.max_new_tokens>0&&static_cast<std::uint64_t>(r.max_prompt)+execution.max_new_tokens<=static_cast<std::uint64_t>(r.capacity),"ASR output budget exceeds context");
        SpeechResult result;result.token_ids.resize(r.batch);result.weight_bytes=model.weight_bytes;result.runtime_bytes=r.bytes+convolution_workspace.bytes+blas_workspace.bytes+workspace.capacity();
        CudaEventTimer timer(device);cudaGraphExec_t audio_exec=nullptr;
        if(execution.audio_graph)audio_exec=captured(r.audio_graphs,static_cast<unsigned>(execution.linear)*2+execution.audio_flash_attention,[&]{encode_audio(execution);});
        timer.start();if(audio_exec)CUDA_CHECK(cudaGraphLaunch(audio_exec,device.stream));else encode_audio(execution);result.audio_ms=timer.stop_ms();
        result.audio_graph_used=execution.audio_graph;result.prefill_graph_used=execution.prefill_graph;
        for(int lane=0;lane<r.batch;++lane){const auto& sample=samples[lane];int tokens=sample.prompt_tokens.size(),audio_row=r.audio_offsets[lane];
            std::vector<int> rows(tokens,-1);for(int t=0;t<tokens;++t)if(sample.prompt_tokens[t]==sample.audio_token_id)rows[t]=audio_row++;
            r.prompt_ids.copy_from_host(sample.prompt_tokens.data(),tokens*4);r.audio_rows.copy_from_host(rows.data(),tokens*4);
            auto prefill=[&]{ops::embed_audio_tokens(model.at("model.language_model.embed_tokens.weight").data,r.features_out.p,static_cast<int*>(r.prompt_ids.p),static_cast<int*>(r.audio_rows.p),r.hidden.p,tokens,hidden_size,device.stream);decoder(tokens,lane,false,execution);};
            cudaGraphExec_t prefill_exec=nullptr;if(execution.prefill_graph)prefill_exec=captured(r.prefill_graphs,static_cast<unsigned>(execution.linear)*64+execution.causal_tensorcore_prefill*32+execution.fused_qk_norm_rope*16+execution.fused_projection_residual*8+lane,prefill);
            timer.start();if(prefill_exec)CUDA_CHECK(cudaGraphLaunch(prefill_exec,device.stream));else prefill();result.language_prefill_ms+=timer.stop_ms();result.prompt_tokens+=tokens;
        }
        r.decode_positions.copy_from_host(r.prompt_lengths.data(),r.batch*4);
        std::vector<int> first(r.batch),output(r.batch);r.decode_ids.copy_to_host(first.data(),r.batch*4);std::vector<bool> finished(r.batch,false);
        for(int lane=0;lane<r.batch;++lane){result.token_ids[lane].push_back(first[lane]);finished[lane]=first[lane]==151643||first[lane]==151645;}
        unsigned graph_key=static_cast<unsigned>(execution.linear)*8+execution.fused_qk_norm_rope*4+execution.fused_projection_residual*2;
        if(execution.decode_graph&&!r.graphs.contains(graph_key)){
            decode_step(execution);device.synchronize();r.decode_positions.copy_from_host(r.prompt_lengths.data(),r.batch*4);r.decode_ids.copy_from_host(first.data(),r.batch*4);
            cudaGraph_t graph=nullptr;CUDA_CHECK(cudaStreamBeginCapture(device.stream,cudaStreamCaptureModeGlobal));decode_step(execution);CUDA_CHECK(cudaStreamEndCapture(device.stream,&graph));
            cudaGraphExec_t exec=nullptr;CUDA_CHECK(cudaGraphInstantiate(&exec,graph,0));CUDA_CHECK(cudaGraphDestroy(graph));r.graphs.emplace(graph_key,exec);
        }
        result.decode_graph_used=execution.decode_graph;
        for(unsigned step=1;step<execution.max_new_tokens&&!std::all_of(finished.begin(),finished.end(),[](bool value){return value;});++step){
            timer.start();if(execution.decode_graph)CUDA_CHECK(cudaGraphLaunch(r.graphs.at(graph_key),device.stream));else decode_step(execution);
            result.decode_ms+=timer.stop_ms();r.decode_ids.copy_to_host(output.data(),r.batch*4);
            for(int lane=0;lane<r.batch;++lane)if(!finished[lane]){result.token_ids[lane].push_back(output[lane]);++result.decode_tokens;finished[lane]=output[lane]==151643||output[lane]==151645;}
        }
        device.synchronize();result.wall_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall_start).count();return result;
    }
};

Program::Program(const EngineOptions& options,DeviceContext& device){auto start=std::chrono::steady_clock::now();impl_=std::make_unique<Impl>(options,device);impl_->load_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();}
Program::~Program()=default;
SpeechResult Program::transcribe(std::vector<SpeechFeatures> samples,const SpeechRunOptions& options){return impl_->transcribe(std::move(samples),options);}
LoadSummary Program::load_summary()const{LoadSummary result;result.architecture="Qwen3ASRForConditionalGeneration";result.model_name="qwen3-asr-1.7b-bf16";result.weight_formats={"bf16"};result.load_seconds=impl_->load_seconds;result.host_to_device_bytes=impl_->model.weight_bytes;return result;}
MemorySummary Program::memory_summary()const{MemorySummary result;result.device=impl_->options.device;result.max_context=impl_->options.max_context;result.kv_cache=KvCacheStorage::BFloat16;
    auto fixed=impl_->convolution_workspace.bytes+impl_->blas_workspace.bytes+impl_->workspace.capacity();result.workspace={fixed,fixed,fixed};
    result.weights={impl_->model.weight_bytes,impl_->model.weight_bytes,impl_->model.weight_bytes};if(impl_->run){const auto& r=*impl_->run;result.kv_payload_bytes=r.k_cache.bytes+r.v_cache.bytes;result.sequence={result.kv_payload_bytes,result.kv_payload_bytes,result.kv_payload_bytes};
        auto workspace=r.bytes-result.kv_payload_bytes+impl_->convolution_workspace.bytes+impl_->blas_workspace.bytes+impl_->workspace.capacity();result.workspace={workspace,workspace,workspace};}return result;}
} // namespace ninfer::models::qwen3_asr
