#include "models/qwen3/embedding_program.h"
#include "artifact/reader.h"
#include "core/arena.h"
#include "core/weight.h"
#include "ninfer/ops/bf16_transforms.h"
#include "ninfer/ops/dense_attention.h"
#include "ninfer/ops/embedding_pool.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include <cublas_v2.h>
#include <chrono>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace ninfer::models::qwen3 {
namespace {
void require(bool value,const char* message){if(!value)throw std::invalid_argument(message);}
void blas(cublasStatus_t status){if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("Embedding cuBLAS failed: "+std::to_string(status));}
std::size_t aligned(std::size_t n){return (n+255)&~std::size_t(255);}
struct Parameter {void* data;std::vector<std::uint64_t> shape;DType dtype;};
class Model {
public:
    artifact::Json config;
    DeviceBuffer storage;
    std::map<std::string,Parameter,std::less<>> parameters;
    std::size_t weight_bytes=0;
    explicit Model(const std::filesystem::path& path){
        artifact::Reader reader(path);config=reader.directory().component("text").config;
        require(config.at("model_type")=="qwen3"&&config.at("architectures")==artifact::Json::array({"Qwen3ForCausalLM"}),"Embedding requires a Qwen3 decoder artifact");
        for(const auto& object:reader.directory().objects)if(const auto* t=std::get_if<artifact::TensorObject>(&object)){
            require(t->layout=="contiguous_le_v1"&&(t->format=="bf16"||(t->id=="text.rope_inv_freq"&&t->format=="fp32")),"Embedding requires contiguous BF16 parameters");
            weight_bytes=aligned(weight_bytes)+t->bytes;
        }
        storage=DeviceBuffer(weight_bytes);std::size_t offset=0;std::vector<std::byte> staging(8*1024*1024);
        for(const auto& object:reader.directory().objects)if(const auto* t=std::get_if<artifact::TensorObject>(&object)){
            offset=aligned(offset);auto* destination=static_cast<std::byte*>(storage.p)+offset;
            for(std::size_t done=0;done<t->bytes;){const auto count=std::min(staging.size(),static_cast<std::size_t>(t->bytes)-done);
                reader.read_into(t->offset+done,std::span(staging.data(),count));CUDA_CHECK(cudaMemcpy(destination+done,staging.data(),count,cudaMemcpyHostToDevice));done+=count;}
            parameters.emplace(t->id,Parameter{destination,t->shape,t->format=="bf16"?DType::BF16:DType::FP32});offset+=t->bytes;
        }
    }
    const Parameter& at(const std::string& name)const{return parameters.at(name);}
    Weight matrix(const std::string& name)const{
        const auto& p=at(name);require(p.shape.size()==2&&p.dtype==DType::BF16,"Embedding matrix representation mismatch");
        Weight w;w.qtype=QType::BF16;w.layout=QuantLayout::Contiguous;w.qdata=w.payload=p.data;w.n=p.shape[0];w.k=p.shape[1];w.ndim=2;
        w.shape[0]=w.padded_shape[0]=w.k;w.shape[1]=w.padded_shape[1]=w.n;w.payload_bytes=static_cast<std::uint64_t>(w.n)*w.k*2;return w;
    }
};
struct Run {
    int tokens=0,batch=0,dimensions=0;
    std::size_t bytes=0;
    DeviceBuffer hidden,norm,qkv,q,k,v,attention,delta,gate_up,ffn,selected,normalized,output;
    DeviceBuffer ids,last,rope_positions,causal_positions,begin,end,sequence_begin,sequence_length;
    std::map<std::vector<int>,cudaGraphExec_t> graphs;
    ~Run(){for(auto [key,graph]:graphs)cudaGraphExecDestroy(graph);}
    DeviceBuffer buffer(std::size_t n){bytes+=n;return DeviceBuffer(n);}
};
}

class EmbeddingProgram::Impl {
public:
    EngineOptions options;
    DeviceContext& device;
    Model model;
    cublasHandle_t cublas=nullptr;
    DeviceBuffer blas_workspace;
    std::unique_ptr<WorkspaceArena> workspace;
    std::unique_ptr<Run> run;
    std::mutex mutex;
    int hidden,layers,qh,kh,head,intermediate,vocab;
    double load_seconds=0;
    Impl(const EngineOptions& o,DeviceContext& d):options(o),device(d),model(o.artifact_path){
        hidden=model.config.at("hidden_size");layers=model.config.at("num_hidden_layers");qh=model.config.at("num_attention_heads");
        kh=model.config.at("num_key_value_heads");head=model.config.at("head_dim");intermediate=model.config.at("intermediate_size");vocab=model.config.at("vocab_size");
        require(hidden==1024&&layers==28&&qh==16&&kh==8&&head==128&&intermediate==3072&&vocab==151669,"Embedding decoder geometry has not been qualified");
        require(model.config.at("hidden_act")=="silu"&&!model.config.at("attention_bias").get<bool>()&&!model.config.at("use_sliding_window").get<bool>()&&
                model.config.at("rms_norm_eps")==1e-6&&model.config.at("rope_theta")==1000000&&model.config.at("rope_scaling").is_null(),"Embedding math differs from the qualified Qwen3 profile");
        require(options.max_context<=model.config.at("max_position_embeddings").get<unsigned>(),"Embedding capacity exceeds model positions");
        const auto shape=[&](const std::string& name,std::initializer_list<std::uint64_t> dims){require(model.at(name).shape==std::vector<std::uint64_t>(dims),"Embedding bound parameter shape mismatch");};
        shape("model.embed_tokens.weight",{static_cast<unsigned>(vocab),static_cast<unsigned>(hidden)});shape("model.norm.weight",{static_cast<unsigned>(hidden)});
        shape("text.rope_inv_freq",{64});
        for(int layer=0;layer<layers;++layer){auto p="model.layers."+std::to_string(layer)+".";
            shape(p+"input_layernorm.weight",{1024});shape(p+"post_attention_layernorm.weight",{1024});
            shape(p+"self_attn.q_norm.weight",{128});shape(p+"self_attn.k_norm.weight",{128});
            shape(p+"self_attn.qkv_proj.weight",{4096,1024});shape(p+"self_attn.o_proj.weight",{1024,2048});
            shape(p+"mlp.gate_up_proj.weight",{6144,1024});shape(p+"mlp.down_proj.weight",{1024,3072});}
    }
    ~Impl(){device.bind_to_current_thread_noexcept();run.reset();if(cublas)cublasDestroy(cublas);}
    void prepare_cublas(){
        if(cublas)return;
        blas_workspace=DeviceBuffer(16*1024*1024);
        blas(cublasCreate(&cublas));blas(cublasSetStream(cublas,device.stream));blas(cublasSetMathMode(cublas,CUBLAS_TENSOR_OP_MATH));
        blas(cublasSetWorkspace(cublas,blas_workspace.p,blas_workspace.bytes));
    }
    void prepare(int tokens,int batch,int dimensions){
        if(run&&run->tokens==tokens&&run->batch==batch&&run->dimensions==dimensions)return;
        device.synchronize();run.reset();workspace.reset();
        auto r=std::make_unique<Run>();r->tokens=tokens;r->batch=batch;r->dimensions=dimensions;
        auto bf=[&](int width){return r->buffer(static_cast<std::size_t>(tokens)*width*2);};
        r->hidden=bf(hidden);r->norm=bf(hidden);r->qkv=bf((qh+2*kh)*head);r->q=bf(qh*head);r->k=bf(kh*head);r->v=bf(kh*head);
        r->attention=bf(qh*head);r->ffn=bf(intermediate);
        r->selected=r->buffer(batch*hidden*2);r->normalized=r->buffer(batch*hidden*2);r->output=r->buffer(batch*dimensions*4);
        r->ids=r->buffer(tokens*4);r->rope_positions=r->buffer(tokens*4);r->causal_positions=r->buffer(tokens*4);
        r->begin=r->buffer(tokens*4);r->end=r->buffer(tokens*4);r->last=r->buffer(batch*4);
        r->sequence_begin=r->buffer(batch*4);r->sequence_length=r->buffer(batch*4);
        auto needed=ops::linear_swiglu_workspace_capacity_bytes(QType::BF16,2*intermediate,hidden,ops::LinearPolicy::A16Only,tokens,tokens);
        needed=std::max(needed,ops::linear_add_workspace_capacity_bytes(QType::BF16,hidden,qh*head,ops::LinearPolicy::A16Only,tokens,tokens));
        needed=std::max(needed,ops::linear_add_workspace_capacity_bytes(QType::BF16,hidden,intermediate,ops::LinearPolicy::A16Only,tokens,tokens));
        // WorkspaceArena requires nonempty storage even when every selected Op
        // reports zero scratch. Keep one aligned arena for the reference-based API.
        workspace=std::make_unique<WorkspaceArena>(std::max<std::size_t>(needed,256));
        run=std::move(r);
    }
    void linear(const void* input,int tokens,const std::string& name,void* output,EmbeddingLinearBackend backend){
        auto w=model.matrix(name+".weight");
        if(backend==EmbeddingLinearBackend::Native){Tensor x(const_cast<void*>(input),DType::BF16,{w.k,tokens}),y(output,DType::BF16,{w.n,tokens});ops::linear(x,w,y,device.stream);}
        else{float alpha=1,beta=0;blas(cublasGemmEx(cublas,CUBLAS_OP_T,CUBLAS_OP_N,w.n,tokens,w.k,&alpha,w.qdata,CUDA_R_16BF,w.k,
                   input,CUDA_R_16BF,w.k,&beta,output,CUDA_R_16BF,w.n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));}
    }
    void project_residual(const void* input,const std::string& name,EmbeddingLinearBackend backend){
        auto& r=*run;
        if(backend==EmbeddingLinearBackend::Native){auto w=model.matrix(name+".weight");Tensor x(const_cast<void*>(input),DType::BF16,{w.k,r.tokens}),y(r.hidden.p,DType::BF16,{w.n,r.tokens});ops::linear_add(x,w,y,ops::LinearPolicy::A16Only,*workspace,device.stream);}
        else{linear(input,r.tokens,name,r.delta.p,backend);ops::bf16_residual_add(r.hidden.p,r.delta.p,r.tokens*hidden,device.stream);}
    }
    void forward(const EmbeddingRunOptions& execution,const std::vector<int>& lengths,bool tensorcore){
        auto& r=*run;const int qw=qh*head,kw=kh*head;
        ops::gather_rows(model.at("model.embed_tokens.weight").data,static_cast<int*>(r.ids.p),r.hidden.p,r.tokens,hidden,device.stream);
        for(int i=0;i<layers;++i){auto p="model.layers."+std::to_string(i)+".";
            ops::rounded_rmsnorm(r.hidden.p,model.at(p+"input_layernorm.weight").data,r.norm.p,r.tokens,hidden,1e-6f,device.stream);
            linear(r.norm.p,r.tokens,p+"self_attn.qkv_proj",r.qkv.p,execution.linear);
            ops::qk_norm_rope(r.qkv.p,model.at(p+"self_attn.q_norm.weight").data,model.at(p+"self_attn.k_norm.weight").data,
                static_cast<float*>(model.at("text.rope_inv_freq").data),static_cast<int*>(r.rope_positions.p),r.q.p,r.k.p,r.v.p,r.tokens,head,qh,kh,1e-6f,device.stream);
            if(tensorcore){
                if(r.batch==1) ops::causal_bf16_attention(r.q.p,r.k.p,r.v.p,r.attention.p,r.tokens,qh,kh,qw,kw,device.stream,device.multiprocessor_count());
                else ops::packed_causal_bf16_attention(r.q.p,r.k.p,r.v.p,r.attention.p,r.batch,
                    *std::max_element(lengths.begin(),lengths.end()),qh,kh,qw,kw,
                    static_cast<int*>(r.sequence_begin.p),static_cast<int*>(r.sequence_length.p),device.stream,device.multiprocessor_count());
            }else ops::dense_bf16_attention(r.q.p,r.k.p,r.v.p,r.attention.p,r.tokens,r.tokens,head,qh,kh,qw,kw,
                static_cast<int*>(r.begin.p),static_cast<int*>(r.end.p),static_cast<int*>(r.causal_positions.p),true,device.stream);
            project_residual(r.attention.p,p+"self_attn.o_proj",execution.linear);
            ops::rounded_rmsnorm(r.hidden.p,model.at(p+"post_attention_layernorm.weight").data,r.norm.p,r.tokens,hidden,1e-6f,device.stream);
            if(execution.linear==EmbeddingLinearBackend::Native){auto w=model.matrix(p+"mlp.gate_up_proj.weight");Tensor x(r.norm.p,DType::BF16,{hidden,r.tokens}),y(r.ffn.p,DType::BF16,{intermediate,r.tokens});ops::linear_swiglu(x,w,y,ops::LinearPolicy::A16Only,*workspace,device.stream,device.multiprocessor_count());}
            else{linear(r.norm.p,r.tokens,p+"mlp.gate_up_proj",r.gate_up.p,execution.linear);ops::rounded_swiglu(r.gate_up.p,r.ffn.p,r.tokens,intermediate,device.stream);}
            project_residual(r.ffn.p,p+"mlp.down_proj",execution.linear);
        }
        ops::gather_rows(r.hidden.p,static_cast<int*>(r.last.p),r.selected.p,r.batch,hidden,device.stream);
        ops::rounded_rmsnorm(r.selected.p,model.at("model.norm.weight").data,r.normalized.p,r.batch,hidden,1e-6f,device.stream);
        ops::bf16_embedding_output(r.normalized.p,static_cast<float*>(r.output.p),r.batch,hidden,r.dimensions,execution.normalize,device.stream);
    }
    EmbeddingResult embed(const std::vector<std::vector<TokenId>>& sequences,const EmbeddingRunOptions& execution){
        std::lock_guard lock(mutex);device.bind_to_current_thread();
        require(!sequences.empty()&&sequences.size()<=options.max_concurrency,"Embedding batch exceeds startup capacity");
        require(execution.linear==EmbeddingLinearBackend::Native||execution.linear==EmbeddingLinearBackend::Cublas,"Embedding linear backend invalid");
        require(execution.attention==EmbeddingAttention::Automatic||execution.attention==EmbeddingAttention::Float32||execution.attention==EmbeddingAttention::TensorCore,"Embedding attention backend invalid");
        const int dimensions=execution.dimensions?execution.dimensions:hidden;require(dimensions>=1&&dimensions<=hidden,"Embedding output dimensions invalid");
        std::vector<int> ids,rope,positions,begin,end,last,lengths,starts;
        for(const auto& sequence:sequences){require(!sequence.empty()&&sequence.size()<=options.max_context,"Embedding sequences must be nonempty and fit max_context");
            const int start=ids.size(),finish=start+sequence.size();require(static_cast<unsigned>(finish)<=options.max_context,"Embedding aggregate tokens exceed max_context");
            for(int i=0;i<static_cast<int>(sequence.size());++i){const int token=sequence[i];require(token>=0&&token<vocab,"Embedding token outside vocabulary");ids.push_back(token);rope.push_back(i);positions.push_back(start+i);begin.push_back(start);end.push_back(finish);}
            last.push_back(finish-1);lengths.push_back(sequence.size());starts.push_back(start);}
        prepare(ids.size(),sequences.size(),dimensions);auto& r=*run;
        if(execution.linear==EmbeddingLinearBackend::Cublas){
            prepare_cublas();
            if(!r.delta.p)r.delta=r.buffer(static_cast<std::size_t>(r.tokens)*hidden*2);
            if(!r.gate_up.p)r.gate_up=r.buffer(static_cast<std::size_t>(r.tokens)*2*intermediate*2);
        }
        const bool tensorcore=execution.attention==EmbeddingAttention::TensorCore||
            (execution.attention==EmbeddingAttention::Automatic&&ids.size()>=64*sequences.size());
        // A Tensor Core capture fixes the maximum sequence grid extent. Sequence
        // metadata is uploaded on every replay, including equal-shape ragged batches.
        std::vector<int> key={static_cast<int>(execution.linear),execution.normalize,tensorcore};
        if(tensorcore)key.push_back(*std::max_element(lengths.begin(),lengths.end()));
        const auto started=std::chrono::steady_clock::now();
        for(auto pair:{std::pair{&r.ids,&ids},{&r.rope_positions,&rope},{&r.causal_positions,&positions},{&r.begin,&begin},{&r.end,&end},{&r.last,&last}})
            pair.first->copy_from_host(pair.second->data(),pair.second->size()*sizeof(int));
        r.sequence_begin.copy_from_host(starts.data(),starts.size()*sizeof(int));
        r.sequence_length.copy_from_host(lengths.data(),lengths.size()*sizeof(int));
        cudaGraphExec_t graph=nullptr;
        if(execution.cuda_graph){auto found=r.graphs.find(key);if(found==r.graphs.end()){
                forward(execution,lengths,tensorcore);device.synchronize();cudaGraph_t captured=nullptr;
                CUDA_CHECK(cudaStreamBeginCapture(device.stream,cudaStreamCaptureModeThreadLocal));
                try{forward(execution,lengths,tensorcore);}catch(...){cudaStreamEndCapture(device.stream,&captured);if(captured)cudaGraphDestroy(captured);throw;}
                CUDA_CHECK(cudaStreamEndCapture(device.stream,&captured));
                auto status=cudaGraphInstantiate(&graph,captured,nullptr,nullptr,0);cudaGraphDestroy(captured);CUDA_CHECK(status);
                r.graphs.emplace(key,graph);
            }else graph=found->second;}
        EmbeddingResult result;result.input_tokens=ids.size();result.weight_bytes=model.weight_bytes;
        result.runtime_bytes=r.bytes+blas_workspace.bytes+workspace->capacity();result.graph_used=graph!=nullptr;
        result.tensorcore_attention_used=tensorcore;
        CudaEventTimer timer(device);timer.start();
        if(graph)CUDA_CHECK(cudaGraphLaunch(graph,device.stream));else forward(execution,lengths,tensorcore);
        result.gpu_ms=timer.stop_ms();std::vector<float> flat(r.batch*dimensions);r.output.copy_to_host(flat.data(),flat.size()*sizeof(float));
        result.wall_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        for(int i=0;i<r.batch;++i)result.vectors.emplace_back(flat.begin()+i*dimensions,flat.begin()+(i+1)*dimensions);
        return result;
    }
};
EmbeddingProgram::EmbeddingProgram(const EngineOptions& options,DeviceContext& device){const auto start=std::chrono::steady_clock::now();impl_=std::make_unique<Impl>(options,device);impl_->load_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();}
EmbeddingProgram::~EmbeddingProgram()=default;
EmbeddingResult EmbeddingProgram::embed(const std::vector<std::vector<TokenId>>& sequences,const EmbeddingRunOptions& options){return impl_->embed(sequences,options);}
LoadSummary EmbeddingProgram::load_summary()const{LoadSummary s;s.architecture="Qwen3ForCausalLM/TextEmbedding";s.model_name="Qwen3-Embedding";s.weight_formats={"bf16"};s.load_seconds=impl_->load_seconds;s.host_to_device_bytes=impl_->model.weight_bytes;s.device_object_count=impl_->model.parameters.size();return s;}
MemorySummary EmbeddingProgram::memory_summary()const{
    MemorySummary s;s.device=impl_->options.device;s.max_context=impl_->options.max_context;
    s.weights.capacity_bytes=s.weights.used_bytes=s.weights.peak_used_bytes=impl_->model.weight_bytes;
    s.workspace.capacity_bytes=s.workspace.used_bytes=s.workspace.peak_used_bytes=
        impl_->blas_workspace.bytes+(impl_->run?impl_->run->bytes:0)+(impl_->workspace?impl_->workspace->capacity():0);
    s.runtime_reservation_bytes=s.workspace.capacity_bytes;return s;
}
} // namespace ninfer::models::qwen3
