#include "ninfer/engine.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using Json=nlohmann::ordered_json;
double median(std::vector<double> values){std::sort(values.begin(),values.end());return values[values.size()/2];}

int main(int argc,char** argv){try{
    std::filesystem::path artifact,input,out;
    ninfer::EmbeddingRunOptions execution;
    int context=32768,batch=8,warmups=2,repeats=7;bool variants=false;
    for(int i=1;i<argc;++i){std::string arg=argv[i];auto value=[&](){if(++i==argc)throw std::invalid_argument("Missing option value");return std::string(argv[i]);};
        if(arg=="--artifact")artifact=value();else if(arg=="--input")input=value();else if(arg=="--out")out=value();
        else if(arg=="--backend"){auto name=value();if(name!="native"&&name!="cublas")throw std::invalid_argument("Backend must be native or cublas");execution.linear=name=="native"?ninfer::EmbeddingLinearBackend::Native:ninfer::EmbeddingLinearBackend::Cublas;}
        else if(arg=="--context")context=std::stoi(value());else if(arg=="--batch")batch=std::stoi(value());
        else if(arg=="--dimensions")execution.dimensions=std::stoul(value());
        else if(arg=="--warmups")warmups=std::stoi(value());else if(arg=="--repeats")repeats=std::stoi(value());
        else if(arg=="--no-graph")execution.cuda_graph=false;else if(arg=="--no-normalize")execution.normalize=false;
        else if(arg=="--attention"){auto name=value();if(name=="auto")execution.attention=ninfer::EmbeddingAttention::Automatic;
            else if(name=="fp32")execution.attention=ninfer::EmbeddingAttention::Float32;
            else if(name=="tensorcore")execution.attention=ninfer::EmbeddingAttention::TensorCore;
            else throw std::invalid_argument("Attention must be auto, fp32 or tensorcore");}
        else if(arg=="--tensorcore-attention")execution.attention=ninfer::EmbeddingAttention::TensorCore;else if(arg=="--variants")variants=true;
        else if(arg=="--help"){std::cout<<"ninfer-embed --artifact model.ninfer --input tokens.json --out vectors.json\n"
            "[--backend native|cublas] [--context 32768] [--batch 8] [--dimensions 1024]\n"
            "[--warmups 2] [--repeats 7] [--no-graph] [--no-normalize] [--attention auto|fp32|tensorcore] [--variants]\n"
            "Input: {input_ids:[[...],...]} or {cases:[{id:...,input_ids:[[...],...]}]}.\n"
            "Context limits the total unpadded tokens per batch; weights stay BF16.\n";return 0;}
        else throw std::invalid_argument("Unknown option: "+arg);
    }
    if(artifact.empty()||input.empty()||out.empty()||context<1||batch<1||repeats<1||warmups<0)throw std::invalid_argument("Invalid arguments");
    std::ifstream file(input);Json specification;file>>specification;
    ninfer::EngineOptions options;options.artifact_path=artifact;options.purpose=ninfer::EnginePurpose::TextEmbedding;
    options.max_context=context;options.max_concurrency=batch;
    ninfer::Engine engine(options);
    Json report={{"artifact",artifact.string()},{"context",context},{"max_batch",batch},{"dtype","bf16"},
                 {"load_seconds",engine.load_summary().load_seconds},{"cases",Json::array()}};
    const auto requests=specification.contains("cases")?specification.at("cases"):Json::array({specification});
    for(const auto& request:requests){
        const auto ids=request.at("input_ids").get<std::vector<std::vector<ninfer::TokenId>>>();
        Json entry={{"id",request.value("id",input.stem().string())},{"batch",ids.size()},{"variants",Json::array()}};
        std::vector<ninfer::EmbeddingRunOptions> settings={execution};
        if(variants){settings.clear();for(auto backend:{ninfer::EmbeddingLinearBackend::Cublas,ninfer::EmbeddingLinearBackend::Native})
            for(auto attention:{ninfer::EmbeddingAttention::Float32,ninfer::EmbeddingAttention::TensorCore}){auto setting=execution;setting.linear=backend;setting.attention=attention;settings.push_back(setting);}}
        for(auto setting:settings){for(int n=0;n<warmups;++n)engine.embed_tokens(ids,setting);}
        std::vector<std::vector<double>> gpu(settings.size()),wall(settings.size());
        std::vector<ninfer::EmbeddingResult> last(settings.size());
        // Alternate the order of implementations to reduce temperature/order bias.
        for(int n=0;n<repeats;++n)for(std::size_t j=0;j<settings.size();++j){
            const auto index=n%2==0?j:settings.size()-1-j;
            const auto started=std::chrono::steady_clock::now();
            last[index]=engine.embed_tokens(ids,settings[index]);
            wall[index].push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count());
            gpu[index].push_back(last[index].gpu_ms);
        }
        for(std::size_t i=0;i<settings.size();++i){const auto& result=last[i];const auto& setting=settings[i];
            Json row={{"backend",setting.linear==ninfer::EmbeddingLinearBackend::Native?"native":"cublas"},
                {"tensorcore_attention",result.tensorcore_attention_used},{"graph",result.graph_used},{"input_tokens",result.input_tokens},
                {"gpu_ms",median(gpu[i])},{"wall_ms",median(wall[i])},{"tokens_per_second",result.input_tokens*1000/median(gpu[i])},
                {"texts_per_second",ids.size()*1000/median(wall[i])},{"weight_bytes",result.weight_bytes},
                {"runtime_bytes",result.runtime_bytes},{"gpu_runs_ms",gpu[i]},{"wall_runs_ms",wall[i]}};
            std::cout<<Json{{"id",entry["id"]},{"stats",row}}.dump()<<std::endl;
            row["vectors"]=result.vectors;entry["variants"].push_back(std::move(row));
        }
        report["cases"].push_back(std::move(entry));std::ofstream stream(out);if(!stream)throw std::runtime_error("Cannot write result");stream<<report.dump(2)<<'\n';
    }
    return 0;
}catch(const std::exception& error){std::cerr<<"Embedding: "<<error.what()<<'\n';return 1;}}
