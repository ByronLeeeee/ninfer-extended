#include "ninfer/engine.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using Json=nlohmann::ordered_json;
template<class T> std::vector<T> binary(const std::filesystem::path& path,std::size_t count){
    std::ifstream stream(path,std::ios::binary|std::ios::ate);
    if(!stream||static_cast<std::size_t>(stream.tellg())!=count*sizeof(T))throw std::invalid_argument("Invalid alignment feature file: "+path.string());
    stream.seekg(0);std::vector<T> result(count);stream.read(reinterpret_cast<char*>(result.data()),count*sizeof(T));
    if(!stream)throw std::runtime_error("Failed to read alignment features");return result;
}
Json execute(ninfer::Engine& engine,const Json& request,const ninfer::AlignmentRunOptions& options,int warmups,int repeats){
    std::vector<ninfer::SpeechFeatures> samples;
    for(const auto& row:request.at("samples")){
        ninfer::SpeechFeatures x;x.frames=row.at("frames");x.mel_bins=row.value("bins",128);
        if(x.frames<=0||x.mel_bins!=128)throw std::invalid_argument("Invalid alignment feature extent");
        x.mel=binary<float>(row.at("features_path").get<std::string>(),static_cast<std::size_t>(x.frames)*x.mel_bins);
        x.frame_mask=binary<std::int32_t>(row.at("mask_path").get<std::string>(),x.frames);
        x.prompt_tokens=row.at("prompt_ids").get<std::vector<std::int32_t>>();samples.push_back(std::move(x));
    }
    for(int i=0;i<warmups;++i)engine.align_features(samples,options);
    Json runs=Json::array();
    for(int i=0;i<repeats;++i){auto result=engine.align_features(samples,options);
        runs.push_back({{"timestamp_classes",result.timestamp_classes},{"timestamp_segment_ms",result.timestamp_segment_ms},
            {"audio_ms",result.audio_ms},{"language_ms",result.language_ms},{"wall_seconds",result.wall_seconds},
            {"prompt_tokens",result.prompt_tokens},{"weight_bytes",result.weight_bytes},{"runtime_bytes",result.runtime_bytes},
            {"kv_bytes",engine.memory_summary().kv_payload_bytes},{"audio_graph",result.audio_graph_used},{"prefill_graph",result.prefill_graph_used}});
    }
    return {{"id",request.value("id",std::string{})},{"runs",runs}};
}
int main(int argc,char** argv){try{
    std::filesystem::path artifact,input,out;int context=8192,batch=4,repeats=1,warmups=0;bool serve=false;
    ninfer::AlignmentRunOptions run;
    for(int i=1;i<argc;++i){std::string arg=argv[i];auto value=[&]{if(++i==argc)throw std::invalid_argument("Missing argument");return std::string(argv[i]);};
        if(arg=="--artifact")artifact=value();else if(arg=="--input")input=value();else if(arg=="--out")out=value();
        else if(arg=="--context")context=std::stoi(value());else if(arg=="--batch")batch=std::stoi(value());
        else if(arg=="--repeats")repeats=std::stoi(value());else if(arg=="--warmups")warmups=std::stoi(value());
        else if(arg=="--backend"){auto backend=value();if(backend!="native"&&backend!="cublas")throw std::invalid_argument("Backend must be native or cublas");run.linear=backend=="native"?ninfer::SpeechLinearBackend::Native:ninfer::SpeechLinearBackend::Cublas;}
        else if(arg=="--eager")run.audio_graph=run.prefill_graph=false;
        else if(arg=="--tensorcore-prefill")run.causal_tensorcore_prefill=true;
        else if(arg=="--scalar-prefill")run.causal_tensorcore_prefill=false;
        else if(arg=="--serve")serve=true;
        else if(arg=="--help"){std::cout<<"ninfer-align --artifact MODEL.ninfer [--input features.json --out result.json | --serve]\n"
            "  --context N (1..8192, default 8192) --batch N (1..8, default 4)\n"
            "  --backend native|cublas --warmups N --repeats N --eager --scalar-prefill\n"
            "  Serve uses one JSON request per line, returning timestamp classes. CPU frontend owns media and word parsing.\n";return 0;}
        else throw std::invalid_argument("Unknown argument: "+arg);
    }
    if(artifact.empty()||context<1||context>8192||batch<1||batch>8||warmups<0||repeats<1||(!serve&&(input.empty()||out.empty())))throw std::invalid_argument("Invalid alignment options; use --help");
    ninfer::EngineOptions options;options.artifact_path=artifact;options.purpose=ninfer::EnginePurpose::ForcedAlignment;
    options.max_context=context;options.max_concurrency=batch;ninfer::Engine engine(options);
    if(serve){std::cout<<Json({{"ready",true},{"load_seconds",engine.load_summary().load_seconds}}).dump()<<std::endl;
        std::string line;while(std::getline(std::cin,line)){try{auto request=Json::parse(line);if(request.value("close",false))break;
            std::cout<<execute(engine,request,run,warmups,repeats).dump()<<std::endl;
        }catch(const std::exception& error){std::cout<<Json({{"error",error.what()}}).dump()<<std::endl;}}
    }else{std::ifstream stream(input);Json spec;stream>>spec;Json results=Json::array();
        for(const auto& request:spec.at("cases"))results.push_back(execute(engine,request,run,warmups,repeats));
        std::ofstream output(out);output<<Json({{"load_seconds",engine.load_summary().load_seconds},{"cases",results}}).dump(2)<<'\n';
        if(!output)throw std::runtime_error("Failed to write alignment result");}
    return 0;
}catch(const std::exception& error){std::cerr<<"Alignment: "<<error.what()<<'\n';return 1;}}
