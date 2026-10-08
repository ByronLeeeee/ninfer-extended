#include "ninfer/engine.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

using Json=nlohmann::ordered_json;
template<class T> std::vector<T> binary(const std::filesystem::path& path,std::size_t n){
    std::ifstream stream(path,std::ios::binary|std::ios::ate);
    if(!stream||static_cast<std::size_t>(stream.tellg())!=n*sizeof(T))throw std::runtime_error("Invalid feature file: "+path.string());
    stream.seekg(0);std::vector<T> result(n);stream.read(reinterpret_cast<char*>(result.data()),n*sizeof(T));
    if(!stream)throw std::runtime_error("Failed to read features");return result;
}
double median(std::vector<double> x){std::sort(x.begin(),x.end());return x[x.size()/2];}
int main(int argc,char** argv){try{
    std::filesystem::path artifact,input,out;ninfer::SpeechRunOptions run;int repeats=3,context=4096,warmups=1;bool abba=false,variants=false;
    for(int i=1;i<argc;++i){std::string arg=argv[i];auto value=[&](){if(++i==argc)throw std::invalid_argument("Missing argument");return std::string(argv[i]);};
        if(arg=="--artifact")artifact=value();else if(arg=="--input")input=value();else if(arg=="--out")out=value();
        else if(arg=="--backend"){auto name=value();if(name!="native"&&name!="cublas")throw std::invalid_argument("Backend must be native or cublas");run.linear=name=="native"?ninfer::SpeechLinearBackend::Native:ninfer::SpeechLinearBackend::Cublas;}
        else if(arg=="--repeats")repeats=std::stoi(value());else if(arg=="--context")context=std::stoi(value());
        else if(arg=="--warmups")warmups=std::stoi(value());
        else if(arg=="--max-new-tokens")run.max_new_tokens=std::stoul(value());
        else if(arg=="--no-graph")run.decode_graph=run.audio_graph=run.prefill_graph=false;
        else if(arg=="--no-decode-graph")run.decode_graph=false;else if(arg=="--no-audio-graph")run.audio_graph=false;else if(arg=="--no-prefill-graph")run.prefill_graph=false;
        else if(arg=="--no-audio-flash")run.audio_flash_attention=false;else if(arg=="--no-fuse")run.fused_qk_norm_rope=false;
        else if(arg=="--no-residual-fuse")run.fused_projection_residual=false;
        else if(arg=="--tensorcore-prefill")run.causal_tensorcore_prefill=true;
        else if(arg=="--abba")abba=true;
        else if(arg=="--variants")variants=true;
        else if(arg=="--help"){std::cout<<"ninfer-asr --artifact model.ninfer --input features.json --out result.json\n"
            "[--backend native|cublas] [--context 4096] [--max-new-tokens 1024] [--repeats 3] [--warmups 1]\n"
            "[--no-graph] [--no-audio-graph] [--no-prefill-graph] [--no-decode-graph]\n"
            "[--no-audio-flash] [--no-fuse] [--no-residual-fuse] [--tensorcore-prefill] [--abba] [--variants]\n"
            "Input accepts {samples:[...]} or {cases:[{id:...,samples:[...]}]}; one to four lanes.\n"
            "CPU frontend supplies log-mel features and audio-token prompts. Encoder and decoder run natively in BF16.\n";return 0;}
        else throw std::invalid_argument("Unknown option: "+arg);
    }
    if(artifact.empty()||input.empty()||out.empty()||repeats<1||context<1||warmups<0)throw std::invalid_argument("Artifact, input and output are required; warmups must be nonnegative");
    std::ifstream file(input);Json spec;file>>spec;
    ninfer::EngineOptions options;options.artifact_path=artifact;options.purpose=ninfer::EnginePurpose::SpeechRecognition;options.max_context=context;options.max_concurrency=4;options.kv_cache=ninfer::KvCacheStorage::BFloat16;
    ninfer::Engine engine(options);
    const bool suite=spec.contains("cases");Json requests=suite?spec.at("cases"):Json::array({spec});
    Json combined={{"artifact",artifact.string()},{"dtype","bf16"},{"kv","bf16"},{"context",context},{"load_seconds",engine.load_summary().load_seconds},{"cases",Json::array()}};
    for(const auto& request:requests){std::vector<ninfer::SpeechFeatures> samples;double audio_seconds=0;
    for(const auto& row:request.at("samples")){ninfer::SpeechFeatures feature;feature.frames=row.at("frames");feature.mel_bins=row.at("bins");
        feature.mel=binary<float>(row.at("features_path").get<std::string>(),static_cast<std::size_t>(feature.frames)*feature.mel_bins);
        feature.frame_mask=binary<std::int32_t>(row.at("mask_path").get<std::string>(),feature.frames);feature.prompt_tokens=row.at("prompt_ids").get<std::vector<std::int32_t>>();feature.audio_token_id=row.at("audio_token_id");
        samples.push_back(std::move(feature));audio_seconds+=row.at("audio_seconds").get<double>();}
    combined["cases"].push_back({{"id",request.value("id",input.stem().string())},{"artifact",artifact.string()},{"input",input.string()},{"dtype","bf16"},{"kv","bf16"},{"batch",samples.size()},{"context",context},{"load_seconds",engine.load_summary().load_seconds},{"runs",Json::array()}});
    auto& result=combined["cases"].back();
    auto execute=[&](ninfer::SpeechRunOptions setting,bool keep){auto measured=engine.transcribe_features(samples,setting);if(!keep)return;
        auto tps=measured.decode_ms>0?measured.decode_tokens/(measured.decode_ms/1000):0;
        Json row={{"backend",setting.linear==ninfer::SpeechLinearBackend::Native?"native":"cublas"},{"graph",measured.decode_graph_used},{"audio_flash",setting.audio_flash_attention},{"fused_qk_norm_rope",setting.fused_qk_norm_rope},{"fused_projection_residual",setting.fused_projection_residual},{"causal_tensorcore_prefill",setting.causal_tensorcore_prefill},
                  {"audio_graph",measured.audio_graph_used},{"prefill_graph",measured.prefill_graph_used},{"audio_ms",measured.audio_ms},{"language_prefill_ms",measured.language_prefill_ms},{"decode_ms",measured.decode_ms},{"wall_seconds",measured.wall_seconds},{"rtf",measured.wall_seconds/audio_seconds},
                  {"prompt_tokens",measured.prompt_tokens},{"decode_tokens",measured.decode_tokens},{"decode_tps",tps},{"language_prefill_tps",measured.prompt_tokens/(measured.language_prefill_ms/1000)},
                  {"weight_bytes",measured.weight_bytes},{"runtime_bytes",measured.runtime_bytes},{"token_ids",measured.token_ids}};
        result["runs"].push_back(row);std::cout<<row.dump()<<std::endl;std::ofstream stream(out);stream<<(suite?combined:result).dump(2)<<'\n';};
    for(int n=0;n<warmups;++n)execute(run,false);
    if(variants){std::vector<ninfer::SpeechRunOptions> settings;auto setting=run;setting.linear=ninfer::SpeechLinearBackend::Cublas;setting.decode_graph=setting.audio_graph=setting.prefill_graph=setting.audio_flash_attention=setting.fused_qk_norm_rope=false;settings.push_back(setting);
        setting.decode_graph=true;settings.push_back(setting);setting.audio_flash_attention=true;settings.push_back(setting);setting.fused_qk_norm_rope=true;settings.push_back(setting);
        setting.linear=ninfer::SpeechLinearBackend::Native;settings.push_back(setting);setting.audio_graph=setting.prefill_graph=true;settings.push_back(setting);
        for(auto option:settings)execute(option,false);for(int n=0;n<repeats;++n){if(n%2==0)for(auto option:settings)execute(option,true);else for(auto i=settings.rbegin();i!=settings.rend();++i)execute(*i,true);}}
    else if(abba){auto ref=run;ref.linear=ninfer::SpeechLinearBackend::Cublas;execute(ref,false);for(int n=0;n<repeats;++n){execute(ref,true);execute(run,true);execute(run,true);execute(ref,true);}}
    else for(int n=0;n<repeats;++n)execute(run,true);
    }
    return 0;
}catch(const std::exception& error){std::cerr<<"ASR: "<<error.what()<<'\n';return 1;}}
