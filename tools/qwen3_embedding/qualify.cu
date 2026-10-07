#include "ninfer/ops/embedding_pool.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/dense_attention.h"
#include "core/device.h"
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace ninfer;
namespace {
std::mt19937 random(6065070);
std::vector<__nv_bfloat16> values(std::size_t n,float scale=0.1f){
    std::normal_distribution<float> distribution(0,scale);std::vector<__nv_bfloat16> result(n);
    for(auto& value:result)value=__float2bfloat16_rn(distribution(random));return result;
}
struct Error {
    double square=0,reference_square=0,max_error=0,max_reference=0;std::size_t count=0;bool finite=true;
    void add(double actual,double reference){finite&=std::isfinite(actual)&&std::isfinite(reference);
        const auto error=actual-reference;square+=error*error;reference_square+=reference*reference;
        max_error=std::max(max_error,std::abs(error));max_reference=std::max(max_reference,std::abs(reference));++count;}
    bool report(const std::string& name,bool activation=false,bool pool=false)const{
        const auto relative=std::sqrt(square/std::max(reference_square,1e-30));
        const auto peak=max_error/std::sqrt(std::max(reference_square/count,1e-30));
        bool pass=finite&&(pool?max_error<1e-6:(activation?relative<=0.0033&&max_error<=0.005+0.0063*max_reference:relative<0.006&&peak<0.045));
        std::cout<<"{\"case\":\""<<name<<"\",\"relative_l2\":"<<relative<<",\"max_abs\":"<<max_error<<",\"oracle_outputs\":"<<count<<",\"pass\":"<<(pass?"true":"false")<<"}\n"<<std::flush;return pass;
    }
};
bool projection(int n,int k,int tokens,int mode,int multiprocessor_count=0){
    const auto x=values(k*tokens,0.5f),w=values(static_cast<std::size_t>(n)*k),initial=values(n*tokens,0.5f);
    const auto rows=mode==2?n/2:n;
    DeviceBuffer xb(x.size()*2),wb(w.size()*2),output(rows*tokens*2);
    xb.copy_from_host(x.data(),xb.bytes);wb.copy_from_host(w.data(),wb.bytes);output.copy_from_host(initial.data(),output.bytes);
    Weight weight;weight.n=n;weight.k=k;weight.qtype=QType::BF16;weight.layout=QuantLayout::Contiguous;weight.qdata=wb.p;
    Tensor input(xb.p,DType::BF16,{k,tokens}),out(output.p,DType::BF16,{rows,tokens});
    const auto capacity=mode==2?ops::linear_swiglu_workspace_capacity_bytes(QType::BF16,n,k,tokens,tokens):
        mode==1?ops::linear_add_workspace_capacity_bytes(QType::BF16,n,k,tokens,tokens):ops::linear_workspace_capacity_bytes(QType::BF16,n,k,ops::LinearPolicy::A16Only,tokens,tokens);
    WorkspaceArena workspace(std::max<std::size_t>(1,capacity));
    if(mode==2)ops::linear_swiglu(input,weight,out,ops::LinearPolicy::A16Only,workspace,nullptr,multiprocessor_count);
    else if(mode==1)ops::linear_add(input,weight,out,workspace,nullptr);else ops::linear(input,weight,out,nullptr);
    std::vector<__nv_bfloat16> actual(rows*tokens);output.copy_to_host(actual.data(),output.bytes);
    Error error;
    // Sample spatial positions, evaluating each selected complete dot product in FP64.
    for(int t=0;t<tokens;++t)for(int sample=0;sample<65;++sample){const int row=sample==64?rows-1:sample*(rows-1)/64;
        double sum=0,up=0;for(int c=0;c<k;++c){const double input_value=__bfloat162float(x[t*k+c]);
            sum+=input_value*__bfloat162float(w[row*k+c]);if(mode==2)up+=input_value*__bfloat162float(w[(rows+row)*k+c]);}
        double expected=mode==2?sum/(1+std::exp(-sum))*up:sum;
        if(mode==1)expected+=__bfloat162float(initial[t*n+row]);error.add(__bfloat162float(actual[t*rows+row]),expected);
    }
    return error.report("projection_"+std::to_string(n)+"x"+std::to_string(k)+"_T"+std::to_string(tokens)+"_mode"+std::to_string(mode)+"_SM"+std::to_string(multiprocessor_count),mode==2);
}
bool pool(int rows,int dimensions,bool normalized,bool zero,float scale=0.5f){
    constexpr int width=1024;auto input=values(rows*width,scale);if(zero)std::fill(input.begin(),input.end(),__float2bfloat16_rn(0));
    DeviceBuffer source(input.size()*2),output(rows*dimensions*4);source.copy_from_host(input.data(),source.bytes);
    ops::bf16_embedding_output(source.p,static_cast<float*>(output.p),rows,width,dimensions,normalized,nullptr);
    std::vector<float> actual(rows*dimensions);output.copy_to_host(actual.data(),output.bytes);
    std::vector<__nv_bfloat16> unchanged(input.size());source.copy_to_host(unchanged.data(),source.bytes);
    if(std::memcmp(unchanged.data(),input.data(),source.bytes)!=0)throw std::runtime_error("Pool modified input");
    Error error;for(int row=0;row<rows;++row){double square=0;for(int c=0;c<dimensions;++c){double value=__bfloat162float(input[row*width+c]);square+=value*value;}
        const double divisor=normalized?std::max(std::sqrt(square),1e-12):1;
        for(int c=0;c<dimensions;++c)error.add(actual[row*dimensions+c],__bfloat162float(input[row*width+c])/divisor);}
    return error.report("pool_B"+std::to_string(rows)+"_D"+std::to_string(dimensions)+"_norm"+std::to_string(normalized)+"_zero"+std::to_string(zero)+"_scale"+std::to_string(scale),false,true);
}
bool packed_attention(const std::vector<int>& lengths,int qh,int kh,int sm=0,int envelope=0){
    constexpr int d=128;int total=0;std::vector<int> begins;
    for(int length:lengths){begins.push_back(total);total+=length;}
    // Padding exercises the public input stride contract independently of the model layout.
    const int qs=qh*d+8,ks=kh*d+16;
    auto q=values(total*qs,.7f),k=values(total*ks,.7f),v=values(total*ks,.5f);
    DeviceBuffer qb(q.size()*2),kb(k.size()*2),vb(v.size()*2),out(total*qh*d*2);
    DeviceBuffer sb(begins.size()*4),lb(lengths.size()*4);
    qb.copy_from_host(q.data(),qb.bytes);kb.copy_from_host(k.data(),kb.bytes);vb.copy_from_host(v.data(),vb.bytes);
    sb.copy_from_host(begins.data(),sb.bytes);lb.copy_from_host(lengths.data(),lb.bytes);
    ops::packed_causal_bf16_attention(qb.p,kb.p,vb.p,out.p,lengths.size(),
        envelope?envelope:*std::max_element(lengths.begin(),lengths.end()),qh,kh,qs,ks,
        static_cast<int*>(sb.p),static_cast<int*>(lb.p),nullptr,sm);
    std::vector<__nv_bfloat16> actual(total*qh*d);out.copy_to_host(actual.data(),out.bytes);
    Error error;
    for(std::size_t s=0;s<lengths.size();++s){
      std::vector<int> rows;
      if(lengths[s]<=256){for(int row=0;row<lengths[s];++row)rows.push_back(row);}
      else{
        rows={0,1,31,32,63,64,127,128,511,512,1023,1024,1535,1536,
              lengths[s]/2,lengths[s]-2,lengths[s]-1};
        rows.erase(std::remove_if(rows.begin(),rows.end(),[&](int row){return row>=lengths[s];}),rows.end());
        std::sort(rows.begin(),rows.end());rows.erase(std::unique(rows.begin(),rows.end()),rows.end());
      }
      for(int row:rows)for(int head=0;head<qh;++head){
        const int query=begins[s]+row,kv_head=head/(qh/kh);
        std::vector<double> scores(row+1);double maximum=-INFINITY;
        for(int key=0;key<=row;++key){double dot=0;
            for(int c=0;c<d;++c)dot+=double(__bfloat162float(q[query*qs+head*d+c]))*
                __bfloat162float(k[(begins[s]+key)*ks+kv_head*d+c]);
            scores[key]=dot/std::sqrt(double(d));maximum=std::max(maximum,scores[key]);}
        double denominator=0;for(auto& score:scores){score=std::exp(score-maximum);denominator+=score;}
        for(int c:{0,7,31,63,95,127}){double expected=0;
            for(int key=0;key<=row;++key)expected+=scores[key]*__bfloat162float(v[(begins[s]+key)*ks+kv_head*d+c]);
            error.add(__bfloat162float(actual[(query*qh+head)*d+c]),expected/denominator);}
      }
    }
    // Check the read-only inputs and metadata after execution.
    for(auto pair:{std::pair{&qb,&q},std::pair{&kb,&k},std::pair{&vb,&v}}){
        std::vector<__nv_bfloat16> copy(pair.second->size());pair.first->copy_to_host(copy.data(),pair.first->bytes);
        if(std::memcmp(copy.data(),pair.second->data(),pair.first->bytes))throw std::runtime_error("Packed attention modified input");}
    std::vector<int> starts_copy(begins.size()),lengths_copy(lengths.size());
    sb.copy_to_host(starts_copy.data(),sb.bytes);lb.copy_to_host(lengths_copy.data(),lb.bytes);
    if(starts_copy!=begins||lengths_copy!=lengths)throw std::runtime_error("Packed attention modified metadata");
    return error.report("packed_attention_B"+std::to_string(lengths.size())+"_T"+std::to_string(total)+
                        "_Q"+std::to_string(qh)+"_KV"+std::to_string(kh)+
                        "_SM"+std::to_string(sm)+"_envelope"+std::to_string(envelope));
}
}
int main(){try{
    DeviceContext device;bool passed=true;int count=0;
    for(auto [n,k]:{std::pair{4096,1024},std::pair{1024,2048},std::pair{6144,1024},std::pair{1024,3072}})
        for(int t:{1,4,32,64,128,129,256}){passed&=projection(n,k,t,0);++count;if(n==1024){passed&=projection(n,k,t,1);++count;}}
    for(int t:{1,4,32,64,128,129,256}){passed&=projection(6144,1024,t,2);++count;}
    for(int t:{5,8,16,17,31,33,65,127,144,160,161,511,512,513,1023,1024,1025}){passed&=projection(6144,1024,t,2);++count;}
    // Independent FP64 math for both schedules, including the capacity boundary
    // and stream-only execution. SM facts select geometry without changing math.
    for(int sm:{0,70,80,81,156})for(int t:{129,160}){passed&=projection(6144,1024,t,2,sm);++count;}
    for(auto [qh,kh]:{std::pair{16,8},std::pair{16,16},std::pair{16,4},std::pair{8,8}}){
        for(const auto& lengths:{std::vector<int>{1},std::vector<int>{128,128,128,128},
             std::vector<int>{1,17,33,65,2,129,32,63},
             std::vector<int>{1535},std::vector<int>{1536},std::vector<int>{1537},
             std::vector<int>{2049},std::vector<int>{63,1535,1536,17,2049},
             std::vector<int>{1536,2048,1537}}){passed&=packed_attention(lengths,qh,kh,device.multiprocessor_count());++count;}
        // Exercise both resource routes and the eight-wave boundary at 2048.
        for(int sm:{0,70,127,128,156}){passed&=packed_attention({2048},qh,kh,sm);++count;}
        // The envelope may exceed device-resident sequence lengths, including
        // the final partially masked 32-column subgroup of a wider load.
        passed&=packed_attention({17,33},qh,kh,512,1536);++count;
        passed&=packed_attention({1,2049},qh,kh,512,2049);++count;
    }
    for(int rows:{1,8})for(int dimensions:{32,63,128,256,1024})for(bool norm:{false,true}){passed&=pool(rows,dimensions,norm,false);++count;}
    passed&=pool(8,1024,true,true);++count;
    for(float scale:{1e30f,1e-14f,1e-30f}){passed&=pool(8,1024,true,false,scale);++count;}
    std::cout<<"{\"checks\":"<<count<<",\"passed\":"<<(passed?"true":"false")<<"}\n";return passed?0:1;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
