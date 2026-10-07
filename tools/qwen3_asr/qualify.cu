#include "ninfer/ops/dense_attention.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/layer_norm.h"
#include "ninfer/ops/bf16_transforms.h"
#include "core/device.h"
#include "core/arena.h"
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace ninfer;using BF=__nv_bfloat16;
namespace {
std::mt19937 rng(1705070);std::normal_distribution<float> distribution(0,.35f);
double f(BF x){return __bfloat162float(x);}BF b(double x){return __float2bfloat16_rn(static_cast<float>(x));}double round_bf(double x){return f(b(x));}
std::vector<BF> random(std::size_t count){std::vector<BF> x(count);for(auto& value:x)value=b(distribution(rng));return x;}
template<class T> DeviceBuffer upload(const std::vector<T>& x){DeviceBuffer y(x.size()*sizeof(T));y.copy_from_host(x.data(),y.bytes);return y;}
std::vector<BF> read(const DeviceBuffer& x){std::vector<BF> y(x.bytes/2);x.copy_to_host(y.data(),x.bytes);return y;}
struct Error {double error=0,reference=0,maximum=0,refmax=0;std::size_t count=0;bool finite=true;
    void add(double actual,double expected){double d=actual-expected;error+=d*d;reference+=expected*expected;maximum=std::max(maximum,std::abs(d));refmax=std::max(refmax,std::abs(expected));++count;finite&=std::isfinite(actual)&&std::isfinite(expected);}
    bool report(const std::string& name,double threshold=.006){double nrms=std::sqrt(error/std::max(reference,1e-30));double peak=maximum/std::sqrt(std::max(reference,1e-30)/count);bool pass=finite&&nrms<=threshold&&peak<=.06;
        std::cout<<"{\"case\":\""<<name<<"\",\"oracle\":\"independent_fp64\",\"samples\":"<<count<<",\"nrms\":"<<nrms<<",\"peak_over_rms\":"<<peak<<",\"pass\":"<<(pass?"true":"false")<<"}\n"<<std::flush;return pass;}
    bool activation_report(const std::string& name){
        // Existing LinearSwiGLU A16 criterion, from tests/ops/linear_swiglu/
        // linear_swiglu_test_common.cpp; private staging does not change it.
        constexpr double relative_l2=3.3e-3,gross_absolute=5e-3,gross_relative=6.3e-3;
        double nrms=std::sqrt(error/std::max(reference,1e-30));
        double gross_limit=gross_absolute+gross_relative*refmax;
        bool pass=finite&&nrms<=relative_l2&&maximum<=gross_limit;
        std::cout<<"{\"case\":\""<<name<<"\",\"oracle\":\"independent_fp64\",\"criterion\":\"LinearSwiGLU_A16\",\"samples\":"<<count
                 <<",\"nrms\":"<<nrms<<",\"nrms_limit\":"<<relative_l2<<",\"maximum_error\":"<<maximum
                 <<",\"gross_limit\":"<<gross_limit<<",\"pass\":"<<(pass?"true":"false")<<"}\n"<<std::flush;
        return pass;
    }
};
std::vector<int> queries(int count){if(count<=17){std::vector<int> x(count);std::iota(x.begin(),x.end(),0);return x;}return {0,1,7,12,count/2,count-3,count-2,count-1};}
bool attention(DeviceContext& device,int d,int qh,int kh,int count,bool causal,bool flash,bool tensorcore=false,bool ranged=false){
    int qw=qh*d,kw=kh*d,stride=qw+2*kw;auto x=random(count*stride);auto xb=upload(x);DeviceBuffer out(count*qw*2);std::vector<int> starts(count),ends(count),pos(count),cu{0};
    int window=causal?count:104;for(int start=0;start<count;start+=window){int stop=std::min(count,start+window);cu.push_back(stop);for(int i=start;i<stop;++i){starts[i]=start;ends[i]=stop;pos[i]=i;}}
    if(ranged)for(int i=0;i<count;++i){starts[i]=std::max(0,i-(i%47+1));ends[i]=std::min(count,i+7);}
    auto sb=upload(starts),eb=upload(ends),pb=upload(pos),cb=upload(cu);auto* base=static_cast<BF*>(xb.p);
    if(tensorcore){ops::causal_bf16_attention(base,base+qw,base+qw+kw,out.p,count,qh,kh,stride,stride,device.stream);device.synchronize();}
    else if(flash){Tensor q(base,DType::BF16,{d,qh,count}),k(base+qw,DType::BF16,{d,kh,count}),v(base+qw+kw,DType::BF16,{d,kh,count}),o(out.p,DType::BF16,{d,qh,count});q.nb[2]=k.nb[2]=v.nb[2]=stride*2;
        Tensor c(cb.p,DType::I32,{static_cast<int>(cu.size())});WorkspaceArena ws(std::max<std::size_t>(256,ops::packed_softmax_attention_workspace_capacity_bytes({d,qh,kh},count,count,cu.size()-1,cu.size()-1)));ops::packed_softmax_attention(q,k,v,{d,qh,kh},1/std::sqrt(float(d)),c,ws,o,device.stream);device.synchronize();}
    else{ops::dense_bf16_attention(base,base+qw,base+qw+kw,out.p,count,count,d,qh,kh,stride,stride,static_cast<int*>(sb.p),static_cast<int*>(eb.p),static_cast<int*>(pb.p),causal,device.stream);device.synchronize();}
    auto actual=read(out);Error error;
    for(int i:queries(count))for(int h=0;h<qh;++h){int stop=causal?std::min(ends[i],i+1):ends[i];std::vector<double> scores;double maximum=-INFINITY;
        for(int j=starts[i];j<stop;++j){double score=0;for(int z=0;z<d;++z)score+=f(x[i*stride+h*d+z])*f(x[j*stride+qw+(h/(qh/kh))*d+z]);score/=std::sqrt(double(d));scores.push_back(score);maximum=std::max(maximum,score);}
        double sum=0;for(auto& score:scores){score=std::exp(score-maximum);sum+=score;}
        for(int z=0;z<d;++z){double result=0;for(int j=starts[i];j<stop;++j)result+=scores[j-starts[i]]*f(x[j*stride+qw+kw+(h/(qh/kh))*d+z]);error.add(f(actual[i*qw+h*d+z]),result/sum);}}
    return error.report(std::string(tensorcore?"causal_tensorcore":flash?"flash":"dense")+"_attention_D"+std::to_string(d)+"_Q"+std::to_string(qh)+"_KV"+std::to_string(kh)+"_T"+std::to_string(count)+(ranged?"_ranges":""));
}
bool decode(DeviceContext& device,int batch,int cap){int d=128,qh=16,kh=8,qw=qh*d,kw=kh*d;auto q=random(batch*qw),k=random(static_cast<std::size_t>(batch)*cap*kw),v=random(k.size());
    std::vector<int> positions(batch);for(int i=0;i<batch;++i)positions[i]=i==0?cap-1:std::min(cap-1,i==1?0:i==2?37:211);auto qb=upload(q),kb=upload(k),vb=upload(v),pb=upload(positions);DeviceBuffer out(batch*qw*2);
    WorkspaceArena ws(std::max<std::size_t>(256,ops::dense_bf16_decode_attention_workspace_capacity(batch,cap,d,qh,kh)));
    ops::dense_bf16_decode_attention(qb.p,kb.p,vb.p,out.p,batch,cap,d,qh,kh,static_cast<int*>(pb.p),ws,device.stream);device.synchronize();auto actual=read(out);Error error;
    for(int i=0;i<batch;++i)for(int h=0;h<qh;++h){std::vector<double> scores(positions[i]+1);double maximum=-INFINITY;
        for(int j=0;j<=positions[i];++j){double score=0;for(int z=0;z<d;++z)score+=f(q[i*qw+h*d+z])*f(k[(static_cast<std::size_t>(i)*cap+j)*kw+(h/2)*d+z]);score/=std::sqrt(double(d));scores[j]=score;maximum=std::max(maximum,score);}
        double sum=0;for(auto& score:scores){score=std::exp(score-maximum);sum+=score;}for(int z=0;z<d;++z){double result=0;for(int j=0;j<=positions[i];++j)result+=scores[j]*f(v[(static_cast<std::size_t>(i)*cap+j)*kw+(h/2)*d+z]);error.add(f(actual[i*qw+h*d+z]),result/sum);}}
    return error.report("decode_attention_B"+std::to_string(batch)+"_C"+std::to_string(cap));
}
bool norm_rope(DeviceContext& device){int t=17,d=128,qh=16,kh=8,qw=d*qh,kw=d*kh,stride=qw+kw*2;auto x=random(t*stride),qweight=random(d),kweight=random(d);std::vector<float> inv(d/2);std::vector<int> pos(t);
    for(int i=0;i<d/2;++i)inv[i]=1/std::pow(1000000.f,float(2*i)/d);for(int i=0;i<t;++i)pos[i]=i==t-1?65535:i*37;
    auto xb=upload(x),qwb=upload(qweight),kwb=upload(kweight),ib=upload(inv),pb=upload(pos);DeviceBuffer qb(t*qw*2),kb(t*kw*2),vb(t*kw*2);
    ops::qk_norm_rope(xb.p,qwb.p,kwb.p,static_cast<float*>(ib.p),static_cast<int*>(pb.p),qb.p,kb.p,vb.p,t,d,qh,kh,1e-6f,device.stream);device.synchronize();auto aq=read(qb),ak=read(kb),av=read(vb);Error error;
    for(int i=0;i<t;++i)for(int h=0;h<qh+kh;++h){bool isq=h<qh;int hh=isq?h:h-qh;const auto& weight=isq?qweight:kweight;int base=i*stride+(isq?0:qw)+hh*d;double sum=0;for(int z=0;z<d;++z)sum+=f(x[base+z])*f(x[base+z]);double rs=1/std::sqrt(sum/d+1e-6);
        for(int z=0;z<d/2;++z){float angle=float(pos[i])*inv[z];double c=round_bf(std::cos(double(angle))),s=round_bf(std::sin(double(angle)));double a=round_bf(round_bf(f(x[base+z])*rs)*f(weight[z])),b0=round_bf(round_bf(f(x[base+z+d/2])*rs)*f(weight[z+d/2]));
            const auto& out=isq?aq:ak;int index=i*(isq?qw:kw)+hh*d+z;error.add(f(out[index]),round_bf(round_bf(a*c)-round_bf(b0*s)));error.add(f(out[index+d/2]),round_bf(round_bf(b0*c)+round_bf(a*s)));}
        if(!isq)for(int z=0;z<d;++z)error.add(f(av[i*kw+hh*d+z]),f(x[i*stride+qw+kw+hh*d+z]));}
    return error.report("fused_qk_rmsnorm_rope",.001);
}
bool rmsnorm(DeviceContext& device,int width){int rows=17;auto x=random(rows*width),w=random(width);auto xb=upload(x),wb=upload(w);DeviceBuffer out(x.size()*2);ops::rounded_rmsnorm(xb.p,wb.p,out.p,rows,width,1e-6f,device.stream);device.synchronize();auto actual=read(out);Error error;
    for(int i=0;i<rows;++i){double sum=0;for(int z=0;z<width;++z)sum+=f(x[i*width+z])*f(x[i*width+z]);double r=1/std::sqrt(sum/width+1e-6);for(int z=0;z<width;++z)error.add(f(actual[i*width+z]),round_bf(round_bf(f(x[i*width+z])*r)*f(w[z])));}
    return error.report("rounded_rmsnorm_W"+std::to_string(width),.001);
}
bool linear(DeviceContext& device,int n,int k){auto weights=random(static_cast<std::size_t>(n)*k);auto wb=upload(weights);Weight w;w.qtype=QType::BF16;w.layout=QuantLayout::Contiguous;w.qdata=wb.p;w.n=n;w.k=k;bool pass=true;
    std::vector<int> extents{1,2,4,13,104};if(k==2048&&n!=151936)extents.insert(extents.end(),{64,65,128,129,191,192,193,211,256,257,407,512,513,805});
    if(n==2048&&k==6144)extents.insert(extents.end(),{256,257,407,512,513,805});
    for(int t:extents){auto x=random(t*k);auto xb=upload(x);DeviceBuffer ob(static_cast<std::size_t>(n)*t*2);Tensor input(xb.p,DType::BF16,{k,t}),out(ob.p,DType::BF16,{n,t});ops::linear(input,w,out,device.stream);device.synchronize();auto actual=read(ob);Error error;
        std::vector<int> rows{0,1,7,8,31,32,63,64,n/2-1,n/2,n-2,n-1};for(int i:queries(t))for(int row:rows){double sum=0;for(int z=0;z<k;++z)sum+=f(x[i*k+z])*f(weights[static_cast<std::size_t>(row)*k+z]);error.add(f(actual[static_cast<std::size_t>(i)*n+row]),sum);}
        pass&=error.report("linear_N"+std::to_string(n)+"_K"+std::to_string(k)+"_T"+std::to_string(t));}
    return pass;
}
bool swiglu(DeviceContext& device){int n=12288,k=2048,width=n/2;auto weights=random(static_cast<std::size_t>(n)*k);auto wb=upload(weights);Weight w;w.qtype=QType::BF16;w.layout=QuantLayout::Contiguous;w.qdata=wb.p;w.n=n;w.k=k;WorkspaceArena ws(std::max<std::size_t>(256,ops::linear_swiglu_workspace_capacity_bytes(QType::BF16,n,k,ops::LinearPolicy::A16Only,1,805)));bool pass=true;
    std::vector<double> decoded_weights(weights.size());
    std::transform(weights.begin(),weights.end(),decoded_weights.begin(),f);
    for(int t:{1,2,4,5,13,64,65,104,128,129,191,192,193,211,256,257,805}){auto x=random(t*k);auto xb=upload(x);DeviceBuffer ob(static_cast<std::size_t>(width)*t*2);Tensor input(xb.p,DType::BF16,{k,t}),out(ob.p,DType::BF16,{width,t});ops::linear_swiglu(input,w,out,ops::LinearPolicy::A16Only,ws,device.stream);device.synchronize();auto actual=read(ob);
        Error ideal_error;
        // The A16 criterion uses whole-tensor L2 and the whole-tensor maximum.
        // Evaluate all tokens and channels so sparse token probes cannot alter
        // either its normalization or its gross-error bound. The oracle remains
        // naive FP64 dot products; CPU workers only divide independent rows.
        std::vector<double> decoded_x(x.size()),ideal(static_cast<std::size_t>(width)*t);
        std::transform(x.begin(),x.end(),decoded_x.begin(),f);
        const int workers=std::min(16,std::max(1,static_cast<int>(std::thread::hardware_concurrency())));
        std::vector<std::thread> threads;
        for(int worker=0;worker<workers;++worker)threads.emplace_back([&,worker]{
            const int begin=width*worker/workers,end=width*(worker+1)/workers;
            for(int row=begin;row<end;++row){const double* gw=decoded_weights.data()+static_cast<std::size_t>(row)*k;
                const double* uw=decoded_weights.data()+static_cast<std::size_t>(row+width)*k;
                for(int i=0;i<t;++i){const double* xi=decoded_x.data()+static_cast<std::size_t>(i)*k;double gate=0,up=0;
                    for(int z=0;z<k;++z){gate+=xi[z]*gw[z];up+=xi[z]*uw[z];}
                    const auto index=static_cast<std::size_t>(i)*width+row;
                    ideal[index]=gate/(1+std::exp(-gate))*up;
                }
            }
        });
        for(auto& thread:threads)thread.join();
        for(std::size_t index=0;index<ideal.size();++index){
            ideal_error.add(f(actual[index]),ideal[index]);
        }
        pass&=ideal_error.activation_report("linear_swiglu_ideal_T"+std::to_string(t));}
    return pass;
}
bool exact(const std::string& name,const std::vector<BF>& actual,const std::vector<BF>& expected){
    bool pass=actual.size()==expected.size();std::size_t errors=0;
    if(pass)for(std::size_t i=0;i<actual.size();++i)errors+=f(actual[i])!=f(expected[i]);
    pass&=errors==0;
    std::cout<<"{\"case\":\""<<name<<"\",\"oracle\":\"independent_exact\",\"elements\":"<<actual.size()<<",\"differences\":"<<errors<<",\"pass\":"<<(pass?"true":"false")<<"}\n"<<std::flush;
    return pass;
}
bool transforms(DeviceContext& device){bool pass=true;
    {
        auto source=random(8193);std::vector<float> values(source.size());
        for(std::size_t i=0;i<values.size();++i)values[i]=float(f(source[i]))+.000013f;
        auto input=upload(values);DeviceBuffer output(values.size()*2);std::vector<BF> expected(values.size());
        for(std::size_t i=0;i<values.size();++i)expected[i]=b(values[i]);
        ops::float_to_bf16(static_cast<float*>(input.p),output.p,values.size(),device.stream);device.synchronize();
        pass&=exact("float_to_bf16_tail",read(output),expected);
    }
    {
        int channels=32,spatial=13,rows=2,n=rows*channels*spatial;
        auto source=random(n),bias=random(channels);
        for(bool activation:{false,true}){auto input=upload(source),bb=upload(bias);
            ops::rounded_bias_gelu(input.p,bb.p,n,channels,spatial,activation,device.stream);device.synchronize();auto actual=read(input);Error error;
            for(int i=0;i<n;++i){double value=round_bf(f(source[i])+f(bias[(i/spatial)%channels]));
                if(activation)value=.5*value*(1+std::erf(value/std::sqrt(2.)));
                error.add(f(actual[i]),round_bf(value));}
            pass&=error.report(activation?"rounded_bias_exact_gelu":"rounded_bias",.001);
        }
        std::vector<float> fp(n);for(int i=0;i<n;++i)fp[i]=float(f(source[i]))+.00017f;
        for(bool activation:{false,true}){auto input=upload(fp),bb=upload(bias);DeviceBuffer output(n*2);
            ops::float_bias_cast(static_cast<float*>(input.p),bb.p,output.p,n,channels,activation,device.stream);device.synchronize();auto actual=read(output);Error error;
            for(int i=0;i<n;++i){double value=round_bf(double(fp[i])+f(bias[i%channels]));
                if(activation)value=.5*value*(1+std::erf(value/std::sqrt(2.)));
                error.add(f(actual[i]),round_bf(value));}
            pass&=error.report(activation?"float_bias_exact_gelu":"float_bias_cast",.001);
        }
    }
    {
        int chunks=2,channels=4,freq=3,steps=13,n=chunks*channels*freq*steps;
        auto source=random(n);auto input=upload(source);DeviceBuffer output(n*2);std::vector<BF> expected(n);
        for(int c=0;c<chunks;++c)for(int t=0;t<steps;++t)for(int h=0;h<channels;++h)for(int d=0;d<freq;++d)
            expected[((c*steps+t)*channels+h)*freq+d]=source[((c*channels+h)*freq+d)*steps+t];
        ops::conv_to_tokens(input.p,output.p,chunks,channels,freq,steps,device.stream);device.synchronize();
        pass&=exact("conv_to_tokens",read(output),expected);
        auto position=random(steps*channels*freq);auto pb=upload(position);
        for(int i=0;i<n;++i)expected[i]=b(f(expected[i])+f(position[i%position.size()]));
        ops::add_chunk_positions(output.p,pb.p,chunks,steps,channels*freq,device.stream);device.synchronize();
        pass&=exact("add_chunk_positions",read(output),expected);
        std::vector<int> indices{25,0,12,12,13};auto ib=upload(indices);DeviceBuffer gathered(indices.size()*channels*freq*2);
        std::vector<BF> gather_expected(indices.size()*channels*freq);
        for(std::size_t i=0;i<gather_expected.size();++i)gather_expected[i]=expected[indices[i/(channels*freq)]*channels*freq+i%(channels*freq)];
        ops::gather_rows(output.p,static_cast<int*>(ib.p),gathered.p,indices.size(),channels*freq,device.stream);device.synchronize();
        pass&=exact("gather_rows_duplicates_and_tail",read(gathered),gather_expected);
    }
    {
        int width=128,tokens=4;auto embeddings=random(32*width),audio=random(7*width);
        std::vector<int> ids{0,31,7,9},rows{-1,-1,6,0};auto eb=upload(embeddings),ab=upload(audio),ib=upload(ids),rb=upload(rows);DeviceBuffer output(tokens*width*2);
        std::vector<BF> expected(tokens*width);
        for(int t=0;t<tokens;++t)for(int d=0;d<width;++d)expected[t*width+d]=rows[t]<0?embeddings[ids[t]*width+d]:audio[rows[t]*width+d];
        ops::embed_audio_tokens(eb.p,ab.p,static_cast<int*>(ib.p),static_cast<int*>(rb.p),output.p,tokens,width,device.stream);device.synchronize();
        pass&=exact("embed_audio_tokens",read(output),expected);
    }
    {
        int t=17,qw=32,kw=16,stride=qw+2*kw;auto packed=random(t*stride);auto input=upload(packed);DeviceBuffer q(t*qw*2),k(t*kw*2),v(t*kw*2);
        std::vector<BF> eq(t*qw),ek(t*kw),ev(t*kw);
        for(int i=0;i<t;++i){std::copy_n(packed.begin()+i*stride,qw,eq.begin()+i*qw);std::copy_n(packed.begin()+i*stride+qw,kw,ek.begin()+i*kw);std::copy_n(packed.begin()+i*stride+qw+kw,kw,ev.begin()+i*kw);}
        ops::split_qkv(input.p,q.p,k.p,v.p,t,qw,kw,device.stream);device.synchronize();
        pass&=exact("split_q",read(q),eq);pass&=exact("split_k",read(k),ek);pass&=exact("split_v",read(v),ev);
        for(bool decoding:{false,true}){int cap=23,lanes=decoding?t:1;std::vector<int> positions(t);for(int i=0;i<t;++i)positions[i]=decoding?(i*7)%cap:i;
            auto pb=upload(positions);std::vector<BF> eck(lanes*cap*kw,b(-.125)),ecv=eck;auto ck=upload(eck),cv=upload(ecv);
            for(int i=0;i<t;++i)for(int d=0;d<kw;++d){int dst=((decoding?i*cap:0)+positions[i])*kw+d;eck[dst]=ek[i*kw+d];ecv[dst]=ev[i*kw+d];}
            ops::append_contiguous_kv(k.p,v.p,ck.p,cv.p,t,kw,cap,static_cast<int*>(pb.p),decoding,device.stream);device.synchronize();
            pass&=exact(decoding?"kv_decode_k_preserves_other_rows":"kv_prefill_k_preserves_other_rows",read(ck),eck);
            pass&=exact(decoding?"kv_decode_v_preserves_other_rows":"kv_prefill_v_preserves_other_rows",read(cv),ecv);
            ops::advance_positions(static_cast<int*>(pb.p),t,device.stream);device.synchronize();std::vector<int> actual(t);pb.copy_to_host(actual.data(),pb.bytes);
            for(auto& p:positions)++p;bool ok=actual==positions;pass&=ok;
            std::cout<<"{\"case\":\"advance_positions_"<<(decoding?"decode":"prefill")<<"\",\"oracle\":\"independent_exact\",\"pass\":"<<(ok?"true":"false")<<"}\n";
        }
    }
    {
        auto x=random(4097),delta=random(x.size());auto xb=upload(x),db=upload(delta);std::vector<BF> expected(x.size());
        for(std::size_t i=0;i<x.size();++i)expected[i]=b(f(x[i])+f(delta[i]));
        ops::bf16_residual_add(xb.p,db.p,x.size(),device.stream);device.synchronize();
        pass&=exact("bf16_residual_add",read(xb),expected);pass&=exact("residual_input_unchanged",read(db),delta);
    }
    for(int lanes:{33,257}){
        std::vector<int> positions(lanes);std::iota(positions.begin(),positions.end(),0);auto input=upload(positions);
        ops::advance_positions(static_cast<int*>(input.p),lanes,device.stream);device.synchronize();std::vector<int> actual(lanes);input.copy_to_host(actual.data(),input.bytes);
        for(auto& p:positions)++p;bool ok=actual==positions;pass&=ok;
        std::cout<<"{\"case\":\"advance_positions_B"<<lanes<<"\",\"oracle\":\"independent_exact\",\"pass\":"<<(ok?"true":"false")<<"}\n";
    }
    for(int width:{1,1023,1024,1025,151936}){
        int rows=4,blocks=(width+1023)/1024;auto values=random(rows*width);std::vector<int> expected(rows);
        for(int r=0;r<rows;++r){int first=std::min(width-1,31),second=width-1;values[r*width+first]=b(7);values[r*width+second]=b(7);expected[r]=first;}
        auto input=upload(values);DeviceBuffer scratch(rows*blocks*4),indices(rows*blocks*4),output(rows*4);
        ops::bf16_argmax(input.p,static_cast<float*>(scratch.p),static_cast<int*>(indices.p),static_cast<int*>(output.p),rows,width,device.stream);device.synchronize();
        std::vector<int> actual(rows);output.copy_to_host(actual.data(),output.bytes);bool ok=actual==expected;pass&=ok;
        std::cout<<"{\"case\":\"argmax_ties_W"<<width<<"\",\"oracle\":\"independent_exact\",\"pass\":"<<(ok?"true":"false")<<"}\n";
    }
    {
        int width=1024,rows=17;auto x=random(rows*width),gamma=random(width),beta=random(width);auto xb=upload(x),gb=upload(gamma),bb=upload(beta);DeviceBuffer output(rows*width*2);
        Tensor tx(xb.p,DType::BF16,{width,rows}),tg(gb.p,DType::BF16,{width}),tb(bb.p,DType::BF16,{width}),to(output.p,DType::BF16,{width,rows});
        ops::layer_norm(tx,tg,tb,1e-5f,to,device.stream);device.synchronize();auto actual=read(output);Error error;
        for(int r=0;r<rows;++r){double mean=0,variance=0;for(int d=0;d<width;++d)mean+=f(x[r*width+d]);mean/=width;
            for(int d=0;d<width;++d){double z=f(x[r*width+d])-mean;variance+=z*z;}double inv=1/std::sqrt(variance/width+1e-5);
            for(int d=0;d<width;++d)error.add(f(actual[r*width+d]),(f(x[r*width+d])-mean)*inv*f(gamma[d])+f(beta[d]));}
        pass&=error.report("layer_norm_W1024");
    }
    return pass;
}
bool linear_add(DeviceContext& device,int k){int n=2048;auto weights=random(static_cast<std::size_t>(n)*k);auto wb=upload(weights);Weight w;w.qtype=QType::BF16;w.layout=QuantLayout::Contiguous;w.qdata=wb.p;w.n=n;w.k=k;WorkspaceArena ws(64*1024*1024);bool pass=true;
    for(int t:{1,2,4,65,128,129,211,256,257,407,512,513,805}){auto x=random(t*k),residual=random(t*n);auto xb=upload(x),ob=upload(residual);Tensor input(xb.p,DType::BF16,{k,t}),out(ob.p,DType::BF16,{n,t});ops::linear_add(input,w,out,ops::LinearPolicy::A16Only,ws,device.stream);device.synchronize();auto actual=read(ob);Error error;
        for(int i:queries(t))for(int row:{0,1,31,32,63,64,n/2,n-1}){double projection=0;for(int z=0;z<k;++z)projection+=f(x[i*k+z])*f(weights[static_cast<std::size_t>(row)*k+z]);error.add(f(actual[static_cast<std::size_t>(i)*n+row]),projection+f(residual[static_cast<std::size_t>(i)*n+row]));}
        pass&=error.report("linear_add_K"+std::to_string(k)+"_T"+std::to_string(t));}
    return pass;
}
}
int main(int argc,char** argv){try{DeviceContext device;bool pass=true;bool projections=argc>1&&std::string(argv[1])=="--linear";
    if(argc>1&&std::string(argv[1])=="--transforms")pass&=transforms(device);
    else if(projections){for(auto shape:std::vector<std::pair<int,int>>{{2048,2048},{4096,2048},{12288,2048},{2048,6144},{151936,2048},{1024,1024},{3072,1024},{4096,1024},{1024,4096},{1024,7680},{2048,1024}})pass&=linear(device,shape.first,shape.second);pass&=swiglu(device);for(int k:{2048,6144})pass&=linear_add(device,k);}
    else{
        for(int t:{1,13,104,197,211}){pass&=attention(device,64,16,16,t,false,false);pass&=attention(device,64,16,16,t,false,true);}
        for(int t:{197,211})pass&=attention(device,64,12,12,t,false,true);
        for(int t:{1,13,211,805})pass&=attention(device,128,16,8,t,true,false);
        for(int t:{192,193,194,195,196,256,257,407})pass&=attention(device,128,16,8,t,true,false);
        for(auto heads:std::vector<std::pair<int,int>>{{8,8},{16,16},{16,4}})
            for(int t:{193,211,805})pass&=attention(device,128,heads.first,heads.second,t,true,false);
        for(int t:{193,211,407,805})pass&=attention(device,128,16,8,t,true,false,false,true);
        for(int t:{1,13,31,32,33,63,64,65,70,104,128,129,197,211,256,257,407,805,1024})
            pass&=attention(device,128,16,8,t,true,false,true);
        for(auto heads:std::vector<std::pair<int,int>>{{8,8},{16,16},{16,4}})
            for(int t:{1,13,65,211})pass&=attention(device,128,heads.first,heads.second,t,true,false,true);
        for(int b0:{1,2,3,4})for(int cap:{1,37,127,128,211,2048,4096})pass&=decode(device,b0,cap);
        pass&=norm_rope(device);for(int width:{128,1024,2048})pass&=rmsnorm(device,width);
    }
    return pass?0:1;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}}
