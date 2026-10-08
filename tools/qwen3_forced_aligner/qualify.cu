#include "ninfer/ops/bf16_transforms.h"
#include "core/device.h"
#include "core/arena.h"
#include <cuda_bf16.h>
#include <iostream>
#include <vector>

int main(){try{
    ninfer::DeviceContext device;
    for(int width:{1,1023,1024,1025,5000})for(int rows:{1,17,64}){
        int stride=width==5000?5120:width+128,blocks=(width+1023)/1024;
        std::vector<__nv_bfloat16> input(static_cast<std::size_t>(rows)*stride);
        std::vector<int> expected(rows),actual(rows);
        for(int row=0;row<rows;++row){int first=width==1?0:(row*83)%(width-1);expected[row]=first;
            for(int col=0;col<stride;++col)input[static_cast<std::size_t>(row)*stride+col]=__float2bfloat16(col>=width?1000.f:-1.f);
            input[static_cast<std::size_t>(row)*stride+first]=__float2bfloat16(5.f);
            input[static_cast<std::size_t>(row)*stride+width-1]=__float2bfloat16(5.f);
        }
        ninfer::DeviceBuffer x(input.size()*2),values(rows*blocks*4),indices(rows*blocks*4),out(rows*4);
        x.copy_from_host(input.data(),x.bytes);
        ninfer::ops::bf16_argmax_valid(x.p,static_cast<float*>(values.p),static_cast<int*>(indices.p),static_cast<int*>(out.p),rows,width,stride,device.stream);
        device.synchronize();out.copy_to_host(actual.data(),out.bytes);
        if(actual!=expected)throw std::runtime_error("Timestamp class argmax mismatch");
    }
    for(int stride:{2,4,8}){
        int steps=(100+stride-1)/stride,channels=2,frequency=3;
        std::vector<int> widths{1,2,74,99,100};
        std::vector<__nv_bfloat16> input(widths.size()*channels*frequency*steps),actual(input.size());
        for(std::size_t i=0;i<input.size();++i)input[i]=__float2bfloat16(static_cast<float>(i%37+1));
        ninfer::DeviceBuffer x(input.size()*2),w(widths.size()*4);x.copy_from_host(input.data(),x.bytes);w.copy_from_host(widths.data(),w.bytes);
        ninfer::ops::zero_conv_padding(x.p,static_cast<int*>(w.p),widths.size(),channels,frequency,steps,stride,device.stream);
        device.synchronize();x.copy_to_host(actual.data(),x.bytes);
        for(std::size_t i=0;i<input.size();++i){int chunk=i/(channels*frequency*steps),step=i%steps;
            float expected=step>=(widths[chunk]+stride-1)/stride?0.f:__bfloat162float(input[i]);
            if(__bfloat162float(actual[i])!=expected)throw std::runtime_error("Convolution padding transform mismatch");}
    }
    std::cout<<"OK exact timestamp argmax and short-convolution padding: valid columns, ties, reduction blocks, preserved valid values\n";
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
