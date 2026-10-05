#pragma once
#include "core/weight.h"
#include "core/tensor.h"
#include "core/arena.h"
#include "ninfer/ops/linear.h"
namespace ninfer::ops::detail {
bool xiaomi_bf16_shape(int n,int k);
void xiaomi_bf16_linear(const Tensor&,const Weight&,Tensor&,cudaStream_t);
void xiaomi_bf16_add(const Tensor&,const Weight&,Tensor&,WorkspaceArena&,cudaStream_t);
void xiaomi_bf16_swiglu(const Tensor&,const Weight&,Tensor&,WorkspaceArena&,cudaStream_t);
void xiaomi_bf16_split(const Tensor&,const Weight&,Tensor&,Tensor&,WorkspaceArena&,cudaStream_t);
void xiaomi_bf16_attn(const Tensor&,const Weight&,Tensor&,Tensor&,Tensor&,Tensor&,WorkspaceArena&,cudaStream_t);
void xiaomi_bf16_control(const Tensor&,const Weight&,const Tensor&,const Tensor&,Tensor&,Tensor&,cudaStream_t);
}
