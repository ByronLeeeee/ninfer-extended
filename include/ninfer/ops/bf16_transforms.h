#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer::ops {
// R(v) denotes round-to-nearest-even BF16. Unless specified otherwise all
// arrays are contiguous device BF16, counts are positive, integer arrays are
// device I32, and outputs do not alias inputs. No Op allocates device memory.
void float_to_bf16(const float* input, void* output, int elements, cudaStream_t stream);

// In-place R(GELU(R(x+bias))) using exact erf GELU, or R(x+bias) when disabled.
// Layout is [batch,channels,spatial]; spatial=1 also handles [tokens,width].
void rounded_bias_gelu(void* input_output, const void* bias, int elements,
    int channels, int spatial, bool gelu, cudaStream_t stream);
// R(GELU(R(FP32_input+bias))) or R(FP32_input+bias), [tokens,channels].
void float_bias_cast(const float* input, const void* bias, void* output,
    int elements, int channels, bool gelu, cudaStream_t stream);

// Exact transpose [chunks,channels,frequency,steps] -> [chunks,steps,channels*frequency].
void conv_to_tokens(const void* input, void* output, int chunks, int channels,
    int frequency, int steps, cudaStream_t stream);
// Exact in-place zeroing of NCHW columns outside ceil(physical_width/stride).
// widths[chunk] is the unpadded source convolution width; valid values remain
// bit-identical. This preserves sub-second convolution boundary semantics.
void zero_conv_padding(void* input_output, const std::int32_t* widths, int chunks,
    int channels, int frequency, int steps, int stride, cudaStream_t stream);
// In-place R(x + positions[step,width]), repeated independently per chunk.
void add_chunk_positions(void* input_output, const void* positions, int chunks,
    int steps, int width, cudaStream_t stream);
// Exact row gather; indices must reference available input rows.
void gather_rows(const void* input, const std::int32_t* indices, void* output,
    int rows, int width, cudaStream_t stream);
// Exact selection from embeddings[ids[t]] when audio_rows[t]<0, otherwise
// audio[audio_rows[t]]. The caller validates both sets of indices.
void embed_audio_tokens(const void* embeddings, const void* audio,
    const std::int32_t* ids, const std::int32_t* audio_rows, void* output,
    int tokens, int width, cudaStream_t stream);
// Exact split of token-major [Q_width,K_width,K_width] into separate Q/K/V.
void split_qkv(const void* input, void* q, void* k, void* v, int tokens,
    int query_width, int kv_width, cudaStream_t stream);

// Q/K: R(R(x/sqrt(mean(x^2)+eps))*gamma), then split-half RoPE with
// BF16 cosine/sine and separate BF16 products. V is copied exactly.
// D=128; gamma has D elements, inv_freq has D/2 FP32 elements.
void qk_norm_rope(const void* packed, const void* query_gamma, const void* key_gamma,
    const float* inv_freq, const std::int32_t* positions, void* q, void* k, void* v,
    int tokens, int head_dim, int query_heads, int kv_heads, float eps, cudaStream_t stream);
// Exact cache writes. Prefill stores row t at positions[t] in [capacity,width].
// Decode stores lane t at [t,positions[t]] in [lanes,capacity,width]. Positions
// must be within capacity; all other cache rows are preserved.
void append_contiguous_kv(const void* k, const void* v, void* cache_k, void* cache_v,
    int tokens, int width, int capacity, const std::int32_t* positions,
    bool decode, cudaStream_t stream);
void advance_positions(std::int32_t* positions, int lanes, cudaStream_t stream);
void rounded_rmsnorm(const void* input, const void* gamma, void* output,
    int rows, int width, float eps, cudaStream_t stream);
void rounded_rope(void* input_output, const float* inv_freq,
    const std::int32_t* positions, int tokens, int heads, int head_dim, cudaStream_t stream);
// Input [tokens,2*width] contains gate then up; output is R(R(SiLU(gate))*up).
void rounded_swiglu(const void* input, void* output, int tokens, int width, cudaStream_t stream);
// In-place R(residual+delta), with no change to delta.
void bf16_residual_add(void* residual, const void* delta, int elements, cudaStream_t stream);
// Exact maximum with ties resolved to the smallest index, for finite logits.
// Scratch values (FP32) and indices (I32) each hold rows*ceil(width/1024).
void bf16_argmax(const void* logits, float* scratch_values, std::int32_t* scratch_indices,
    std::int32_t* output_indices, int rows, int width, cudaStream_t stream);
// Same exact argmax over width valid columns in rows with row_stride columns.
// Padding columns never participate. Scratch holds rows*ceil(width/1024).
void bf16_argmax_valid(const void* logits, float* scratch_values, std::int32_t* scratch_indices,
    std::int32_t* output_indices, int rows, int width, int row_stride, cudaStream_t stream);
}
