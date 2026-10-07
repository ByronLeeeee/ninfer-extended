#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include "core/arena.h"

namespace ninfer::ops {

// Causal attention over one complete token-major sequence. Query i attends
// keys [0,i]. D=128; qualified (Q_heads,KV_heads) are (16,8), (16,16),
// (16,4), and (8,8). Element strides may include padding or interleaved Q/K/V;
// they must be multiples of eight and cover their logical head widths.
// Inputs are 16-byte-aligned represented BF16; output is contiguous BF16
// [tokens,Q_heads*128] and does not alias inputs. FP32 online softmax and MMA
// accumulation use a BF16 probability fragment plus its BF16 residual to retain
// probability precision. No device allocation or scratch. The optional physical
// SM count selects launch geometry only; zero selects conservative geometry.
void causal_bf16_attention(const void* q, const void* k, const void* v, void* out,
    int tokens, int query_heads, int kv_heads, int q_stride, int kv_stride,
    cudaStream_t stream, std::int32_t multiprocessor_count = 0);

// The same causal formula over independently packed sequences. The device I32
// metadata is four-byte-aligned and contains a begin offset and positive length per sequence; lengths
// must be <= max_sequence_tokens and ranges must be disjoint valid token spans.
// Every query attends only its own sequence's preceding keys including itself.
// Input/output layouts, numerical behavior and alias rules match the single-
// sequence entry above. Only valid spans are written; inputs/metadata are read-only.
// No allocation or workspace. sequences is in [1,65535]. The physical SM count
// has the same resource-only meaning as in the single-sequence entry.
void packed_causal_bf16_attention(const void* q, const void* k, const void* v, void* out,
    int sequences, int max_sequence_tokens, int query_heads, int kv_heads,
    int q_stride, int kv_stride, const std::int32_t* sequence_begin,
    const std::int32_t* sequence_length, cudaStream_t stream,
    std::int32_t multiprocessor_count = 0);

// Stable FP32 online-softmax over represented BF16 Q/K/V; D is 64 or 128.
// Input storage is token-major with explicit token strides. Every head h
// reads KV head floor(h/(query_heads/kv_heads)). Each query has one [begin,end)
// key range. For causal execution, end is clipped to position[query]+1.
// All ranges and positions are device I32. Inputs remain unchanged.
void dense_bf16_attention(const void* q, const void* k, const void* v, void* out,
    int queries, int keys, int head_dim, int query_heads, int kv_heads,
    int q_stride, int kv_stride, const std::int32_t* key_begin,
    const std::int32_t* key_end, const std::int32_t* positions, bool causal,
    cudaStream_t stream);

// Decode over independent contiguous BF16 caches [lane,capacity,KV_width].
// Positions are zero-based committed positions for the current query.
// Workspace is caller-owned FP32 partial online-softmax state; it is scoped
// to the call and never persists. Invalid geometry throws. The operator may
// partition KV along its token axis, and rounds the final output to BF16.
std::size_t dense_bf16_decode_attention_workspace_capacity(int batch, int capacity,
    int head_dim, int query_heads, int kv_heads);
void dense_bf16_decode_attention(const void* q, const void* k, const void* v, void* out,
    int batch, int capacity, int head_dim, int query_heads, int kv_heads,
    const std::int32_t* positions, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops
