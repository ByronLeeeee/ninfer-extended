# Qwen3 Embedding

Qwen3-Embedding-0.6B runs natively in BF16 through a Qwen3 decoder and the
`TextEmbedding` Engine purpose. The decoder produces 1,024-dimensional vectors
from the final non-padding token, with optional dimensional truncation and FP32
L2 normalization. The source checkpoint has 28 layers, a 32K position limit,
16 query heads, eight KV heads and 128-dimensional attention heads.

## Convert and run

The ready-to-run BF16 artifact is available on
[Hugging Face](https://huggingface.co/ByronLeeee/Qwen3-Embedding-0.6B-Ninfer) and
[ModelScope](https://modelscope.cn/models/ByronLeeee/Qwen3-Embedding-0.6B-Ninfer).
Both model cards include complete-forward Transformers comparisons and vector
quality measurements. Use the NInfer Extended main branch to run the artifact.

Download [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
into `model/`, then use the source repository's Python environment:

```bash
python tools/qwen3_embedding/convert.py \
  --model model --out qwen3-embedding-0.6b-bf16.ninfer
cmake --build build --target ninfer-embed -j
```

Conversion keeps every BF16 parameter unchanged and verifies all 310 source
parameters against the completed artifact. Q/K/V and gate/up weights are
concatenated by row. The artifact includes the model configuration, RoPE
frequencies and original tokenizer resources.

Store document texts in a JSON file:

```json
{"texts": ["An embedding represents the meaning of a text as a vector.", "向量检索可以查找语义相关的文本。"]}
```

```bash
python tools/qwen3_embedding/embed.py \
  --artifact qwen3-embedding-0.6b-bf16.ninfer \
  --binary build/apps/ninfer-embed --input texts.json --out vectors.json
```

For retrieval queries, add `--instruction "Given a web search query, retrieve
relevant passages that answer the query"`. This creates the original
`Instruct: {instruction}\nQuery:{query}` format. Documents use their raw text.

The CPU helper handles tokenization and batches. The C++ engine performs all
decoder computation, token selection and normalization on CUDA. It supports up
to eight texts per batch and a startup token capacity of up to 32,768 **total
unpadded tokens per batch**. The helper splits batches to fit this capacity and
preserves input order. Individual passages exceeding the capacity must be split.
Use `--dimensions 32`, `128`, `256`, `512` or `1024` to select a shorter vector.
Shortening happens before L2 normalization.

## Engine API

```cpp
ninfer::EngineOptions options;
options.artifact_path = "qwen3-embedding-0.6b-bf16.ninfer";
options.purpose = ninfer::EnginePurpose::TextEmbedding;
options.max_context = 32768;
options.max_concurrency = 8;
ninfer::Engine engine(options);

ninfer::EmbeddingRunOptions execution;
execution.dimensions = 256;
auto result = engine.embed_tokens(tokenized_texts, execution);
// result.vectors contains FP32 vectors, in input order.
```

Reuse an Engine across requests to retain weights and captured CUDA Graphs.
There is no decoding loop or persistent KV cache. Buffers follow the current
packed batch shape. The cuBLAS reference backend and its extra buffers are
initialized only when explicitly selected.

`--attention auto` uses FP32 register attention for short batches and compensated
Tensor Core attention when the mean text length reaches 64 tokens. Explicit
`--attention fp32` and `--attention tensorcore` modes are available for comparison.
Multiple texts share one packed Tensor Core attention invocation per layer.
Device offsets and lengths keep each text's causal range independent. Graph
captures fix the maximum text length used for the grid and read fresh metadata
on every replay. Different partitions with the same token count and maximum
length can reuse a capture. FP32 attention also reads uploaded boundaries.

## Numerical and performance checks

```bash
cmake -S . -B build -DNINFER_BUILD_EMBEDDING_QUALIFICATION=ON
cmake --build build --target embedding-qualify -j
build/embedding-qualify
```

The independent qualification tool compares represented BF16 projections and
SwiGLU outputs against complete FP64 dot products at sampled output positions.
It checks pooling against FP64 norms, including zero vectors, shortened vectors
and very large or small finite values. Packed causal attention is checked against
FP64 dot products, stable softmax and weighted value sums, including ragged texts,
padded input strides and every registered query/KV head ratio. Inputs and metadata
remain unchanged.

Long-input causal attention can stage 64 K/V columns while updating softmax and
compensated value accumulation in the original 32-column groups. The wider
schedule is selected from the maximum sequence length, query-head count, sequence
count and the physical SM count supplied by the caller. Saturated grids retain
the original narrow kernel. Both schedules preserve BF16 probability residuals,
FP32 accumulation, independent sequence ranges and CUDA Graph execution, and
require no device scratch. Qualification covers both resource routes, their
boundary and envelopes larger than the device-resident sequence lengths.

On RTX 6000D, a single 2,048-token BF16 forward measured 23.000 ms before this
long-input schedule and 21.298 ms after it: 7.99% higher whole-model throughput,
or 96,159 token/s. These are median CUDA timings from separate A-B-B-A process
windows, with 25 warmups and 15 measured forwards per process (30 per version).
The full forward includes all decoder layers and vector pooling. A refreshed
Transformers baseline measured 21.251 ms with fast CUDA attention. Both GPUs
passed 164 embedding and 117 ASR FP64 checks, and all 249 embedding vectors from
38 complete input cases were identical to the preceding version. RTX 5070 Ti
retains the original narrow attention instructions for these shapes; its paired
timings establish regression coverage rather than a new narrow-kernel speedup.

The embedding profile adds `[6144,1024]` BF16 gate/up projection. Its fused SwiGLU
retains paired FP32 accumulators in registers through the activation and rounds
the public output to BF16. It requires no gate/up staging allocation or separate
activation launch. Shared projections use matrix geometry and token count to select compact,
short-column or wider MMA tiles. No operator dispatch depends on a model name.
For 129–160 packed tokens, the fused `[6144,1024]` profile selects a 128-row,
32-column tile when the grid supplies at least three CTA waves for the physical
SM count passed by the caller. This reduces unused tail work on RTX 5070 Ti;
RTX 6000D keeps the 64-row, 64-column tile that measured faster there. Other
extents keep their existing tiles. Stream-only callers use the established tile
without a device query. Both schedules are qualified against the FP64 oracle.

`ninfer-embed --variants --input tokens.json --out measurements.json` runs
cuBLAS/native projection and FP32/Tensor Core attention combinations in alternating
order. The output records warm GPU time, host-call latency, throughput, vectors
and tracked weight/runtime bytes. Token JSON accepts `input_ids: [[...], ...]`,
or a `cases` array with an `id` and `input_ids` for each batch.

RTX 5070 Ti and RTX 6000D validation compares vectors, retrieval ranking and semantic scores
with the original Transformers implementation. Embedding has a prefill phase
and vector output; decode throughput does not apply.
