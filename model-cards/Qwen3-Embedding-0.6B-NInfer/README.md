---
license: apache-2.0
base_model: Qwen/Qwen3-Embedding-0.6B
pipeline_tag: feature-extraction
library_name: ninfer
tags:
- ninfer
- embeddings
- bf16
- cuda
- qwen3
---

# Qwen3-Embedding-0.6B for NInfer Extended

A native BF16 conversion of [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) for text embeddings, semantic retrieval and similarity. All 310 source BF16 parameters are preserved, and the tokenizer is embedded in the artifact. Tested on RTX 5070 Ti and RTX 6000D.

## Build and run

Use **[NInfer Extended](https://github.com/ByronLeeeee/ninfer-extended)** from the main branch to run this artifact.

```bash
git clone https://github.com/ByronLeeeee/ninfer-extended.git
cd ninfer-extended
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCUDNN_ROOT=/path/to/cudnn -DCUBLAS_ROOT=/path/to/cublas
cmake --build build --target ninfer-embed -j
pip install "transformers==5.17.0" numpy
pip install huggingface_hub
hf download ByronLeeee/Qwen3-Embedding-0.6B-Ninfer qwen3-embedding-0.6b-bf16.ninfer --local-dir models
```

Build dependencies: 64-bit Linux (WSL2 on Windows), CUDA supporting `sm_120a`, C++20, CMake ≥3.28, Ninja, FFmpeg development libraries, libcurl, PCRE2, cuDNN 9 and cuBLAS. Omit `CUBLAS_ROOT` when CUDA provides cuBLAS.

Save the input as `texts.json`:

```json
{"texts": ["A passage for semantic retrieval.", "Text embeddings support semantic search."]}
```

```bash
python tools/qwen3_embedding/embed.py \
  --artifact models/qwen3-embedding-0.6b-bf16.ninfer \
  --binary build/apps/ninfer-embed --input texts.json --out vectors.json \
  --dimensions 1024
```

The output contains normalized 1,024-dimensional vectors. For retrieval queries, add `--instruction "Given a web search query, retrieve relevant passages that answer the query"`; documents use their raw text.

The frontend batches up to eight texts and 32,768 total unpadded tokens at a time. Larger collections are split into batches. Supported output dimensions are 32, 128, 256, 512 and 1024.

[Conversion guide and Engine API](https://github.com/ByronLeeeee/ninfer-extended/blob/main/docs/qwen3-embedding.md)

## Transformers vs NInfer performance

Both engines use BF16 and produce normalized 1,024-dimensional vectors. GPU timings include the complete language prefill and vector pooling on pre-tokenized input; weight loading, tokenization and initial graph capture are excluded. Text embedding has no decode phase.

Each Transformers baseline uses the faster eager or fullgraph-compiled path. All 310 parameter tensors and computation modules were checked on CUDA, with fast CUDA attention and MATH SDPA disabled. Compiled execution has zero graph breaks and error suppression is disabled. Environment: PyTorch 2.14.0+cu132 and Transformers 5.17.0; CUDA toolkit 13.3 on 5070 Ti and 13.2 on 6000D.

| GPU | Texts × tokens | TF GPU ms | NInfer GPU ms | TF token/s | NInfer token/s | Throughput change |
| --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti 16 GB | 1×128 | 3.266 | 2.578 | 39,196 | 49,656 | +26.69% |
| RTX 5070 Ti 16 GB | 1×512 | 8.781 | 8.150 | 58,308 | 62,826 | +7.75% |
| RTX 5070 Ti 16 GB | 1×2048 | 42.161 | 37.610 | 48,576 | 54,454 | +12.10% |
| RTX 5070 Ti 16 GB | 4×128 | 7.418 | 7.340 | 69,019 | 69,753 | +1.06% |
| RTX 5070 Ti 16 GB | 8×512 | 62.459 | 56.905 | 65,579 | 71,979 | +9.76% |
| RTX 6000D | 1×128 | 2.797 | 1.980 | 45,763 | 64,657 | +41.29% |
| RTX 6000D | 1×512 | 6.187 | 4.616 | 82,757 | 110,912 | +34.02% |
| RTX 6000D | 1×2048 | 21.251 | 21.298 | 96,371 | 96,159 | -0.22% |

Changes use Transformers as the denominator: `(TF time / NInfer time − 1) × 100%`. The two engines are effectively tied for 2,048 tokens on 6000D. Values are medians: 30 measured NInfer forwards per case, or 62 for local long input. The audited baseline and NInfer runs use the same GPU and input settings in separate windows; the 6000D 2,048-token reference was refreshed with 31 measured forwards.

## Vector quality and result agreement

Both eager and compiled Transformers outputs are checked. Vector agreement covers 249 vectors; semantic similarity uses the first 100 STS-B pairs; the Chinese/English retrieval set contains eight queries.

| GPU | TF path | Mean cosine | TF Spearman | NInfer Spearman | Top1 correct | Top1 / Top3 agreement |
| --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti 16 GB | eager | 0.99980168 | 0.946255 | 0.946417 | 8/8 | 100% / 100% |
| RTX 5070 Ti 16 GB | compiled | 0.99979238 | 0.946207 | 0.946417 | 8/8 | 100% / 100% |
| RTX 6000D | eager | 0.99980005 | 0.946255 | 0.946417 | 8/8 | 100% / 100% |
| RTX 6000D | compiled | 0.99978163 | 0.946561 | 0.946417 | 8/8 | 100% / 100% |

Both engines retrieve the correct Top1 document for all eight queries, with identical Top1 and Top3 document sets. Floating-point vectors differ slightly, with mean cosine around 0.9998 and closely matching Spearman scores. Each GPU passes 164 independent FP64 mathematical checks.

## Memory and artifact

Tracked GPU weight storage is about 1.110 GiB, with no KV cache. Explicit runtime buffers use about 15.04 MiB for 4×128, 120.14 MiB for 8×512, and 0.938 GiB at 32K total capacity; CUDA driver and Graph objects add overhead.

`qwen3-embedding-0.6b-bf16.ninfer` · 1,203,049,125 bytes · BF16 · Apache-2.0

SHA256: `3229813fdae278ab6871f712e8a943fc4c8949d5580567e89f27f2f18d4e6ae6`

[Measurements](https://huggingface.co/ByronLeeee/Qwen3-Embedding-0.6B-Ninfer/blob/main/gpu-results.json) · [Artifact manifest](https://huggingface.co/ByronLeeee/Qwen3-Embedding-0.6B-Ninfer/blob/main/artifact-manifest.json) · [Conversion record](https://huggingface.co/ByronLeeee/Qwen3-Embedding-0.6B-Ninfer/blob/main/conversion.json)

[Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) · [STS-B](https://huggingface.co/datasets/sentence-transformers/stsb)
