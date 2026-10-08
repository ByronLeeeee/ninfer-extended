---
license: apache-2.0
base_model: SeerRay-Lab/Xiaomi-OCR-0
pipeline_tag: image-text-to-text
library_name: ninfer
language:
- zh
- en
tags:
- ninfer
- bf16
- ocr
- document-parsing
- qwen3_5
- cuda
---

# Xiaomi-OCR-0 BF16 for NInfer

This BF16 conversion of [Xiaomi-OCR-0](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0) supports OCR and document parsing. The source model is based on Qwen3.5-0.8B-Base. The artifact includes language and vision weights, the tokenizer and image-processing configuration.

## Build and run

Build the **[ninfer-extended](https://github.com/ByronLeeeee/ninfer-extended)** main branch to run this artifact.

```bash
git clone https://github.com/ByronLeeeee/ninfer-extended.git
cd ninfer-extended
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCUDNN_ROOT=/path/to/cudnn -DCUBLAS_ROOT=/path/to/cublas
cmake --build build --target ninfer-serve -j
pip install huggingface_hub
hf download ByronLeeee/Xiaomi-OCR-0-Ninfer xiaomi-ocr-0-bf16.ninfer --local-dir models
./build/apps/ninfer-serve models/xiaomi-ocr-0-bf16.ninfer \
  --host 127.0.0.1 --port 8080 --model-id ocr --vision \
  --max-context 32768 --kv-capacity 131072 --max-concurrency 4 \
  --kv-dtype bf16 --prefill-chunk 1024 --no-thinking --greedy
```

This starts an OCR service with 32K context per request, four concurrent requests and BF16 weights/KV. Build dependencies: 64-bit Linux (WSL2 on Windows), CUDA supporting `sm_120a`, C++20, CMake ≥3.28, Ninja, FFmpeg development libraries, libcurl, PCRE2, cuDNN 9 and cuBLAS. Omit `CUBLAS_ROOT` when cuBLAS is installed with CUDA.

[Build and conversion guide](https://github.com/ByronLeeeee/ninfer-extended/blob/main/docs/xiaomi-ocr.md)

## Transformers vs NInfer performance

On RTX 5070 Ti, full vision/language prefill throughput improves by **0.8–12.9%** and decode reaches **2.83–2.99×** compiled Transformers. On RTX 6000D, prefill changes by **−1.3% to +4.1%** and decode reaches **2.29–2.34×**.

Both engines use BF16 weights/KV, 4K context, one request, identical images/prompts, greedy decoding and a 256-token output cap. NInfer uses 1,024-token prefill chunks. Text pages finish naturally; the formula page measures its first 256 output tokens. Results are medians.

Transformers runs compiled vision/language prefill and decode with verified fused causal-conv1d and Gated DeltaNet kernels. Prefill includes vision encoding and language prefill; Transformers uses synchronized stage timing and NInfer uses CUDA events. Decode starts after the first generated token.

| GPU | Input | TF prefill tok/s | NInfer prefill tok/s | Prefill change | TF decode tok/s | NInfer decode tok/s | Decode speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| RTX 5070 Ti | Chinese document | 19,991 | 22,240 | +11.3% | 140.7 | 413.3 | 2.94× |
| RTX 5070 Ti | English contract | 19,793 | 22,338 | +12.9% | 138.6 | 414.7 | 2.99× |
| RTX 5070 Ti | Dense formulas (256 tokens) | 20,560 | 20,734 | +0.8% | 145.0 | 410.7 | 2.83× |
| RTX 6000D | Chinese document | 36,426 | 37,921 | +4.1% | 231.9 | 542.4 | 2.34× |
| RTX 6000D | English contract | 36,439 | 37,664 | +3.4% | 231.8 | 540.8 | 2.33× |
| RTX 6000D | Dense formulas (256 tokens) | 36,072 | 35,615 | -1.3% | 234.7 | 538.3 | 2.29× |

Prefill change is `(NInfer / Transformers − 1) × 100%`; decode speedup is `NInfer / Transformers`.

### End-to-end request time

Request time includes preprocessing, inference and the local HTTP round trip.

| GPU | Input | Transformers request time | NInfer request time | Speedup |
|---|---|---:|---:|---:|
| RTX 5070 Ti | Chinese document | 1006.0 ms | 437.4 ms | 2.30× |
| RTX 5070 Ti | English contract | 1004.9 ms | 446.0 ms | 2.25× |
| RTX 5070 Ti | Dense formulas (256 tokens) | 2009.8 ms | 834.7 ms | 2.41× |
| RTX 6000D | Chinese document | 582.0 ms | 298.0 ms | 1.95× |
| RTX 6000D | English contract | 582.7 ms | 298.8 ms | 1.95× |
| RTX 6000D | Dense formulas (256 tokens) | 1203.5 ms | 581.2 ms | 2.07× |

## OCR accuracy and output agreement

The two complete text pages contain 478 annotated characters. Character accuracy is `1 − CER`, scored after character normalization and removal of whitespace and Markdown markers. Output agreement covers both complete text outputs and the 256-token formula output.

| GPU | Transformers character accuracy | NInfer character accuracy | Exact output agreement | Character difference |
|---|---:|---:|---:|---:|
| RTX 5070 Ti | 100% (CER 0%) | 100% (CER 0%) | 3/3 (100%) | 0% |
| RTX 6000D | 100% (CER 0%) | 100% (CER 0%) | 3/3 (100%) | 0% |

On both GPUs, NInfer also scores **100% character accuracy (CER 0%)** on six annotated text images totaling 1,117 normalized characters. BF16 projection, fused SwiGLU and D64 attention pass independent FP64 checks, including CUDA Graph replay and dispatch boundaries.

## Artifact and license

`xiaomi-ocr-0-bf16.ninfer` · 1,726,110,464 bytes · BF16 · Apache-2.0

SHA256: `1db6a88a5947ba06e88210feff0c44f8eeee6d7b506b4744cbf9f94a0e61cb9d`

[Measurements](gpu-results.json) · [Artifact manifest](artifact-manifest.json)
