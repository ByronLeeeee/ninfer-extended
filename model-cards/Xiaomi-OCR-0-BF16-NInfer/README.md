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

Both engines use BF16 weights/KV, 4K context, one request, identical images/prompts, greedy decoding, a 1,024-token prefill chunk and a 256-token output cap. Text pages finish naturally; the formula page measures the first 256 tokens. Prefill includes vision encoding and language prefill GPU time; decode starts after the first generated token.

Transformers uses fused causal-conv1d, FLA GDN and attention, compiled vision/language prefill/decode, and decode CUDA Graphs; computational fallback was checked. The audited compiled Transformers baseline was measured on 2026-10-06 with two warmups and three formal runs; current NInfer was rechecked on 2026-10-07 with 10–16 formal runs. Tables report medians on the same GPU and settings, from separate rounds.

| GPU | Input | TF prefill tok/s | NInfer prefill tok/s | Prefill change | TF decode tok/s | NInfer decode tok/s | Decode speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | Chinese document | 23,060 | 22,490 | -2.5% | 173.7 | 423.9 | 2.44× |
| RTX 5070 Ti | English contract | 22,646 | 21,927 | -3.2% | 169.4 | 422.1 | 2.49× |
| RTX 5070 Ti | Dense formulas (256 tokens) | 22,197 | 20,640 | -7.0% | 171.1 | 420.6 | 2.46× |
| RTX 6000D | Chinese document | 36,193 | 37,718 | +4.2% | 233.7 | 543.0 | 2.32× |
| RTX 6000D | English contract | 36,154 | 38,483 | +6.4% | 232.2 | 541.4 | 2.33× |
| RTX 6000D | Dense formulas (256 tokens) | 35,990 | 35,428 | -1.6% | 235.1 | 537.8 | 2.29× |

Changes and speedups use Transformers as the baseline: prefill change is `(NInfer / Transformers − 1) × 100%`; decode speedup is `NInfer / Transformers`.

## OCR accuracy and output agreement

Two synthetic text pages contain 478 annotated characters. Character accuracy is `1 − CER`, scored after character normalization and removal of whitespace and Markdown markers. Output agreement compares both complete text pages and the 256-token formula output.

| GPU | Transformers accuracy | NInfer accuracy | Exact output agreement | Character difference |
| --- | --- | --- | --- | --- |
| RTX 5070 Ti | 100% (CER 0%) | 100% (CER 0%) | 3/3 (100%) | 0% |
| RTX 6000D | 100% (CER 0%) | 100% (CER 0%) | 3/3 (100%) | 0% |

A separate 11-page full-output comparison found 10/11 exact matches with Transformers and a 0.1933% character difference, with differences on one handwritten formula page. The current source preserves all 11 NInfer outputs; each GPU passes 225 independent FP64 operator checks.

## Artifact and license

`xiaomi-ocr-0-bf16.ninfer` · 1,726,110,464 bytes · BF16 · Apache-2.0

SHA256: `1db6a88a5947ba06e88210feff0c44f8eeee6d7b506b4744cbf9f94a0e61cb9d`

[Measurements](gpu-results.json) · [Artifact manifest](artifact-manifest.json)
