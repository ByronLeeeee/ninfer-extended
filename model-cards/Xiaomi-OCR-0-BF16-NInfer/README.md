---
license: apache-2.0
base_model: SeerRay-Lab/Xiaomi-OCR-0
pipeline_tag: image-text-to-text
tags:
  - ninfer
  - bf16
  - ocr
  - document-parsing
  - qwen3_5
  - cuda
---


# Xiaomi-OCR-0 BF16 for NInfer

This repository provides a BF16 NInfer model of [Xiaomi-OCR-0](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0) for OCR and document parsing. The original model was trained from **Qwen3.5-0.8B-Base**. The conversion includes language and vision weights, the tokenizer, and image processing configuration.

## Setup and usage

Build the `main` branch of [ninfer-qwen3.5-0.8b](https://github.com/ByronLeeeee/ninfer-qwen3.5-0.8b). This NInfer fork adds Qwen3.5-0.8B support and has been tested with Xiaomi-OCR-0.

```bash
git clone https://github.com/ByronLeeeee/ninfer-qwen3.5-0.8b.git
cd ninfer-qwen3.5-0.8b
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

hf download ByronLeeee/Xiaomi-OCR-0-Ninfer xiaomi-ocr-0-bf16.ninfer --local-dir models

./build/apps/ninfer-serve models/xiaomi-ocr-0-bf16.ninfer \
  --host 127.0.0.1 --port 8080 --model-id ocr --vision \
  --max-context 32768 --kv-capacity 131072 --max-concurrency 4 \
  --kv-dtype bf16 --prefill-chunk 1024 --no-thinking --greedy
```

The command starts an OCR service with a 32K context limit, four concurrent requests, and BF16 weights and KV cache.

Build dependencies: 64-bit Linux, CUDA supporting `sm_120a`, C++20, CMake ≥3.28, Ninja, FFmpeg development libraries, libcurl ≥7.85, pkg-config, and libpcre2-dev. The initial build used CUDA 13.2 and GCC 15.2 and ran on **RTX 5070 Ti 16 GB (WSL2)** and **RTX 6000D (Linux)**.

[Build and conversion instructions](https://github.com/ByronLeeeee/ninfer-qwen3.5-0.8b/blob/main/docs/xiaomi-ocr.md)

## RTX 5070 Ti and RTX 6000D latest results

Measured on 2026-10-06 with runtime source `49574557`. Both GPUs use the same
Xiaomi-OCR-0 BF16 model, BF16 KV, 4K context per request, a 1,024-token prefill
chunk, greedy decoding, decode CUDA Graphs and a 256-token output cap. Prefix
reuse and image caching are disabled. Text pages finish naturally; the dense
formula case measures the first 256 output tokens.

RTX 5070 Ti runs under Ubuntu WSL2 with CUDA 13.3.33, GCC 13.3.0 and driver
617.14. RTX 6000D runs under Linux with CUDA 13.2.86, GCC 15.2.0 and driver
595.99.02. Each input/version has two warmups and five adjacent A-B-B-A cycles,
giving ten formal bursts per version. The tables show medians.

### Current NInfer speeds

| GPU | Requests | Input | Prompt tokens | Vision+language prefill tok/s | Decode tok/s/request | Aggregate completion tok/s |
|---|---:|---|---:|---:|---:|---:|
| RTX 5070 Ti | 1 | zh_legal | 1,807 | 22,358 | 421.9 | 274.3 |
| RTX 5070 Ti | 1 | en_contract | 1,807 | 22,198 | 419.3 | 261.5 |
| RTX 5070 Ti | 1 | dense-equations | 2,368 | 20,443 | 418.7 | 312.0 |
| RTX 5070 Ti | 2 | zh_legal | 1,807 | 22,291 | 409.8 | 428.1 |
| RTX 5070 Ti | 2 | en_contract | 1,807 | 22,362 | 410.3 | 430.9 |
| RTX 5070 Ti | 4 | zh_legal | 1,807 | 22,072 | 379.9 | 619.8 |
| RTX 5070 Ti | 4 | en_contract | 1,807 | 21,957 | 377.1 | 614.5 |
| RTX 6000D | 1 | zh_legal | 1,807 | 37,911 | 541.9 | 381.3 |
| RTX 6000D | 1 | en_contract | 1,807 | 37,912 | 540.6 | 384.7 |
| RTX 6000D | 1 | dense-equations | 2,368 | 35,475 | 538.3 | 441.0 |
| RTX 6000D | 2 | zh_legal | 1,807 | 37,896 | 527.3 | 637.1 |
| RTX 6000D | 2 | en_contract | 1,807 | 37,991 | 526.7 | 643.4 |
| RTX 6000D | 4 | zh_legal | 1,807 | 37,876 | 487.1 | 951.8 |
| RTX 6000D | 4 | en_contract | 1,807 | 38,342 | 487.4 | 960.3 |

**Vision+language prefill** divides prompt tokens by the combined GPU time for
vision encoding and language prefill. CPU image preprocessing is outside this
measurement. **Decode** is the per-request speed after the first generated
token. **Aggregate throughput** divides all completion tokens in a burst by
client wall time, including preparation, prefill and scheduling.

### Single-request vision and language timings

| GPU | Input | Vision encoding ms | Language prefill ms | Language prefill tok/s |
|---|---|---:|---:|---:|
| RTX 5070 Ti | zh_legal | 42.08 | 38.70 | 46,698 |
| RTX 5070 Ti | en_contract | 42.09 | 39.08 | 46,237 |
| RTX 5070 Ti | dense-equations | 63.15 | 52.37 | 45,239 |
| RTX 6000D | zh_legal | 24.66 | 22.92 | 78,852 |
| RTX 6000D | en_contract | 24.81 | 22.73 | 79,503 |
| RTX 6000D | dense-equations | 37.84 | 28.95 | 81,791 |

### RTX 6000D paired operator comparison

The baseline is the server's existing NInfer executable before this operator
rollout. Both versions read the same BF16 model. Speeds are tok/s; decode is
per request.

| Requests | Input | Baseline vision+language prefill | Updated vision+language prefill | Baseline decode | Updated decode | Decode change |
|---:|---|---:|---:|---:|---:|---:|
| 1 | zh_legal | 35,238 | 37,911 | 514.2 | 541.9 | +5.39% |
| 1 | en_contract | 35,201 | 37,912 | 515.0 | 540.6 | +4.97% |
| 1 | dense-equations | 32,858 | 35,475 | 511.1 | 538.3 | +5.32% |
| 2 | zh_legal | 35,254 | 37,896 | 356.3 | 527.3 | +47.99% |
| 2 | en_contract | 35,310 | 37,991 | 355.9 | 526.7 | +47.97% |
| 4 | zh_legal | 35,349 | 37,876 | 341.2 | 487.1 | +42.74% |
| 4 | en_contract | 35,327 | 38,342 | 341.3 | 487.4 | +42.83% |

Combined vision/language prefill improves by **7.1–8.5%**. Single-request decode
improves by **5.0–5.4%**, two-request decode by **48.0%**, and four-request decode
by **42.7–42.8%**. Aggregate throughput improves by **30.8–31.3%** with two
requests and **24.5–24.8%** with four. For the single Chinese page, vision time
falls from 28.48 to 24.66 ms, while language prefill is 22.73 versus 22.92 ms;
its prefill gain comes mainly from vision encoding.

The 5070 Ti's last projection step was paired against `b33b81b8`, which already
included the earlier operator updates. That step adds **0–2.8%** combined
prefill throughput with essentially unchanged decode. The earlier small-batch
update improved two-/four-request decode by **35.2–39.5%**. The [per-iteration tables](https://github.com/ByronLeeeee/ninfer-qwen3.5-0.8b/blob/main/docs/xiaomi-ocr-performance.md) record each paired baseline.

### Output agreement and OCR accuracy

| GPU | Formal responses | Exact baseline/updated agreement | Character difference | CER on two text pages | FP64 checks |
|---|---:|---:|---:|---:|---:|
| RTX 5070 Ti | 340 | 100% | 0% | 0% | 225/225 |
| RTX 6000D | 300 | 100% | 0% | 0% | 225/225 |

The text accuracy check uses two synthetic pages with **478 annotated
characters**. Exact agreement covers the complete text outputs and the formula
prefix in the timed workload. The 5070 Ti count includes 300 primary responses
and 40 extra single-English confirmation responses. A separate **32K per
request / four-request** test completes Chinese text, an English contract,
mixed numbers and a formula page on both GPUs; sequential and concurrent
outputs match across versions. The 5070 Ti also passes 48 additional
same-process public-Op FP64 checks.

### Qwen3.8-27B control on RTX 6000D

The same candidate was tested with the existing mixed NVFP4/FP8/BF16/integer
Qwen3.8-27B artifact, FP8 KV and **DFlash2 K7**. Context is 16K per request,
with one/four requests, a 1,024-token prefill chunk and a 128-token output cap.
The controlled task copies 72 or 300 synthetic records (2,504/10,256 prompt
tokens). Sequential A-B-B-A phases use two warmups and three formal bursts per
input/phase, giving six bursts per input/version.

| Requests | Prompt tokens | Baseline prefill | Updated prefill | Baseline decode | Updated decode | Decode change |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2,504 | 6,480 | 6,493 | 352.5 | 352.6 | +0.03% |
| 1 | 10,256 | 6,415 | 6,416 | 347.2 | 346.9 | -0.09% |
| 4 | 2,504 | 6,498 | 6,505 | 295.5 | 295.5 | +0.00% |
| 4 | 10,256 | 6,440 | 6,437 | 283.0 | 282.8 | -0.08% |

Prefill, decode and aggregate changes are within **±0.21%**; all **120**
formal responses agree exactly. The measured 27B workloads keep their existing
large-model computation routes. An open-ended single-request writing control
measured **116–131 decode tok/s**, with changes within ±0.1%. DFlash2 acceptance
is 100% on copying versus 26.1% on that writing task.

### Earlier language-only context benchmark on RTX 6000D

This earlier executable was tested with pretokenized repetitions of OCR output
text, BF16 weights/KV, one request, a 262,144-token capacity and a 1,024-token
prefill chunk. Each length has one warmup and three formal runs, followed by
256 fixed decode steps. These numbers exclude vision encoding and tokenization.

| Prompt tokens | Language prefill ms | Language prefill tok/s | Decode tok/s |
|---:|---:|---:|---:|
| 2,048 | 21.42 | 95,630 | 513.6 |
| 8,192 | 95.40 | 85,866 | 491.2 |
| 32,768 | 556.06 | 58,929 | 440.7 |
| 65,536 | 1580.05 | 41,477 | 356.6 |
| 131,072 | 5020.59 | 26,107 | 280.0 |
| 260,000 | 17883.40 | 14,539 | 192.6 |

The current candidate's image-OCR measurements are recorded above; this older
pure-text series has not been rerun with it.

[GPU measurement data](../../docs/xiaomi-ocr-gpu-results.json) · [Full test report](https://github.com/ByronLeeeee/ninfer-qwen3.5-0.8b/blob/main/docs/xiaomi-ocr-performance.md)

## Initial Transformers comparison

The following paired comparison used the initial NInfer build, before the operator updates above.

Measured on 2026-10-06 with BF16 weights and KV cache, a 4K context, one active request, identical images and prompts, greedy decoding, and up to 256 output tokens. Each input had two warmups and three measured runs; the table reports medians.

The Transformers baseline used fused causal-conv1d, FLA GDN, and attention kernels. Vision encoding, language prefill, and decode were compiled, with CUDA Graphs for decode. NInfer was built from the fork's complete C++/CUDA source.

Prefill throughput is input tokens divided by the **combined vision encoding and language prefill time**. Decode measures generation after the first token. Request time includes client and server scheduling. Cold compilation and warmups are excluded; prefill timing excludes CPU image preprocessing.

| GPU | Input | Transformers prefill tok/s | NInfer prefill tok/s | Transformers decode tok/s | NInfer decode tok/s | Transformers / NInfer request ms |
|---|---|---:|---:|---:|---:|---:|
| RTX 5070 Ti | zh_legal | 23060 | 21924 | 173.7 | 438.4 | 839.6 / 423.1 |
| RTX 5070 Ti | en_contract | 22646 | 21201 | 169.4 | 438.7 | 829.5 / 414.8 |
| RTX 5070 Ti | dense-equations | 22197 | 19859 | 171.1 | 427.1 | 1726.0 / 810.7 |
| RTX 6000D | zh_legal | 36193 | 35306 | 233.7 | 513.8 | 579.1 / 317.2 |
| RTX 6000D | en_contract | 36154 | 35140 | 232.2 | 514.4 | 584.7 / 316.1 |
| RTX 6000D | dense-equations | 35990 | 33031 | 235.1 | 511.0 | 1206.7 / 618.1 |

Compiled Transformers prefill is slightly faster on these inputs. NInfer decode is **2.46–2.59×** faster on the 5070 Ti and **2.17–2.22×** faster on the 6000D. Overall request speed improves by about **2.0–2.1×** and **1.8–1.95×**, respectively. Both text pages run to completion; the formula input is measured over its first 256 output tokens.

A separate A-B-B-A comparison found that compiling Transformers prefill reduced its time by 17.0%–29.8% on the 5070 Ti and 0.7%–1.3% on the 6000D.

## OCR accuracy and output agreement

Accuracy was measured on two synthetic text pages with 478 annotated characters. Output agreement covers both complete text outputs and the first 256 tokens of the formula input.

| GPU | Transformers normalized character accuracy | NInfer normalized character accuracy | Exact output agreement | Normalized character difference |
|---|---:|---:|---:|---:|
| RTX 5070 Ti | 100% (CER 0%) | 100% (CER 0%) | 3/3 (100%) | 0% |
| RTX 6000D | 100% (CER 0%) | 100% (CER 0%) | 3/3 (100%) | 0% |

Character accuracy is 1−CER, where CER is character edit distance divided by the annotated character count. Scoring applies NFKC normalization and removes whitespace and Markdown heading/emphasis markers. Exact agreement compares raw outputs. Character difference uses NFKC and whitespace removal, weighted by output length.

An earlier NInfer build was also tested on 11 complete pages using the 5070 Ti with a 32K context and four concurrent requests. **10/11 pages matched exactly (90.91%)**, with a **0.1933%** character difference. One handwritten formula page differed; visual inspection found two locations where Transformers was more accurate.

## GPU memory in the initial comparison

These measurements use a warmed 4K context and one active request. Local video playback and VSR were paused; other server services were idle.

| GPU | Transformers | NInfer | Measurement scope |
|---|---:|---:|---|
| RTX 5070 Ti | 2.80 GiB | 2.02 GiB | WSL total-GPU increment estimate |
| RTX 6000D | 3.44 GiB | 2.26 GiB | Per-process nvidia-smi |

[Test report](https://github.com/ByronLeeeee/ninfer-qwen3.5-0.8b/blob/main/docs/xiaomi-ocr-performance.md) · [Test data](https://github.com/ByronLeeeee/ninfer-qwen3.5-0.8b/blob/main/docs/xiaomi-ocr-results.json)

## Model file and license

- Original checkpoint: `SeerRay-Lab/Xiaomi-OCR-0`.
- Source revision: `e4d1c4a6804bd9ef342b93d705a73af003e2ef4e`.
- Runtime source base: `Neroued/ninfer`, `594930e7b609efa4bcea3ae4f24cd9d66b5f224f`.
- Artifact: `xiaomi-ocr-0-bf16.ninfer`, 1,726,110,464 bytes.
- SHA256: `1db6a88a5947ba06e88210feff0c44f8eeee6d7b506b4744cbf9f94a0e61cb9d`.


The model and runtime use the Apache-2.0 license.
