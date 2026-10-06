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

Build dependencies: 64-bit Linux, CUDA supporting `sm_120a`, C++20, CMake ≥3.28, Ninja, FFmpeg development libraries, libcurl ≥7.85, pkg-config, and libpcre2-dev. The tested build used CUDA 13.2 and GCC 15.2 and ran on **RTX 5070 Ti 16 GB (WSL2)** and **RTX 6000D (Linux)**.

[Build and conversion instructions](https://github.com/ByronLeeeee/ninfer-qwen3.5-0.8b/blob/main/docs/xiaomi-ocr.md)

## Prefill and decode performance

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

## GPU memory

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
