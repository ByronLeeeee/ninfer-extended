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

This repository distributes a BF16 NInfer v3 conversion of
[SeerRay-Lab/Xiaomi-OCR-0](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0), which
is trained from **Qwen3.5-0.8B-Base**. It contains language/vision weights and
embedded tokenizer/processor resources. There is no retraining: projection
matrices use BF16/A16, mathematical scalars follow the converter's prescribed
representations, and tied embedding/output weights remain shared.

## Required runtime

**Build [ByronLeeeee/ninfer](https://github.com/ByronLeeeee/ninfer) from its `main` branch.** This fork adds the
small Qwen3.5-0.8B kernel shapes and frontend support required by this artifact.
An unmodified upstream NInfer binary must not be assumed to load it.
The `.ninfer` file is not Transformers safetensors or GGUF.

```bash
git clone https://github.com/ByronLeeeee/ninfer.git
cd ninfer
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

hf download ByronLeeee/Xiaomi-OCR-0-Ninfer xiaomi-ocr-0-bf16.ninfer --local-dir models

./build/apps/ninfer-serve models/xiaomi-ocr-0-bf16.ninfer \
  --host 127.0.0.1 --port 8080 --model-id ocr --vision \
  --max-context 32768 --kv-capacity 131072 --max-concurrency 4 \
  --kv-dtype bf16 --prefill-chunk 1024 --no-thinking --greedy
```

Dependencies: 64-bit Linux, CUDA supporting `sm_120a`, C++20, CMake >=3.28, Ninja,
FFmpeg development libraries, libcurl >=7.85, pkg-config and libpcre2-dev.
The complete native source was clean-built using CUDA 13.2/GCC 15.2. Its binary
was executed on **RTX 5070 Ti 16 GB (WSL2)** and **RTX 6000D (Linux)**, both compute
capability 12.0. Other Blackwell architectures are not automatically qualified.
See [build and conversion details](https://github.com/ByronLeeeee/ninfer/blob/main/docs/xiaomi-ocr.md).

The 32K/four-lane command is a deployment example; the measurements below use
4K/one lane on both engines. FP8 KV is optional and needs separate output checks.

## Full vision/language prefill and decode performance

Measured on 2026-10-06 with BF16 weights and KV, 4,096-token capacity, one active
request, identical images/prompts, greedy decoding and at most 256 output tokens.
Each input has two warmups and three measured repetitions; numbers are medians.
The reference is the original Transformers checkpoint with fused causal-conv1d,
FLA GDN and attention kernels, **compiled vision and language prefill**, and
compiled decode with CUDA Graphs. Every HTTP generation entered compiled language
prefill, with actual StaticCache capacity 4,096. No slow linear-kernel fallback
was accepted. Native results use a clean build of the complete fork source.

Prefill throughput is input tokens divided by synchronized wall time for
**vision encoding plus language prefill**, not language-only throughput. Decode
counts the N-1 generated tokens after the first. HTTP request time includes
client/backend scheduling. CPU image preprocessing is outside the prefill GPU
stage; cold compilation and warmups are excluded. Full compiled GPU stages do
not mean every page's output was generated to completion.

| GPU | Input | Transformers prefill tok/s | NInfer prefill tok/s | Transformers decode tok/s | NInfer decode tok/s | Transformers / NInfer request ms |
|---|---|---:|---:|---:|---:|---:|
| RTX 5070 Ti | zh_legal | 23060 | 21924 | 173.7 | 438.4 | 839.6 / 423.1 |
| RTX 5070 Ti | en_contract | 22646 | 21201 | 169.4 | 438.7 | 829.5 / 414.8 |
| RTX 5070 Ti | dense-equations | 22197 | 19859 | 171.1 | 427.1 | 1726.0 / 810.7 |
| RTX 6000D | zh_legal | 36193 | 35306 | 233.7 | 513.8 | 579.1 / 317.2 |
| RTX 6000D | en_contract | 36154 | 35140 | 232.2 | 514.4 | 584.7 / 316.1 |
| RTX 6000D | dense-equations | 35990 | 33031 | 235.1 | 511.0 | 1206.7 / 618.1 |

Compiled Transformers prefill is slightly faster on these inputs. NInfer decode
is about 2.46–2.59x faster on the RTX 5070 Ti and 2.17–2.22x on the RTX 6000D;
HTTP request speedup is approximately 2.0–2.1x and 1.8–1.95x, respectively.
The two text pages terminate naturally; dense-equations reaches the 256-token cap.

Compiling the reference prefill reduced its wall time by 17.0–29.8% on the 5070 Ti
and 0.7–1.3% on the 6000D in a separate A-B-B-A test with compiled decode enabled
in both variants. The 6000D change is approximately unchanged in practice.

## OCR accuracy and output agreement

Accuracy against ground truth and agreement between engines measure different things.

| GPU | Transformers normalized character accuracy | NInfer normalized character accuracy | Exact output agreement | Normalized character difference |
|---|---:|---:|---:|---:|
| RTX 5070 Ti | 100% (CER 0%) | 100% (CER 0%) | 3/3 (100%) | 0% |
| RTX 6000D | 100% (CER 0%) | 100% (CER 0%) | 3/3 (100%) | 0% |

**The accuracy columns cover only two synthetic text pages, with 478 ground-truth
characters per GPU.** Both pages finish naturally (113 and 114 output tokens).
CER is Levenshtein edits divided by ground-truth character count after NFKC
normalization, whitespace removal and Markdown heading/emphasis removal;
character accuracy is 1 - CER. This is a small transcription check, not a general
document-parsing benchmark or a claim of 100% accuracy on real documents.

The agreement columns cover those two complete outputs plus the dense-equations
**256-token prefix**. Exact agreement compares raw text; normalized difference
uses NFKC and whitespace removal, with summed edits divided by the sum of the
larger output lengths. The formula page reaches the output limit and has no
full-page accuracy score in this run.

A separate historical 32K/four-lane **full-output** comparison on the RTX 5070 Ti
matched 10/11 pages exactly (90.91%), with 0.1933% weighted normalized character
difference. One handwritten-formula page differed; visual inspection favored
Transformers at two locations. That test used the earlier native binary and was
not repeated with the clean build at 4K, so it is historical evidence rather than
the acceptance result of this release. There is no independently annotated
full-page benchmark for the published native artifact on either GPU.

## Memory and reproducibility

| GPU | Transformers | NInfer | Measurement scope |
|---|---:|---:|---|
| RTX 5070 Ti | 2.80 GiB | 2.02 GiB | WSL total-GPU increment estimate |
| RTX 6000D | 3.44 GiB | 2.26 GiB | Per-process nvidia-smi |

These are warmed 4K/one-lane measurements, not 32K/four-lane serving residency.
Local video/VSR was paused; server services remained resident and idle.
All ten independent FP64 attention/RoPE/convolution qualification cases passed.
Compiler-op schema, fake-layout and initial-state ownership checks passed.

[Measurement report](https://github.com/ByronLeeeee/ninfer/blob/main/docs/xiaomi-ocr-performance.md) ·
[Sanitized data](https://github.com/ByronLeeeee/ninfer/blob/main/docs/xiaomi-ocr-results.json)

## Provenance and license

- Original checkpoint: `SeerRay-Lab/Xiaomi-OCR-0`.
- Source revision: `e4d1c4a6804bd9ef342b93d705a73af003e2ef4e`.
- Runtime source base: `Neroued/ninfer`, `594930e7b609efa4bcea3ae4f24cd9d66b5f224f`.
- Artifact: `xiaomi-ocr-0-bf16.ninfer`, 1,726,110,464 bytes.
- SHA256: `1db6a88a5947ba06e88210feff0c44f8eeee6d7b506b4744cbf9f94a0e61cb9d`.

The model and runtime are Apache-2.0 licensed. Original attribution and license
terms apply. The source model card describes the training and original research
benchmarks; those benchmark scores were not remeasured for this conversion.
