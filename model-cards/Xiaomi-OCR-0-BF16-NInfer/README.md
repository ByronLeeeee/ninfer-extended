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
- mtp
- cuda
---

# Xiaomi-OCR-0 BF16 with MTP for NInfer

Native OCR and document parsing for **[NInfer Extended](https://github.com/ByronLeeeee/ninfer-extended)**,
based on Xiaomi-OCR-0 / Qwen3.5-0.8B-Base. The model file includes the original
text, vision and MTP weights, a **131,072-row BF16 proposal head**, tokenizer and
image-processing resources. Weights and KV remain BF16; target verification uses
the full vocabulary.

## Build and run

Use the current **ninfer-extended main branch**. **MTP1 is recommended**:

```bash
git clone https://github.com/ByronLeeeee/ninfer-extended.git
cd ninfer-extended
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCUDNN_ROOT=/path/to/cudnn -DCUBLAS_ROOT=/path/to/cublas
cmake --build build --target ninfer-serve -j
hf download ByronLeeee/Xiaomi-OCR-0-Ninfer xiaomi-ocr-0-bf16-MTP.ninfer --local-dir models
./build/apps/ninfer-serve models/xiaomi-ocr-0-bf16-MTP.ninfer \
  --host 127.0.0.1 --port 8080 --model-id ocr --vision \
  --max-context 32768 --kv-capacity 131072 --max-concurrency 4 \
  --kv-dtype bf16 --prefill-chunk 1024 --no-thinking --greedy \
  --spec mtp --draft-tokens 1 --lm-head-draft
```

Use `--prefill-chunk 2048` on RTX 6000D. MTP3 uses `--draft-tokens 3` and applies
the checkpoint's single trained MTP layer repeatedly. Remove the three MTP options
for ordinary BF16 decoding. Omit `--lm-head-draft` to draft with the full head.

Build requirements: 64-bit Linux (WSL2 on Windows), CUDA supporting `sm_120a`,
C++20, CMake ≥3.28, Ninja, FFmpeg development libraries, libcurl, PCRE2, cuDNN 9
and cuBLAS. [Conversion and usage](https://github.com/ByronLeeeee/ninfer-extended/blob/main/docs/xiaomi-ocr.md).

## Matched MTP1: Transformers and NInfer

Both engines use **MTP1**, the same **131,072 original BF16 proposal rows**,
and full-vocabulary target verification. NInfer decode is **2.49–2.68×** faster
on RTX 5070 Ti and **2.41–2.47×** faster on RTX 6000D.

Here, **TF MTP1 means Transformers + a custom PyTorch MTP adapter**.
Transformers 5.17.0 does not load this checkpoint's MTP module, and its native
`use_mtp=True` call rejects the stateful Qwen3.5 model. The adapter loads the
checkpoint's single trained MTP layer and uses upstream Qwen3.5 layers, SDPA,
fused causal-conv1d/FLA, and fullgraph compilation for vision, language and MTP.
Rejected drafts restore convolution state, FP32 recurrent state and KV write
positions, then replay the accepted prefix.

BF16 weights and KV, one request, 4K context, greedy decoding, 256-token output
cap. Each input and engine has two warmups, followed by three alternating
A-B-B-A/B-A-A-B cycles; values are medians of six measured requests per engine.
Prefill includes vision, language and MTP prompt processing. NInfer stage times
use CUDA events; the PyTorch reference uses synchronized wall time. Complete
request times include preprocessing, inference and the local HTTP round trip.
NInfer prefill chunks are 1,024 on 5070 Ti and 2,048 on 6000D.

| GPU | Input | TF prefill tok/s | NInfer prefill tok/s | TF decode tok/s | NInfer decode tok/s | Decode speedup | TF → NInfer prefill ms | TF → NInfer request ms | Draft acceptance (both) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | Chinese document | 19,697 | 23,100 | 265.5 | 712.6 | 2.68× | 91.74 → 78.24 | 616.0 → 328.7 | 96.49% |
| RTX 5070 Ti | English contract | 19,264 | 24,175 | 278.6 | 717.3 | 2.57× | 93.83 → 74.75 | 607.8 → 322.2 | 100.00% |
| RTX 5070 Ti | Dense formulas (256 tokens) | 19,427 | 22,041 | 283.1 | 703.9 | 2.49× | 121.90 → 107.44 | 1130.8 → 566.5 | 99.22% |
| RTX 6000D | Chinese document | 32,722 | 40,214 | 366.2 | 905.5 | 2.47× | 55.22 → 44.93 | 405.8 → 214.3 | 96.49% |
| RTX 6000D | English contract | 32,657 | 40,065 | 380.0 | 914.1 | 2.41× | 55.33 → 45.10 | 393.2 → 212.9 | 100.00% |
| RTX 6000D | Dense formulas (256 tokens) | 32,918 | 36,506 | 380.0 | 914.9 | 2.41× | 71.94 → 64.87 | 791.4 → 386.7 | 99.22% |

All three outputs match exactly on both GPUs. The Chinese and English pages
finish naturally at 113 and 114 tokens; formulas compare the same 256-token
prefix. Drafted counts, accepted counts and acceptance rates match between the
two engines: **96.49%, 100.00%, 99.22%** for the three inputs.

[Full-head, ordinary and MTP3 controls](https://github.com/ByronLeeeee/ninfer-extended/blob/main/docs/xiaomi-ocr-transformers-mtp.md).

## Native ordinary Transformers vs NInfer MTP1

Official compiled BF16 Transformers ordinary decoding versus BF16 NInfer MTP1.
This comparison includes both the engine and MTP gains. Both engines use the same
images, greedy decoding, one request, 4K context and a 256-token output cap.
Prefill includes vision encoding and language prefill. Decode is measured after
the first token; CUDA Graphs remain enabled. The reference requires fused
causal-conv1d/Gated DeltaNet and compiled vision, language prefill and decode.
NInfer uses CUDA-event stage timing; Transformers uses synchronized wall timing.
Medians follow two warmups and three alternating A-B-B-A/B-A-A-B cycles.

| GPU | Input | TF ordinary prefill tok/s | MTP1 prefill tok/s | Prefill change | TF ordinary decode tok/s | MTP1 decode tok/s | Decode speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | Chinese document | 21,764 | 24,168 | +11.0% | 166.4 | 717.1 | 4.31× |
| RTX 5070 Ti | English contract | 22,066 | 23,383 | +6.0% | 163.3 | 718.6 | 4.40× |
| RTX 5070 Ti | Dense formulas (256 tokens) | 21,353 | 22,204 | +4.0% | 162.3 | 717.6 | 4.42× |
| RTX 6000D | Chinese document | 36,383 | 40,038 | +10.0% | 232.8 | 904.4 | 3.88× |
| RTX 6000D | English contract | 36,337 | 40,141 | +10.5% | 231.9 | 916.9 | 3.95× |
| RTX 6000D | Dense formulas (256 tokens) | 36,254 | 36,284 | +0.1% | 235.4 | 910.8 | 3.87× |

Complete request time includes image preparation, inference and the local HTTP round trip:

| GPU | Input | TF request ms | MTP1 request ms | Speedup |
| --- | --- | --- | --- | --- |
| RTX 5070 Ti | Chinese document | 879.0 | 333.5 | 2.64× |
| RTX 5070 Ti | English contract | 890.3 | 326.1 | 2.73× |
| RTX 5070 Ti | Dense formulas (256 tokens) | 1818.2 | 575.7 | 3.16× |
| RTX 6000D | Chinese document | 577.2 | 214.3 | 2.69× |
| RTX 6000D | English contract | 584.2 | 212.4 | 2.75× |
| RTX 6000D | Dense formulas (256 tokens) | 1202.7 | 388.8 | 3.09× |

All three measured outputs match compiled Transformers on both GPUs.

## MTP1 and MTP3

Each column below comes from a separate paired comparison with ordinary decoding
on the same NInfer build. Decode is tok/s per request. Complete-request throughput
includes both prefill and decode.

| GPU | Input | Ordinary → MTP1 decode tok/s | Complete-request throughput change | Ordinary → MTP3 decode tok/s | Complete-request throughput change |
| --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | Chinese document | 435.1 → 682.8 | +25.8% | 462.9 → 646.5 | +19.1% |
| RTX 5070 Ti | English contract | 434.2 → 688.1 | +26.7% | 454.0 → 660.2 | +22.7% |
| RTX 5070 Ti | Dense formulas (256 tokens) | 425.1 → 678.0 | +39.0% | 444.3 → 670.7 | +32.6% |
| RTX 6000D | Chinese document | 586.3 → 909.1 | +30.9% | 588.8 → 807.6 | +22.7% |
| RTX 6000D | English contract | 586.1 → 914.8 | +32.9% | 585.6 → 750.8 | +17.9% |
| RTX 6000D | Dense formulas (256 tokens) | 580.8 → 913.8 | +41.3% | 581.3 → 845.4 | +33.3% |

Four concurrent requests, aggregate output tokens divided by complete HTTP burst time:

| GPU | Input | Ordinary → MTP1 output tok/s | Change | Ordinary → MTP3 output tok/s | Change |
| --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | Chinese document | 590.6 → 676.8 | +14.6% | 626.8 → 581.8 | -7.2% |
| RTX 5070 Ti | English contract | 610.7 → 678.8 | +11.2% | 633.0 → 603.2 | -4.7% |
| RTX 5070 Ti | Dense formulas (256 tokens) | 818.9 → 952.3 | +16.3% | 817.0 → 765.6 | -6.3% |
| RTX 6000D | Chinese document | 995.8 → 1164.3 | +16.9% | 817.0 → 841.0 | +2.9% |
| RTX 6000D | English contract | 1002.8 → 1181.0 | +17.8% | 860.1 → 726.9 | -15.5% |
| RTX 6000D | Dense formulas (256 tokens) | 1226.4 → 1509.1 | +23.1% | 1034.4 → 943.3 | -8.8% |

MTP1 improves all measured single-request and four-request workloads. MTP3 improves
single-request decode, but usually reduces four-request throughput. MTP1 remains
the default.

Draft acceptance on complete pages, with a 4,096-token output budget and natural completion:

| GPU | Complete input | MTP1 draft acceptance | MTP3 draft acceptance |
| --- | --- | --- | --- |
| RTX 5070 Ti | Chinese document | 96.49% | 45.83% |
| RTX 5070 Ti | English contract | 100.00% | 46.81% |
| RTX 5070 Ti | Classical Chinese calligraphy | 78.57% | 26.19% |
| RTX 5070 Ti | Dense formulas (complete page) | 96.02% | 46.73% |
| RTX 5070 Ti | Handwritten formulas | 98.61% | 47.60% |
| RTX 5070 Ti | Receipt | 94.12% | 44.27% |
| RTX 6000D | Chinese document | 96.49% | 45.83% |
| RTX 6000D | English contract | 100.00% | 46.81% |
| RTX 6000D | Classical Chinese calligraphy | 78.57% | 26.19% |
| RTX 6000D | Dense formulas (complete page) | 96.02% | 46.73% |
| RTX 6000D | Handwritten formulas | 98.61% | 47.60% |
| RTX 6000D | Receipt | 94.12% | 44.27% |

Acceptance is accepted draft tokens divided by all proposed tokens. MTP3 includes
all three draft positions; it is a proposal metric, not OCR character accuracy.

## Accuracy and output agreement

| Check, on each GPU | Ordinary BF16 | MTP1 | MTP3 |
| --- | --- | --- | --- |
| Six labelled text images, 1,117 scored characters | CER 0% | CER 0% | CER 0% |
| Eleven complete pages, raw agreement with ordinary decoding | Reference | 11/11 (100%) | 11/11 (100%) |
| Complete-page normalized character difference, 7,269 characters | Reference | 0% | 0% |
| Four-cycle mixed batches of 4/2/3 requests, raw agreement | Reference | 36/36 (100%) | 36/36 (100%) |
| Mixed-batch normalized character difference, 27,708 characters | Reference | 0% | 0% |
| Repeat-stable images across four mixed-batch cycles | 9/9 | 9/9 | 9/9 |

Complete-page tests use 16K context; mixed batches use 32K per request. Both retain
BF16 KV and CUDA Graphs. The unlabelled calligraphy, receipt and formula pages
measure output agreement; CER is scored on the six labelled text fixtures.
Independent FP64 and changed-input CUDA Graph tests cover projection, attention,
GDN state handling and token selection, including per-column agreement with
single-token execution.

## Model file

- Filename: `xiaomi-ocr-0-bf16-MTP.ninfer`
- Size: **2,035,979,776 bytes** (2.036 GB).
- SHA-256: `0881cb5e19af3583dc6c814f65c24dbeab3ab0c3e9e9bc7601fa4c9b975ec659`.
- Original text, vision and MTP tensors are retained byte-for-byte. Proposal rows
  are exact copies of original output-head rows selected by upstream token ranking.
- Draft token-ID SHA-256: `c348bf8d2e70d502718ebb8eeefdc794936dc8462ae357b802ba0a987fad756c`.
- One file supports ordinary decoding, MTP1 and MTP3. The non-MTP BF16 artifact
  remains available separately.
- License: Apache-2.0, following [Xiaomi-OCR-0](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0).

## Current-main regression check

On 5070 Ti, all 11 complete pages match the preceding qualified build under ordinary, MTP1 and MTP3 decoding. Six labelled text pages contain 1,117 scored characters with CER 0%. All 768 formal single-request, four-lane and mixed-batch requests pass, with no confirmed performance regression.
