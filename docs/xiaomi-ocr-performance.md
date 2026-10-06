# Xiaomi-OCR-0 test results

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

[Test data](xiaomi-ocr-results.json)

## Prefill compilation comparison

The A-B-B-A test alternates eager and compiled vision/language prefill while
keeping compiled decode enabled. Each variant has six measured requests per
input after phase warmups. Image grid metadata stays outside compilation.

| GPU | Page | Eager prefill ms | Compiled prefill ms | Time reduction |
|---|---|---:|---:|---:|
| RTX 5070 Ti | zh_legal | 109.20 | 76.66 | 29.8% |
| RTX 5070 Ti | en_contract | 103.04 | 75.09 | 27.1% |
| RTX 5070 Ti | dense-equations | 129.41 | 107.43 | 17.0% |
| RTX 6000D | zh_legal | 49.45 | 48.83 | 1.3% |
| RTX 6000D | en_contract | 49.50 | 48.99 | 1.0% |
| RTX 6000D | dense-equations | 65.78 | 65.32 | 0.7% |

Both GPUs captured five graphs across vision, language, and decode, with no
recorded graph breaks. All 36 measured token sequences per GPU matched the
eager reference. The 6000D timing improvement was small.

## Numerical and compilation checks

All ten independent FP64 attention, RoPE, and convolution checks passed.
Compiler-op schema, fake-layout, and initial-state ownership checks also passed.
Execution counters confirmed compiled language prefill for all 15 HTTP
generations on each GPU, with a 4,096-token cache.

Cache storage is initialized outside Dynamo to keep addresses stable for decode
CUDA Graphs. This fixes the local decode slowdown caused by skipping cache
initialization.
