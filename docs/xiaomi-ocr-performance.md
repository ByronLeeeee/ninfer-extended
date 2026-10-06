# Xiaomi-OCR-0 test results

## RTX 5070 Ti small-batch BF16 operators

Measured on 2026-10-06 on RTX 5070 Ti 16 GB under Ubuntu WSL2, driver 617.14,
with CUDA 13.3.33 and GCC 13.3.0. The baseline is `75ad8669`, which already
includes the D64 vision and single-column projection optimizations reported below.
Both variants use unchanged BF16 weights and KV, a 4,096-token context per
request, a 1,024-token prefill chunk, greedy decoding and a 256-token output cap.
Prefix reuse and media caching are disabled; decode CUDA Graphs remain enabled.

The new routes share the BF16 GEMV/SIMT computation cores and use matrix shape
and token count to select a kernel through the public Ops:

- Small-batch projections use exact T=2/3/4 kernels and a masked T=5–8 kernel.
  The vocabulary projection and larger token batches retain their MMA routes.
- Residual and split-output projections write their final destinations directly
  in both prefill and decode, removing intermediate projection buffers and
  separate add/split launches. The corresponding projection Ops need no scratch.
- Fused SwiGLU pairs gate/up weight rows in the same warp and writes the final
  activation directly. It preserves BF16 projection and SiLU rounding and keeps
  the existing stored weight layout. The selected intervals are T=1–4 for
  7,168×1,024 and T=1–8 for 14,336×5,120; larger batches use materialized MMA.
- Control projections retain compile-time geometry for constant division and
  loop optimization.

The A-B-B-A sequence uses two warmups and five measured requests or bursts per
input per phase, giving ten measurements per variant. Multi-request bursts
submit the same page simultaneously on each lane. Tables report medians.
Prefill includes vision encoding and language prefill, excluding CPU image
preprocessing. Concurrent decode speed is per request. End-to-end total speed
is all completion tokens divided by the burst's client wall time, including
prefill and scheduling.

| Input | Baseline prefill tok/s | Optimized prefill tok/s | Baseline decode tok/s | Optimized decode tok/s | Decode change |
|---|---:|---:|---:|---:|---:|
| zh_legal | 20,248 | 21,398 | 403.5 | 412.7 | +2.3% |
| en_contract | 20,696 | 21,383 | 405.0 | 412.0 | +1.7% |
| dense-equations | 19,412 | 19,535 | 399.0 | 403.6 | +1.2% |

| Lanes | Input | Baseline decode tok/s/request | Optimized decode tok/s/request | Decode gain | Baseline end-to-end total tok/s | Optimized end-to-end total tok/s | Total gain |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | zh_legal | 283.9 | 396.0 | 39.5% | 329.2 | 397.0 | 20.6% |
| 2 | en_contract | 284.0 | 395.8 | 39.4% | 332.0 | 401.2 | 20.9% |
| 4 | zh_legal | 271.4 | 366.8 | 35.2% | 483.2 | 555.8 | 15.0% |
| 4 | en_contract | 271.5 | 369.1 | 36.0% | 491.1 | 567.4 | 15.5% |

Observed interval-average decode batches reached 1.97 and
3.88, confirming that both multi-request routes ran.
The main throughput gain comes from small-batch decode. Single-request decode
improves by 1.2–2.3% in this comparison.

All 60 measured single-request outputs and 240 concurrent-request outputs were
stable and matched the baseline exactly. Raw agreement is 100%, and normalized
character difference is 0%. Both variants score CER 0% on the two complete text
pages, totaling 478 annotated characters; the formula page compares its first
256 output tokens.

A separate 32K-context/four-request check submits Chinese text, English text,
mixed numbers and a formula page together. All four pages finish and match both
sequential execution and the baseline. Its interval-average decode batch reaches
3.38.

All **145 independent FP64 checks** passed: 135 projection, SwiGLU,
control and D64 vision checks, plus ten attention, RoPE and convolution checks.
Projection/SwiGLU coverage includes T=1, 2, 3, 4, 5, 7, 8, 9, 17, 64 and 65,
with full output checks for the hot small-token profiles and independent sampled
row reductions for larger extents. SwiGLU uses the existing A16 relative-L2 and
maximum-absolute-error criteria. All output elements are checked for finiteness.
Warmed single-request GPU-memory increments were about 2.02/2.02
GiB for baseline/optimized, estimated from WDDM total-GPU readings.

[Small-batch measurements](xiaomi-ocr-5070ti-small-batch.json)

## RTX 5070 Ti operator optimization

Measured on 2026-10-06 on RTX 5070 Ti 16 GB under Ubuntu WSL2, driver 617.14.
Both variants were built locally with CUDA 13.3.33 and GCC 13.3.0. The comparison
uses BF16 weights and KV, a 4K context, one request, a 1,024-token prefill chunk,
greedy decode and a 256-token output limit. Prefix and media reuse are disabled.
An A-B-B-A run uses two warmups and five measured requests per input in each
phase; results below are medians of ten measurements per variant.

The implementation changes are:

- D64 vision attention stages 64 features instead of padding them to 128.
  Its 64×64 tile uses 24 KiB of shared memory instead of 48 KiB, and QK uses
  four MMA iterations instead of five.
- Single-column BF16 residual projections write the sum directly, retaining
  BF16 projection rounding before the addition.
- Single-column GDN and attention projections write their final output planes
  directly. This removes the intermediate projection and split launches.

Decode CUDA Graphs remain enabled.

| Input | Baseline prefill tok/s | Optimized prefill tok/s | Prefill gain | Baseline decode tok/s | Optimized decode tok/s | Decode gain |
|---|---:|---:|---:|---:|---:|---:|
| zh_legal | 19,446 | 20,401 | 4.9% | 400.0 | 403.6 | 0.9% |
| en_contract | 19,666 | 20,446 | 4.0% | 401.5 | 407.9 | 1.6% |
| dense-equations | 18,207 | 19,487 | 7.0% | 394.9 | 402.0 | 1.8% |

Prefill includes vision encoding and language prefill. Vision encoding time
fell by 11.0–11.6%. Median request times were 455.7→457.8 ms, 459.1→444.4 ms,
and 864.3→843.3 ms, respectively. The gain is concentrated in prefill; decode
improvements are small. The original deployed binary was also rerun and gave
comparable baseline decode speeds of 394.7–404.9 tok/s in this session.

All 60 measured requests produced stable output. Baseline and optimized outputs
matched exactly on all three inputs: two complete text pages and the first
256 tokens of dense-equations. Character difference was 0%, and both versions
scored CER 0% on the 478 annotated characters.

All 40 independent FP64 checks passed. The 30 added checks cover the changed
projection routes, T=1/2/4, attention tile boundaries, packed segments, padded
strides, and the actual 7,168/9,216-patch vision workloads. Warmed GPU-memory
increments were approximately 2.02 GiB and 2.03 GiB under WDDM.

A separate 32K/four-request check submitted four complete pages together. The
optimized service's final interval averaged 3.50 active decode rows per round;
all four outputs matched both its sequential outputs and the baseline service.

[Optimization measurements](xiaomi-ocr-5070ti-optimization.json)

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
