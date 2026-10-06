# Xiaomi-OCR-0 test results

## RTX 5070 Ti grouped-query decode attention

Measured on 2026-10-06 on RTX 5070 Ti 16 GB under Ubuntu WSL2, driver 617.14,
CUDA 13.3.33 and GCC 13.3.0. The baseline is `6e97374a`, which includes the
normalized-control fusion below. Weights and KV are BF16 in both versions.

The public causal-attention Op selects a compact split profile for 256-dimensional
heads, 8 query heads and 2 KV heads, at query width 1–4 and execution envelopes up
to 8,192 visible keys. It uses two warps per partial CTA, halves the split scale,
and merges output in 128-dimensional tiles. Selection uses geometry, dtype, query
width, batch size and the execution envelope. The capacity query follows the same
profile, reducing transient attention workspace by about 50% on these routes.
Single-row, single-column calls with envelopes up to 2,048 keys keep the original
profile: the initial complete public-Op measurement found that profile faster.
Wider queries and larger envelopes use the existing profiles.

Eight complete kernel candidates passed 72 independent FP64 checks before timing.
The selected public route then passed qualification independently. The public-Op
comparison calls both complete public-Op implementations in one CUDA process,
using the same input and KV addresses. Baseline host symbols are renamed to keep
the two archives distinct; their device computation is unchanged. Both versions
pass 32 FP64 checks in this process before timing.
Timings include cache append, attention and split-output reduction,
captured in CUDA Graphs with at least 64 calls per graph and a rotating KV pool of
at least 128 MiB. Each version warms for at least 200 ms of GPU execution at each
point, then runs six A-B-B-A cycles targeting 20 ms per sample. The table uses the
median of all twelve samples per version.

| Visible keys | Rows | Baseline attention µs | Updated attention µs | Latency reduction |
|---:|---:|---:|---:|---:|
| 1,807 | 2 | 24.07 | 19.69 | 18.2% |
| 1,807 | 4 | 39.00 | 33.42 | 14.3% |
| 4,096 | 1 | 27.79 | 21.88 | 21.3% |
| 4,096 | 2 | 45.59 | 37.21 | 18.4% |
| 4,096 | 4 | 70.38 | 60.25 | 14.4% |
| 8,192 | 1 | 39.07 | 34.08 | 12.8% |
| 8,192 | 2 | 63.93 | 60.25 | 5.8% |
| 8,192 | 4 | 114.56 | 102.78 | 10.3% |

Whole-model measurements use 4K context per request, a 1,024-token prefill chunk,
decode CUDA Graphs, and a 256-token output cap. Prefix reuse and media caching are
disabled. Both versions explicitly use temperature 0, presence/frequency penalties
0, top-p 1, top-k 0, min-p 0 and seed 123.
Both real servers stay resident; only one engine's burst executes at a time.
Each input has two warmups and five short A-B-B-A cycles, giving ten formal bursts
per input and version. Tables show medians. Prefill includes vision and language
GPU processing; concurrent decode is per request. Total throughput includes
client wall time, prefill and scheduling.

| Lanes | Input | Baseline prefill tok/s | Updated prefill tok/s | Baseline decode tok/s/request | Updated decode tok/s/request | Decode change |
|---:|---|---:|---:|---:|---:|---:|
| 1 | zh_legal | 22,491 | 22,307 | 430.6 | 431.3 | +0.2% |
| 1 | en_contract | 22,286 | 22,419 | 429.5 | 430.4 | +0.2% |
| 1 | dense-equations | 20,674 | 20,729 | 421.0 | 428.8 | +1.8% |
| 2 | zh_legal | 22,651 | 22,084 | 416.8 | 420.2 | +0.8% |
| 2 | en_contract | 22,345 | 22,242 | 419.6 | 419.8 | +0.0% |
| 4 | zh_legal | 21,821 | 21,841 | 375.0 | 378.8 | +1.0% |
| 4 | en_contract | 21,749 | 21,345 | 375.4 | 379.3 | +1.0% |

| Lanes | Input | Baseline total tok/s | Updated total tok/s | Total change |
|---:|---|---:|---:|---:|
| 2 | zh_legal | 418.1 | 418.4 | +0.1% |
| 2 | en_contract | 422.2 | 426.3 | +1.0% |
| 4 | zh_legal | 573.0 | 573.8 | +0.1% |
| 4 | en_contract | 572.0 | 581.8 | +1.7% |

The attention latency reduction is **5.8–21.3%** on the measured compact
routes. Whole-model changes are much smaller; the paired results above give the
prefill, decode and total throughput changes separately.

All **300 formal OCR outputs** match their baseline exactly, with 100% agreement
and 0% normalized character difference. Both versions score CER 0% on the two
complete text pages with 478 annotated characters. The dense-equations comparison
uses the first 256 generated tokens. A separate 32K/four-request check completes
Chinese text, English text, mixed numbers and a formula page; every output matches
sequential execution and baseline.

All **197 independent FP64 checks** pass: 165 retained checks and 32 new causal
attention checks. New coverage includes reordered physical pages and table rows,
masked tails and empty rows, exact cache writes, read-only cached attention,
query widths 1–16, batch sizes up to 8, and 2K/8K/32K route boundaries. The FP64
oracle uses represented BF16 inputs and the BF16-key/FP16-value cache boundary;
the existing NRMS < 0.006 and peak-error/reference-RMS < 0.045 criteria are retained.

[Grouped-query attention measurements](xiaomi-ocr-5070ti-causal-attention.json)

## RTX 5070 Ti fused normalization and GDN controls

Measured on 2026-10-06 on RTX 5070 Ti 16 GB under Ubuntu WSL2, driver 617.14,
CUDA 13.3.33 and GCC 13.3.0. The baseline is `bddacdc3`, which includes the
small-batch operators below. Both versions use BF16 weights and KV, a 4K context
per request, a 1,024-token prefill chunk and a 256-token output cap. Prefix reuse
and media caching are disabled; decode CUDA Graphs are enabled. The paired runs
use temperature 0, presence penalty 1.5 and the remaining model sampling defaults.

The public `gdn_norm_gating_proj` Op now fuses offset RMSNorm and GDN control
projections for the 1,024-input/16-head profile at T=1–8. A 512-thread CTA owns
one token. The first 256 threads retain the existing pair-wise norm reduction;
all 16 warps then project the shared BF16 normalized values into controls.
The explicit normalized output and gate/beta results match the composed Ops
bit for bit on the qualification inputs. Larger positive T uses composition.
The stored weight layout and the zero-scratch contract are preserved.

Matched CUDA Graph microbenchmarks with a 128 MiB rotating control-weight pool
reduce the combined norm/control segment from about 4.2 µs to 3.0 µs at T=1–8.
Public-Op qualification and complete OCR measurements follow candidate selection.
The initial long-phase A-B-B-A run showed about 10% baseline decode variation,
so the tables use a shorter paired comparison. Both real servers stay resident,
and requests alternate in five A-B-B-A cycles, after two warmups per input and
engine. Only one engine's burst runs at a time. All ten formal bursts per input
and variant are included; tables show medians.
Prefill includes vision encoding and language processing, excluding CPU image
preprocessing. Concurrent decode is per request. Total speed divides all
completion tokens by client burst time, including prefill and scheduling.

| Input | Baseline prefill tok/s | Optimized prefill tok/s | Baseline decode tok/s | Optimized decode tok/s | Decode change |
|---|---:|---:|---:|---:|---:|
| zh_legal | 22,012 | 22,199 | 427.0 | 428.7 | +0.4% |
| en_contract | 21,828 | 21,540 | 415.9 | 420.2 | +1.0% |
| dense-equations | 19,959 | 20,014 | 408.6 | 412.8 | +1.0% |

| Lanes | Input | Baseline decode tok/s/request | Optimized decode tok/s/request | Decode change | Baseline total tok/s | Optimized total tok/s | Total change |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | zh_legal | 400.7 | 404.2 | +0.9% | 401.3 | 403.6 | +0.6% |
| 2 | en_contract | 400.6 | 404.1 | +0.9% | 406.9 | 408.1 | +0.3% |
| 4 | zh_legal | 373.0 | 377.9 | +1.3% | 574.1 | 586.0 | +2.1% |
| 4 | en_contract | 372.5 | 375.3 | +0.7% | 580.1 | 578.0 | -0.4% |

The paired full-model decode changes are **+0.4–1.3%**, much smaller than the
operator-segment gain. Prefill and end-to-end changes are small and mixed.

All **600 formal outputs** across the original and paired comparisons match
their baseline under the same settings: 100% agreement and 0% normalized
character difference. The original single-request comparison uses presence
penalty 0; its formula prefix differs from the presence-penalty-1.5 runs, with
exact baseline/optimized agreement in each configuration.
Both versions score CER 0% on two complete text pages with 478 annotated
characters; the formula page compares its first 256 output tokens.
A separate 32K/four-request test completes Chinese text, English text,
mixed numbers and a formula page, all matching sequential execution and baseline.

All **165 independent FP64 checks** pass. The normalized-control coverage
includes T=1, 2, 3, 4, 5, 7, 8, 9, 17 and 65, with complete output checks and
bitwise comparison against the composed public Ops. Existing numerical criteria
are retained. Warmed single-request GPU-memory increments are about
2.02/2.02 GiB for baseline/optimized,
estimated from WDDM total-GPU readings.

[Normalized-control measurements](xiaomi-ocr-5070ti-norm-gating.json)

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
