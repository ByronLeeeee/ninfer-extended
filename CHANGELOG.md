# Changelog

## Shared ASR and forced-alignment CPU frontend

Use native Transformers 5.17.0 for both speech frontends with one shared
requirements file. Remove the legacy `qwen-asr` wrapper dependency from the
alignment tool. Preserve the checkpoint's timestamp prompt and the official
Korean segmentation dictionary.

Validation retains byte-identical features, masks and prompt IDs for English,
Chinese, sub-second and 60-second inputs. All eleven language word lists and
106 timestamp-repair cases match the previous official frontend. A complete
ASR-to-alignment run and a mixed English/Chinese batch retain identical
timestamp classes and parsed words.

## Batched Qwen3-ASR language prefill on RTX 6000D

Pack actual prompt rows across two to four ASR lanes for shared projections,
normalization, FFNs and the first-token head. Keep per-sample FP32 probability
attention, local RoPE positions and independent KV. Allocate workspace for
the packed row count before Graph capture. Single-lane ASR and forced alignment
retain their existing execution paths.

BF16 weights/KV, 4096 context tokens per lane, greedy decoding to natural EOS.
Full prefill includes audio encoding, language prefill and the first token.
Results compare the previous NInfer Extended build with this change and
strict GPU BF16 compiled Transformers. NInfer medians pool ten warm runs per
input from an old/new/new/old order; Transformers uses three warm runs.

| Four-lane input | Previous prefill ms | Updated prefill ms | TF prefill ms | Speedup over previous |
|---|---:|---:|---:|---:|
| English 15.05 s × 4 | 40.211 | 29.374 | 34.437 | 1.37× |
| Chinese 4.20 s × 4 | 21.348 | 12.534 | 18.098 | 1.70× |
| English 30 s × 4 | 65.799 | 57.538 | 58.855 | 1.14× |
| Mixed English/Chinese × 4 | 27.963 | 18.693 | 50.569 | 1.50× |

Four-lane prefill time falls **12.6–41.3%**. Single-lane and decode speed remain
within measurement variation. Additional Engine runtime allocation is
**13.9–92.6 MiB** for the measured four-lane inputs; KV precision is unchanged.

Quality regression: all 73 recordings retain identical tokens and text between
previous and updated execution, both singly and in groups of up to four.
Single-lane WER stays 3.5652%; batched WER stays 3.6522%. Changed-input Graph
replays match fresh Engines; heterogeneous two/three-lane requests and
Graph/eager checks pass. Forced-aligner timestamps and runtime allocation
are identical. Web frontend automatic/forced language and prompt/hotword
checks pass. Temperatures are 52–72°C and SM clocks 2400–2422 MHz.

## BF16 partial RoPE and batched decode fusion on RTX 6000D

- Fuse Q/K RMSNorm and 64-dimension partial RoPE for 256-dimension heads,
  including one-axis and three-axis positions. Preserve the normalized BF16
  boundary. Six Full Attention layers remove 12 kernel launches per decode step.
- Tune 2–8-token BF16 projections with 1024 output rows and 2048/3584 input
  columns while preserving the previous accumulation order.
- Extend BF16 GDN projection/convolution/state-snapshot fusion to 2–8 requests
  with one token each. Preserve per-request state and BF16 projection rounding;
  this route needs no workspace and removes another 18 launches per batch step.

Nsight Systems confirms **222 → 210 kernels** per single-request decode replay
and **240 → 210 kernels** per four-request replay. The following measurements
use unprofiled public HTTP inference and compare against the preceding NInfer
Extended build, which already included the two decode optimizations below.

Xiaomi-OCR-0 on RTX 6000D, BF16 weights/KV, 32768-token context capacity per
request, 2048-token prefill chunks, greedy decoding, and no prefix/media cache.
Short Chinese/English pages contain 1807 input tokens; the dense-equation page
contains 2368, and the handwritten page 8232. A 256-token output cap is used
for short-output measurements; complete-page measurements use a 4096-token cap
and every request reaches natural completion.

Single-request short-output medians, 12 balanced paired cycles after warmup:

| Input | Previous decode tok/s | Updated decode tok/s | Full prefill ms, previous → updated | HTTP latency ms, previous → updated |
|---|---:|---:|---:|---:|
| Chinese document | 579.47 | 584.86 | 44.670 → 44.523 | 279.58 → 277.53 |
| English contract | 579.49 | 584.96 | 44.506 → 44.576 | 279.68 → 277.42 |
| Dense equations, capped output | 575.30 | 581.20 | 64.547 → 64.529 | 547.37 → 543.29 |

Paired single-request decode gains are **0.89–1.09%**; their bootstrap 95%
intervals are positive. Single-request end-to-end throughput improves about
0.6–0.8%. Full vision/language prefill has no confirmed change.

Four-request short-page tests pool all 60 paired cycles, including the initial
12 cycles with negative/noisy results. Each version processes 480 formal
requests per input. Paired gains take the median of within-cycle speed ratios;
the absolute rates separately summarize all requests. These differ when GPU
clocks drift. Bootstrap resamples whole paired cycles.

| Input | Decode tok/s per request, previous → updated | Paired decode gain, 95% interval | Actual total tok/s, previous → updated | Paired total-throughput gain, 95% interval |
|---|---:|---:|---:|---:|
| Chinese document | 454.22 → 470.08 | +5.05%, [4.13%, 5.28%] | 903.32 → 911.90 | +2.60%, [2.08%, 2.76%] |
| English contract | 452.40 → 473.09 | +5.14%, [4.34%, 5.42%] | 904.07 → 933.19 | +2.53%, [0.95%, 2.69%] |

Actual total throughput is all output tokens divided by HTTP batch completion
time, including preprocessing, serial prefill and decode. Temperatures and SM
clocks vary naturally across this pooled test: 61–86°C and 930–2422 MHz.
No cooling, power or clock settings were changed; all formal samples are retained.

Complete-page four-request medians, eight balanced paired cycles:

| Input | Decode tok/s per request, previous → updated | Full prefill ms/request, previous → updated | Batch completion s, previous → updated | Actual total tok/s, previous → updated |
|---|---:|---:|---:|---:|
| Dense equations | 414.21 → 424.94 | 74.662 → 74.771 | 5.302 → 5.126 | 1524.34 → 1576.59 |
| Handwritten formulas | 374.98 → 387.48 | 726.130 → 725.117 | 6.042 → 5.824 | 756.10 → 784.30 |

Both complete-page decode gains have positive paired intervals. Dense-equation
total throughput improves 3.43% by the aggregate medians, with a positive paired
interval. The handwritten-page total-throughput median improves 3.73%, but its
paired interval crosses zero, so a stable throughput gain is not established.

All 11 single-request complete pages match exactly. Four-request output variants
and their occurrence counts match on all 11 pages; dense and handwritten formulas
retain the two variants already present in the preceding build. Six annotated
text fixtures retain CER 0% over 1117 unique reference characters, using NFKC,
whitespace removal and Markdown heading/emphasis cleanup. Full raw output
comparisons use no normalization. The quality suite contains 220 formal requests.

Independent FP64 reference, guard, read-only operand, state-transition and
changed-input CUDA Graph tests pass for the affected public Ops. ASR token IDs,
embedding vectors and forced-alignment timestamp classes match exactly. Follow-up
tests retain the initial measurements: pooled embedding B4×512 and 60-second
alignment changes are -0.25% and -0.55%, with 95% intervals crossing zero.
Short-token embedding improves about 1.0–1.2% with positive paired intervals.
The deployed OCR service passes complete four-request and streaming-output checks.

## BF16 GDN decode fusion on RTX 6000D

The BF16 [8192,1024] single-column GDN input projection now writes convolution,
SiLU and history snapshots directly from its GEMV epilogue. It preserves the
previous projection reduction order and BF16 rounding, removes the intermediate
projected plane and eliminates 18 separate convolution launches per OCR token.
The 16-head, 1024-input norm/control projection also distributes one-column work
across two head groups. Multi-column routes retain their existing schedules.

Xiaomi-OCR-0, BF16 weights/KV, 4K context, 2,048-token prefill chunks and greedy
decoding; seven alternating A-B-B-A/B-A-A-B cycles, two warmups and 14 measured
requests per engine/input, with a 256-token output cap:

| Input | Previous decode tok/s | Updated decode tok/s | Change |
|---|---:|---:|---:|
| Chinese document | 564.1 | 579.7 | +2.8% |
| English contract | 563.2 | 579.3 | +2.9% |
| Dense equations | 560.2 | 574.9 | +2.6% |

Every paired cycle improved. End-to-end throughput increased 1.8–2.2%; full
vision/language prefill stayed within -0.07% to +0.14%. Four-request decode
remained within +0.04% to +0.08%. These figures compare with the preceding
NInfer Extended build, which already included the narrow-projection optimization.

A separate same-session comparison against the build before both decode
iterations measured 540.4 → 579.5, 540.1 → 579.6 and 537.2 → 575.7 tok/s for
these three inputs, respectively: +7.2–7.3% cumulatively. The GPU remained at
2,422 MHz throughout the seven paired cycles.

All 11 complete pages matched exactly across 44 formal requests at 16K context
and a 4,096-token cap. The six annotated text fixtures retained CER 0% over
1,117 reference characters. Public GDN snapshot and norm/control Ops passed
independent FP64 reference tests, state/guard checks and CUDA Graph replay
with changed inputs and selectors. The added 1024-wide norm test uses the BF16
rounding bound; its inherited tighter threshold rejected the same valid
rounding case in both the preceding and updated kernels.

After controlling starting GPU temperature, ASR, embedding and forced-alignment
outputs matched exactly; measured stage/throughput changes stayed within
-0.14% to +0.11%. The installed OCR service passed complete four-request and
streaming-output checks with its existing configuration.

FFN schedule variants and vocabulary Tensor Core variants did not demonstrate
useful gains and were not selected. Measurements affected by temperature-related
clock variation were excluded from speed claims.

## BF16 narrow decode projections on RTX 6000D

Single-token BF16 projections with 1,024 output rows and 2,048 or 3,584 input
columns now launch 256 row CTAs instead of 32. The schedule preserves the
previous per-row accumulation order; weights and the BF16 rounding boundaries
are unchanged. Vocabulary projections and multi-token routes retain their
existing schedules.

Xiaomi-OCR-0 on RTX 6000D, BF16 weights/KV, 4K context, 2,048-token prefill
chunks, greedy decoding, seven alternating A-B-B-A/B-A-A-B cycles and 14 measured
requests per engine/input after warmup:

| Input | Previous decode tok/s | Updated decode tok/s | Change |
|---|---:|---:|---:|
| Chinese document | 540.8 | 564.5 | +4.4% |
| English contract | 541.1 | 563.8 | +4.2% |
| Dense equations | 538.2 | 559.4 | +3.9% |

Single-request end-to-end throughput improved 2.8–3.2%; full vision/language
prefill remained within -0.2% to +0.3%. Four-request decode remained within
+0.05% to +0.30%. These comparisons are against the preceding NInfer Extended
build, using a 256-token output cap for speed tests.

Separate complete-page checks used a 16K context and a 4,096-token cap: all 11
pages finished normally and matched exactly across 44 formal requests. The six
annotated text fixtures retained CER 0% over 1,117 reference characters. Public
Linear and LinearAdd routes passed independent numerical-oracle tests, including
CUDA Graph replay with changed inputs.

ASR token IDs, embedding vectors and forced-alignment timestamp classes matched
the preceding build. Initial long-input embedding timing regressions did not
reproduce with longer warmup; no embedding speedup is claimed. The deployed OCR
service passed complete four-request and streaming-output checks.

## Fused vision projection and resource-aware prefill

New BF16 `linear_bias_add` and `linear_bias_gelu` Ops fuse vision projection,
bias and residual/GELU epilogues. They preserve the original projection and
bias-add BF16 rounding seams and remove separate intermediate reads and writes.
Registered geometries cover residual projections to 768 channels and GELU
projections to 3,072 channels, with exact and tanh GELU supported.

Gated DeltaNet state passing uses narrow strips when the grid fits the device's
SM count. BF16 SwiGLU uses three-stage staging for partial prompt tails when wider
tiles underfill the GPU. Dispatch uses tensor geometry and physical resources.
The 6000D OCR deployment uses 2,048-token chunks; 5070 Ti retains 1,024.
Weights and KV remain BF16, with decode CUDA Graphs enabled.

| GPU | Input | Requests | Previous prefill tok/s | Updated prefill tok/s | Prefill change | Previous decode tok/s | Updated decode tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| RTX 5070 Ti | Chinese document | 1 | 23,171 | 23,623 | +1.95% | 450.5 | 452.7 |
| RTX 5070 Ti | English contract | 1 | 23,888 | 24,256 | +1.54% | 451.7 | 451.8 |
| RTX 5070 Ti | Dense formulas (256 tokens) | 1 | 22,572 | 22,968 | +1.76% | 447.9 | 448.1 |
| RTX 5070 Ti | Chinese document | 4 | 22,585 | 22,613 | +0.13% | 373.4 | 374.1 |
| RTX 5070 Ti | English contract | 4 | 22,177 | 22,386 | +0.94% | 372.9 | 373.0 |
| RTX 6000D | Chinese document | 1 | 37,871 | 40,491 | +6.92% | 540.6 | 541.3 |
| RTX 6000D | English contract | 1 | 37,775 | 40,649 | +7.61% | 541.1 | 542.6 |
| RTX 6000D | Dense formulas (256 tokens) | 1 | 35,551 | 36,622 | +3.01% | 537.4 | 537.4 |
| RTX 6000D | Chinese document | 4 | 38,066 | 40,564 | +6.56% | 489.5 | 488.4 |
| RTX 6000D | English contract | 4 | 37,950 | 40,563 | +6.89% | 488.9 | 488.2 |

Both GPUs return **11/11 identical complete pages in single-request tests** compared with the preceding
Extended OCR build, with **0% normalized character difference**. Six annotated
text images contain 1,117 scored characters and retain CER 0%. Timed OCR controls
also match compiled Transformers exactly on all three inputs.

ASR, embedding and forced-alignment controls preserve every tested token ID,
embedding vector and timestamp class. Numerical qualification includes FP64
recurrence output and final state, resource-based schedule parity, chunk seams,
BF16 SwiGLU boundaries, fused bias epilogues and preserved inputs. The new fusion
tests use an independent FP64 oracle, retain the two observable BF16 rounding
seams, leave the final ideal output unrounded and also check exact staged parity
and CUDA Graph replay. Long normalized recurrence fixtures
retain the 0.41% relative-L2 budget; the shared gross-output cap accounts for two
BF16 unit roundoffs. Alternative attention scheduling and prefill FFN graphs were
evaluated and excluded from this update. Whole-language prefill graphs were also
excluded because their incremental benefit was below the acceptance threshold.

Paired measurements cover single and four-request OCR, with decode CUDA Graphs
retained. Embedding and alignment are checked one fixture at a time with five
alternating A-B-B-A/B-A-A-B cycles. Server controls use the preceding source
rebuilt with the same compiler.

Additional mixed-concurrency checks compare full Chinese, English and formula
pages at 32K context. Fused and pre-fusion builds with the same 2,048-token chunk
produce identical outputs on both runs. The preceding build also changes the
handwritten formula's `P`/`p` casing between single and mixed decode batches;
single-request agreement is reported separately from concurrency controls.

[Current Transformers comparison](model-cards/Xiaomi-OCR-0-BF16-NInfer/README.md)
· [Detailed results](docs/xiaomi-ocr-performance.md).


## BF16 prefill operator optimization

Register-fused BF16 gate/up + SwiGLU removes **14 MiB** of projection scratch
per 1,024-token chunk. Geometry-based MMA and D64 attention tuning raise full
vision/language prefill throughput on **RTX 5070 Ti** by **3.56–3.94%** for one request and
**2.05–4.77%** for four requests compared with the preceding Extended build.

| GPU | Input | Requests | Previous prefill tok/s | Updated prefill tok/s | Prefill change | Previous decode tok/s | Updated decode tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| RTX 5070 Ti | Chinese document | 1 | 22,118 | 22,905 | +3.56% | 421.4 | 421.0 |
| RTX 5070 Ti | English contract | 1 | 22,131 | 22,947 | +3.69% | 421.3 | 432.3 |
| RTX 5070 Ti | Dense formulas (256 tokens) | 1 | 20,290 | 21,089 | +3.94% | 426.9 | 426.4 |
| RTX 5070 Ti | Chinese document | 4 | 22,398 | 22,857 | +2.05% | 391.0 | 384.7 |
| RTX 5070 Ti | English contract | 4 | 21,922 | 22,966 | +4.77% | 381.4 | 377.0 |
| RTX 6000D | Chinese document | 1 | 37,779 | 37,832 | +0.14% | 540.2 | 539.9 |
| RTX 6000D | English contract | 1 | 37,909 | 37,862 | -0.13% | 540.0 | 541.2 |
| RTX 6000D | Dense formulas (256 tokens) | 1 | 35,337 | 35,506 | +0.48% | 537.2 | 537.7 |
| RTX 6000D | Chinese document | 4 | 38,216 | 38,051 | -0.43% | 488.6 | 488.1 |
| RTX 6000D | English contract | 4 | 37,857 | 38,158 | +0.80% | 489.3 | 489.2 |

On RTX 6000D, full prefill changes by **−0.43% to +0.80%** and decode by **−0.10% to +0.22%**, with no material whole-model speed change. Both GPUs pass the BF16 projection, SwiGLU and D64 attention numerical checks.

All 304 timed outputs on 5070 Ti and 220 on 6000D match the preceding build. In an 11-page complete-output check on each GPU, 10 pages match exactly and character difference is 0.1795%; the changed
formula page restores a transpose present in the source image. Six annotated
text images score CER 0%. In the 5070 Ti regression control, ASR full GPU prefill stays within 0.5% and decode
within 0.12%; embedding GPU time stays within 0.3%, with identical tested
transcripts and vectors.

Against the paired compiled Transformers control, the updated 5070 Ti build
delivers **0.8–12.9% more full prefill throughput**, **2.83–2.99× decode speed**
and **2.25–2.41× end-to-end request speed**. The updated 6000D build delivers **−1.3% to +4.1% full prefill change**, **2.29–2.34× decode speed** and **1.95–2.07× end-to-end request speed** against its paired compiled Transformers control. [Full results](docs/xiaomi-ocr-performance.md).

## Earlier native model support and operator updates

| Workload | Unmodified upstream NInfer | NInfer Extended |
|---|---|---|
| Xiaomi-OCR-0 / Qwen3.5-0.8B | Missing the required small-model BF16/vision routes | Native vision, language prefill and decode on both tested GPUs |
| Qwen3-ASR-1.7B | No ASR architecture/frontend | Native audio encoding, language prefill and decode on both tested GPUs |
| Qwen3-Embedding-0.6B | No Qwen3 embedding architecture/frontend | Native BF16 vector inference on RTX 5070 Ti and RTX 6000D |
| Existing Qwen3.8-27B artifact | Supported | Existing large-model routes retained; 6000D control below |

### Xiaomi OCR: initial BF16 adaptation vs optimized operators

The baseline is the **initial working BF16 OCR adaptation before operator optimization**, using the same artifact and matched toolchain. Measurements use the complete Chinese document, 4K context, BF16 KV, 1,024-token prefill chunks and greedy decoding. Prefill includes vision and language GPU time; decode is tok/s per request.

| GPU | Requests | Baseline prefill tok/s | Extended prefill tok/s | Prefill change | Baseline decode tok/s | Extended decode tok/s | Decode change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | 1 | 20,016 | 22,082 | +10.3% | 406.4 | 418.6 | +3.0% |
| RTX 5070 Ti | 2 | 19,573 | 21,602 | +10.4% | 282.6 | 403.2 | +42.7% |
| RTX 5070 Ti | 4 | 19,480 | 21,538 | +10.6% | 268.9 | 370.8 | +37.9% |
| RTX 6000D | 1 | 35,238 | 37,911 | +7.6% | 514.2 | 541.9 | +5.4% |
| RTX 6000D | 2 | 35,254 | 37,896 | +7.5% | 356.3 | 527.3 | +48.0% |
| RTX 6000D | 4 | 35,349 | 37,876 | +7.1% | 341.2 | 487.1 | +42.7% |

This earlier two-GPU comparison retains exact outputs, passes 225 independent FP64 OCR checks per GPU, and completes 4K/32K one-/two-/four-request regression checks. [Detailed OCR measurements](docs/xiaomi-ocr-performance.md).

### Qwen3.8-27B: existing NInfer server vs extension on RTX 6000D

The existing pre-rollout NInfer executable and extended executable read the same mixed NVFP4/FP8/BF16/integer artifact, FP8 KV, 16K context and DFlash2 K7. The copy workload produces 128 tokens; its draft acceptance is 100%.

| Requests | Prompt tokens | Existing prefill tok/s | Extended prefill tok/s | Existing decode tok/s | Extended decode tok/s |
| --- | --- | --- | --- | --- | --- |
| 1 | 2504 | 6,480 | 6,493 | 352.5 | 352.6 |
| 1 | 10256 | 6,415 | 6,416 | 347.2 | 346.9 |
| 4 | 2504 | 6,498 | 6,505 | 295.5 | 295.5 |
| 4 | 10256 | 6,440 | 6,437 | 283.0 | 282.8 |

Changes stay within ±0.21%, with all 120 formal outputs matching. Open-ended single-request writing reaches 116–131 decode tok/s at 26.1% draft acceptance. These operator extensions do not establish a 27B speed gain. This 27B control was measured on 6000D; no 5070 Ti 27B result is claimed.

### Qwen3 Embedding: optimizations after native BF16 support

The native embedding route uses packed causal attention and register-fused SwiGLU. A 4×128 forward needs 228 kernel launches, including 28 attention invocations and no separate SwiGLU activation; the initial supported route needed 340 launches and 112 attention invocations. These are measurements of the supported extension, since upstream has no embedding frontend.

| GPU | Input | Earlier supported NInfer ms | Optimized NInfer ms | Whole-model throughput change |
|---|---|---:|---:|---:|
| RTX 5070 Ti | 4×33 tokens | 3.563 | 3.316 | +7.45% |
| RTX 5070 Ti | 8×17 tokens | 3.489 | 3.236 | +7.82% |
| RTX 6000D | 1×2048 tokens | 23.000 | 21.298 | +7.99% |

Short packed batches reduce unused projection tile work. Long-input attention can stage 64 K/V columns while retaining the original 32-column softmax and compensated value accumulation. Resource-based dispatch selects the wider schedule on 6000D for the measured 2K input; 5070 Ti retains the original narrow GPU instructions. The wider kernel is compiled in a separate non-RDC CUDA module. It adds no global scratch allocation.

Both GPUs pass 164 embedding and 117 ASR FP64 checks; all 249 tested embedding vectors match the preceding version exactly, and the tested ASR outputs remain token-identical. [Embedding guide](docs/qwen3-embedding.md).

The model repositories report **Transformers vs NInfer Extended** speed and result agreement. OCR and ASR include complete prefill/decode; the ASR cards also report warm latency and inference time/throughput for 60-second audio. Embedding reports complete forward/pooling throughput, vector similarity, retrieval ranking agreement and semantic scores.
