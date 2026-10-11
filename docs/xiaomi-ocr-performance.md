# Xiaomi-OCR-0 test results

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

[Current Transformers comparison](../model-cards/Xiaomi-OCR-0-BF16-NInfer/README.md)
· [Detailed results](xiaomi-ocr-performance.md).

### Earlier compiled Transformers comparison

| GPU | Input | TF prefill tok/s | NInfer prefill tok/s | Prefill change | TF decode tok/s | NInfer decode tok/s | Decode speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| RTX 5070 Ti | Chinese document | 20,983 | 24,125 | +15.0% | 148.6 | 438.6 | 2.95× |
| RTX 5070 Ti | English contract | 21,402 | 23,830 | +11.3% | 143.6 | 439.5 | 3.06× |
| RTX 5070 Ti | Dense formulas (256 tokens) | 20,971 | 22,710 | +8.3% | 147.7 | 435.8 | 2.95× |
| RTX 6000D | Chinese document | 36,263 | 40,404 | +11.4% | 232.2 | 540.5 | 2.33× |
| RTX 6000D | English contract | 36,247 | 40,541 | +11.8% | 231.7 | 540.6 | 2.33× |
| RTX 6000D | Dense formulas (256 tokens) | 36,174 | 36,842 | +1.8% | 234.0 | 537.4 | 2.30× |

| GPU | Input | TF request ms | NInfer request ms | Speedup |
|---|---|---:|---:|---:|
| RTX 5070 Ti | Chinese document | 956.7 | 412.8 | 2.32× |
| RTX 5070 Ti | English contract | 980.5 | 432.7 | 2.27× |
| RTX 5070 Ti | Dense formulas (256 tokens) | 1981.8 | 791.3 | 2.50× |
| RTX 6000D | Chinese document | 582.2 | 295.1 | 1.97× |
| RTX 6000D | English contract | 583.5 | 295.6 | 1.97× |
| RTX 6000D | Dense formulas (256 tokens) | 1208.1 | 581.1 | 2.08× |


## Earlier BF16 prefill operator update

The update adds register-fused `[7168,1024]` gate/up and SwiGLU, tunes the
`[1024,3584]` down projection, and uses 128-key tiles for D64 attention with
single or equal-length segments of at least 6,656 tokens. Dispatch uses tensor
geometry. The fused SwiGLU preserves BF16 gate/up and SiLU rounding and removes
**14 MiB of projection scratch** per 1,024-token chunk. Weights and KV remain BF16.

### Previous Extended build vs updated operators

Both builds use 4K context per request, 1,024-token prefill chunks, greedy
decoding and a 256-token output cap, with prefix/image caching disabled.
Results are medians. Full prefill includes vision and language GPU time;
decode is tok/s per request after the first token. Decode CUDA Graphs stay enabled.

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

Single-request full prefill falls from **81.700 to 78.890 ms** for Chinese,
**81.650 to 78.746 ms** for English, and **116.720 to 112.283 ms** for formulas.
Decode changes by -0.11% to +2.62% for one request and -1.60% to -1.15% for four.
All **304 measured speed responses** on 5070 Ti and **220** on 6000D match the preceding build exactly. On 6000D, full prefill changes by **−0.43% to +0.80%** and decode by **−0.10% to +0.22%**; the language-stage gain is largely offset by vision-stage timing, leaving complete prefill effectively unchanged.

### Full-output quality and regression checks

With 16K context and a 4,096-token output budget, all 11 images finish naturally.
**10/11 pages match the preceding Extended build exactly on each GPU**, with **0.1795%**
weighted normalized character difference. On the long formula page, the update
restores the outer transpose in equation (8) shown in the image; the other changes
are equivalent LaTeX transpose notation in equation (7). Six annotated text images
totaling 1,117 normalized characters score **100% character accuracy (CER 0%)**.

BF16 LinearSwiGLU, LinearAdd and D64 packed attention pass independent FP64
qualification, including complete outputs, input preservation, segment isolation,
dispatch boundaries and CUDA Graph replay. ASR GPU prefill stays within **0.5%**
and decode within **0.12%** of the preceding build on 5070 Ti; transcripts are token-identical.
Embedding GPU time stays within **0.3%** for 1×128, 1×2048 and 4×512 inputs,
with identical vectors. The default prefill chunk remains 1,024 tokens.

### Compiled Transformers vs NInfer Extended

The paired comparison uses BF16 weights/KV, 4K context, one request, greedy
decoding and a 256-token output cap. Transformers uses compiled vision/language
prefill and decode, verified fused causal-conv1d/Gated DeltaNet, and a 4K static
cache. Prefill includes vision encoding and language prefill; Transformers stages
use synchronized wall time and NInfer stages use CUDA events.

| GPU | Input | TF prefill tok/s | NInfer prefill tok/s | Prefill change | TF decode tok/s | NInfer decode tok/s | Decode speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| RTX 5070 Ti | Chinese document | 19,991 | 22,240 | +11.3% | 140.7 | 413.3 | 2.94× |
| RTX 5070 Ti | English contract | 19,793 | 22,338 | +12.9% | 138.6 | 414.7 | 2.99× |
| RTX 5070 Ti | Dense formulas (256 tokens) | 20,560 | 20,734 | +0.8% | 145.0 | 410.7 | 2.83× |
| RTX 6000D | Chinese document | 36,426 | 37,921 | +4.1% | 231.9 | 542.4 | 2.34× |
| RTX 6000D | English contract | 36,439 | 37,664 | +3.4% | 231.8 | 540.8 | 2.33× |
| RTX 6000D | Dense formulas (256 tokens) | 36,072 | 35,615 | -1.3% | 234.7 | 538.3 | 2.29× |

Both GPUs use the updated operators and a fresh paired compiled Transformers control.

End-to-end request time includes preprocessing, inference and the local HTTP round trip:

| GPU | Input | Transformers request time | NInfer request time | Speedup |
|---|---|---:|---:|---:|
| RTX 5070 Ti | Chinese document | 1006.0 ms | 437.4 ms | 2.30× |
| RTX 5070 Ti | English contract | 1004.9 ms | 446.0 ms | 2.25× |
| RTX 5070 Ti | Dense formulas (256 tokens) | 2009.8 ms | 834.7 ms | 2.41× |
| RTX 6000D | Chinese document | 582.0 ms | 298.0 ms | 1.95× |
| RTX 6000D | English contract | 582.7 ms | 298.8 ms | 1.95× |
| RTX 6000D | Dense formulas (256 tokens) | 1203.5 ms | 581.2 ms | 2.07× |

All 60 measured responses on each GPU match Transformers exactly. Both complete text pages score CER 0% on 478 annotated characters; formulas compare the same 256-token prefix.

[Transformers comparison data](../model-cards/Xiaomi-OCR-0-BF16-NInfer/gpu-results.json)

## Earlier operator gains on RTX 6000D


The baseline is the initial working BF16 OCR adaptation. Both versions read the same BF16 model. Speeds are tok/s; decode is
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

## Language-only context benchmark on RTX 6000D


This earlier executable was tested with pretokenized repetitions of OCR output
text, BF16 weights/KV, one request, a 262,144-token capacity and a 1,024-token
prefill chunk. The output uses 256 fixed decode steps. These numbers exclude vision encoding and tokenization.

| Prompt tokens | Language prefill ms | Language prefill tok/s | Decode tok/s |
|---:|---:|---:|---:|
| 2,048 | 21.42 | 95,630 | 513.6 |
| 8,192 | 95.40 | 85,866 | 491.2 |
| 32,768 | 556.06 | 58,929 | 440.7 |
| 65,536 | 1580.05 | 41,477 | 356.6 |
| 131,072 | 5020.59 | 26,107 | 280.0 |
| 260,000 | 17883.40 | 14,539 | 192.6 |

This series uses an earlier validated build and measures language processing only.

## GPU memory


These measurements use a warmed 4K context and one active request. These results are from the earlier validated build.

| GPU | Transformers | NInfer | Measurement scope |
|---|---:|---:|---|
| RTX 5070 Ti | 2.80 GiB | 2.02 GiB | WSL total-GPU increment estimate |
| RTX 6000D | 3.44 GiB | 2.26 GiB | Per-process nvidia-smi |

## Additional operator measurements

Earlier iterations added compact D64 vision attention, fused residual/output
projections, small-batch BF16 kernels and fewer split-KV partitions. The measured
compact decode-attention routes reduced public-Op latency by **5.8–21.3%** and
temporary workspace by approximately **50%**. Selected prefill projection routes
reduced public-Op latency by **3–15%**. Full-model gains are reported separately above.

- [Two-GPU operator results and latest prefill update](xiaomi-ocr-gpu-results.json)
- [BF16 prefill projection measurements](xiaomi-ocr-5070ti-prefill-projection.json)
- [Grouped-query attention measurements](xiaomi-ocr-5070ti-causal-attention.json)
- [Small-batch measurements](xiaomi-ocr-5070ti-small-batch.json)
- [Initial operator measurements](xiaomi-ocr-5070ti-optimization.json)


## BF16 MTP1 and MTP3

## Full vision/language prefill and decode performance

Compiled BF16 Transformers versus BF16 NInfer MTP1. Both engines use the same
images, greedy decoding, one request, 4K context and a 256-token output cap.
Prefill includes vision encoding and language prefill. Decode is measured after
the first token; CUDA Graphs remain enabled. The reference requires fused
causal-conv1d/Gated DeltaNet and compiled vision, language prefill and decode.
NInfer uses CUDA-event stage timing; Transformers uses synchronized wall timing.
Medians follow two warmups and three alternating A-B-B-A/B-A-A-B cycles.

| GPU | Input | TF prefill tok/s | MTP1 prefill tok/s | Prefill change | TF decode tok/s | MTP1 decode tok/s | Decode speedup |
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
- One file supports ordinary decoding, MTP1 and MTP3. The non-MTP BF16 artifact
  remains available separately.
- License: Apache-2.0, following [Xiaomi-OCR-0](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0).
