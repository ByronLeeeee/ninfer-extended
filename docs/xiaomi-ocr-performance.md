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

### Current compiled Transformers comparison

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

### Compiled Transformers versus BF16 MTP1



BF16 weights and KV, one request, 4K context, identical images/prompts, greedy
decoding, two warmups and six measured requests per engine/input in alternating
A-B-B-A/B-A-A-B cycles. Output is capped at 256 tokens: both text pages complete;
the dense-formula row measures the same 256-token prefix. Results are medians.

Transformers vision, language prefill and decode are compiled, with fused
causal-conv1d and Gated DeltaNet. Runtime checks confirm fused backends and
compiled language/decode calls. Prefill includes vision encoding and language
prefill; Transformers uses synchronized wall timing and NInfer CUDA events.
Decode counts tokens after the first token. NInfer uses 1,024-token chunks on
5070 Ti and 2,048 on 6000D.

| GPU | Input | TF prefill tok/s | MTP1 prefill tok/s | Prefill change | TF decode tok/s | MTP1 decode tok/s | Decode speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | Chinese document | 21,238 | 23,553 | +10.9% | 162.7 | 707.5 | 4.35× |
| RTX 5070 Ti | English contract | 21,003 | 23,812 | +13.4% | 155.2 | 728.1 | 4.69× |
| RTX 5070 Ti | Dense formulas (256 tokens) | 21,937 | 21,350 | -2.7% | 159.6 | 748.5 | 4.69× |
| RTX 6000D | Chinese document | 36,299 | 40,061 | +10.4% | 233.8 | 872.6 | 3.73× |
| RTX 6000D | English contract | 36,311 | 40,212 | +10.7% | 233.4 | 926.6 | 3.97× |
| RTX 6000D | Dense formulas (256 tokens) | 36,366 | 36,450 | +0.2% | 235.6 | 950.4 | 4.03× |

Prefill change is `(MTP1 / TF − 1) × 100%`; decode speedup is `MTP1 / TF`.

### Complete request latency

Includes preprocessing, inference and local HTTP overhead:

| GPU | Input | TF request ms | MTP1 request ms | Speedup |
| --- | --- | --- | --- | --- |
| RTX 5070 Ti | Chinese document | 866.2 | 322.6 | 2.68× |
| RTX 5070 Ti | English contract | 963.2 | 331.0 | 2.91× |
| RTX 5070 Ti | Dense formulas (256 tokens) | 1,823.8 | 566.5 | 3.22× |
| RTX 6000D | Chinese document | 576.0 | 219.2 | 2.63× |
| RTX 6000D | English contract | 576.5 | 210.7 | 2.74× |
| RTX 6000D | Dense formulas (256 tokens) | 1,197.8 | 377.0 | 3.18× |

## Ordinary BF16, MTP1 and MTP3

The ordinary/MTP pairs use the same NInfer Extended binary and artifact. Each
MTP window has its own paired reference run. Acceptance is accepted draft tokens
divided by proposed draft tokens; target-model tokens are excluded.

| GPU | Input | Ordinary decode tok/s | MTP1 decode tok/s | MTP1 acceptance | MTP3 decode tok/s | MTP3 acceptance |
| --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | Chinese document | 435.0 | 662.5 | 80.6% | 674.0 | 39.7% |
| RTX 5070 Ti | English contract | 434.5 | 701.1 | 93.2% | 680.8 | 40.5% |
| RTX 5070 Ti | Dense formulas (256 tokens) | 432.1 | 721.0 | 97.7% | 761.6 | 49.5% |
| RTX 6000D | Chinese document | 583.3 | 872.8 | 80.6% | 838.3 | 39.7% |
| RTX 6000D | English contract | 582.6 | 923.4 | 93.2% | 863.7 | 40.5% |
| RTX 6000D | Dense formulas (256 tokens) | 580.2 | 956.5 | 97.7% | 960.0 | 49.5% |

MTP3 acceptance by draft position, as a share of verification rounds. A later
position also requires the earlier proposals to pass:

| GPU | Input | First position | Second position | Third position |
| --- | --- | --- | --- | --- |
| RTX 5070 Ti | Chinese document | 82.7% | 32.7% | 3.8% |
| RTX 5070 Ti | English contract | 90.2% | 25.5% | 5.9% |
| RTX 5070 Ti | Dense formulas (256 tokens) | 97.1% | 37.9% | 12.6% |
| RTX 6000D | Chinese document | 82.7% | 32.7% | 3.8% |
| RTX 6000D | English contract | 90.2% | 25.5% | 5.9% |
| RTX 6000D | Dense formulas (256 tokens) | 97.1% | 37.9% | 12.6% |

### Direct MTP1/MTP3 comparison

Both windows are compared directly in six measured requests per input and engine:

| GPU | Input | MTP1 tok/s | MTP3 tok/s | MTP3 decode change | MTP3 complete-request throughput change |
| --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | Chinese document | 719.1 | 705.2 | -1.9% | -0.0% |
| RTX 5070 Ti | English contract | 759.6 | 723.1 | -4.8% | -2.3% |
| RTX 5070 Ti | Dense formulas (256 tokens) | 780.9 | 803.1 | +2.8% | +2.2% |
| RTX 6000D | Chinese document | 873.1 | 839.5 | -3.9% | -3.2% |
| RTX 6000D | English contract | 927.7 | 863.8 | -6.9% | -4.2% |
| RTX 6000D | Dense formulas (256 tokens) | 955.0 | 958.7 | +0.4% | +0.5% |

MTP1 performs better across the measured text pages and four-lane throughput.
MTP3 gives an additional single-request gain on the dense-formula prefix; on a
complete classical-Chinese example, 6000D MTP3 acceptance is 18.8% and complete
request throughput is 1.6% below ordinary decoding. The full-vocabulary target
continues to verify every proposal.

### Four-lane complete-request throughput

Total output tokens divided by the wall time of all four requests, including
prefill and HTTP overhead. These values are total throughput across four lanes.

| GPU | Input × 4 | Ordinary output tok/s | MTP1 output tok/s | MTP1 change | MTP3 output tok/s | MTP3 change |
| --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | Chinese document | 620.6 | 703.0 | +13.3% | 674.5 | +9.2% |
| RTX 5070 Ti | English contract | 599.9 | 698.1 | +16.4% | 660.1 | +6.7% |
| RTX 5070 Ti | Dense formulas (256 tokens) | 817.7 | 988.5 | +20.9% | 957.6 | +17.2% |
| RTX 6000D | Chinese document | 1,007.4 | 1,167.8 | +15.9% | 1,103.9 | +9.6% |
| RTX 6000D | English contract | 1,016.7 | 1,210.3 | +19.0% | 1,118.2 | +9.9% |
| RTX 6000D | Dense formulas (256 tokens) | 1,249.2 | 1,592.9 | +27.5% | 1,510.6 | +20.9% |

## Accuracy and output agreement

In the three Transformers/MTP1 speed inputs, raw outputs match **3/3 (100%)** on
both GPUs. The two complete annotated text pages contain 478 scoring characters;
both engines have **100% character accuracy, CER 0%**.

The full-document test compares MTP with ordinary NInfer Extended: eleven complete
pages, 16K context, 4,096-token output cap, 44 formal requests per window/GPU.
Every request ends naturally. Six annotated text fixtures contain 1,117 scoring
characters; the other pages measure output agreement.

| GPU | Mode | Annotated text accuracy | Complete-page raw agreement | Character difference |
| --- | --- | --- | --- | --- |
| RTX 5070 Ti | MTP1 | 100% (CER 0%) | 10/11 (90.9%) | 0.0414% |
| RTX 5070 Ti | MTP3 | 100% (CER 0%) | 10/11 (90.9%) | 0.0414% |
| RTX 6000D | MTP1 | 100% (CER 0%) | 10/11 (90.9%) | 0.0414% |
| RTX 6000D | MTP3 | 100% (CER 0%) | 10/11 (90.9%) | 0.0414% |

The three single-lane differences are uppercase/lowercase `P/p` in a handwritten
formula. Scoring uses NFKC and removes whitespace; text CER also removes Markdown
headings/emphasis and preserves case.

Mixed document batches use 32K context, changing groups of four, two and three
requests, nine unique pages and 18 formal outputs per mode/GPU:

| GPU | Mode | Mixed-batch raw agreement | Character difference |
| --- | --- | --- | --- |
| RTX 5070 Ti | MTP1 | 16/18 (88.9%) | 0.1081% |
| RTX 5070 Ti | MTP3 | 18/18 (100.0%) | 0.0000% |
| RTX 6000D | MTP1 | 15/18 (83.3%) | 0.2163% |
| RTX 6000D | MTP3 | 18/18 (100.0%) | 0.0000% |

Mixed MTP1 differences occur in formula letter case and LaTeX superscripts/primes.
For fixed single-lane execution, **11/11 complete MTP1 outputs match eager and
CUDA Graph execution on both GPUs**. GDN accepted-prefix recovery, MTP packing,
and the BF16 proposal projection pass independent numerical and changed-input
Graph checks.

## GPU memory

Separate backend processes, after two warmup requests, 4K context per lane:

| GPU | Lanes | Ordinary MiB | MTP1 MiB | MTP3 MiB | MTP1 extra MiB |
| --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | 1 | 2071 | 2249 | 2249 | 178 |
| RTX 5070 Ti | 4 | 2275 | 2483 | 2491 | 208 |
| RTX 6000D | 1 | 2317 | 2497 | 2497 | 180 |
| RTX 6000D | 4 | 2523 | 2731 | 2739 | 208 |

Values are the increase over measured system idle residency. The 5070 Ti uses
WDDM/WSL, so these are GPU-wide memory deltas. All original 564 parameter bindings
retain their values. The 65,536 proposal rows are exact BF16 copies of the
original output head, selected with upstream token-frequency ranking.

## Artifact and license

`xiaomi-ocr-0-bf16.ninfer` · **1,901,499,904 bytes** · BF16 · Apache-2.0

SHA256: `c0d0e1ec96e45661657fb9ac31753a46dac5545f35a4ebc993345c5f0e9e2472`

[Measurement data](results/xiaomi-ocr-mtp-bf16.json) · [Artifact manifest](https://huggingface.co/ByronLeeee/Xiaomi-OCR-0-Ninfer/resolve/main/artifact-manifest.json)

### BF16 indexed head optimization

The paired comparison keeps the same MTP window and compares the full BF16 proposal head with the 65,536-row BF16 indexed head.

| GPU | Window | Input | Full-head decode tok/s | Indexed-head decode tok/s | Decode change | Complete-request throughput change |
| --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | MTP1 | Chinese document | 689.2 | 719.5 | +4.4% | +2.0% |
| RTX 5070 Ti | MTP1 | English contract | 678.3 | 756.5 | +11.5% | +5.7% |
| RTX 5070 Ti | MTP1 | Dense formulas (256 tokens) | 676.8 | 777.9 | +14.9% | +10.2% |
| RTX 5070 Ti | MTP3 | Chinese document | 536.5 | 697.8 | +30.1% | +16.3% |
| RTX 5070 Ti | MTP3 | English contract | 564.6 | 721.6 | +27.8% | +12.2% |
| RTX 5070 Ti | MTP3 | Dense formulas (256 tokens) | 582.6 | 770.0 | +32.2% | +17.0% |
| RTX 6000D | MTP1 | Chinese document | 851.1 | 873.3 | +2.6% | +2.4% |
| RTX 6000D | MTP1 | English contract | 843.6 | 921.3 | +9.2% | +6.2% |
| RTX 6000D | MTP1 | Dense formulas (256 tokens) | 843.2 | 950.7 | +12.7% | +9.1% |
| RTX 6000D | MTP3 | Chinese document | 686.2 | 838.2 | +22.1% | +13.6% |
| RTX 6000D | MTP3 | English contract | 707.0 | 863.8 | +22.2% | +13.7% |
| RTX 6000D | MTP3 | Dense formulas (256 tokens) | 732.6 | 958.0 | +30.8% | +21.8% |

### Ordinary decoding regression

Compare the preceding accepted NInfer Extended binary with this build, using the same new BF16 artifact and ordinary decoding.

| GPU | Input | Decode change | Prefill change | Complete-request throughput change | Output exact |
| --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | Chinese document | +2.1% | +1.9% | -0.1% | True |
| RTX 5070 Ti | English contract | +4.2% | -0.8% | +2.7% | True |
| RTX 5070 Ti | Dense formulas (256 tokens) | +3.4% | -0.8% | +0.7% | True |
| RTX 6000D | Chinese document | +0.6% | +0.4% | +0.2% | True |
| RTX 6000D | English contract | +0.1% | +0.1% | +0.2% | True |
| RTX 6000D | Dense formulas (256 tokens) | +0.3% | -0.2% | -0.0% | True |
