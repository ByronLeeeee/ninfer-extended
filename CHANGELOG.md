# Changelog

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
