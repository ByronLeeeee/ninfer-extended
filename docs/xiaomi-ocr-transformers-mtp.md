# Xiaomi-OCR-0: matched Transformers and NInfer MTP comparison

[简体中文](xiaomi-ocr-transformers-mtp.zh-CN.md)

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

## MTP1 with the same 128K proposal head

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

## MTP1 with the full original BF16 head

Both engines draft with the complete original BF16 output head. All outputs match; draft acceptance is 100% on these inputs.

| GPU | Input | TF prefill tok/s | NInfer prefill tok/s | TF decode tok/s | NInfer decode tok/s | Decode speedup | TF → NInfer prefill ms | TF → NInfer request ms | Draft acceptance (both) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | Chinese document | 19,996 | 23,695 | 280.4 | 662.8 | 2.36× | 90.37 → 76.26 | 586.7 → 332.8 | 100.00% |
| RTX 5070 Ti | English contract | 19,432 | 23,917 | 285.6 | 657.9 | 2.30× | 93.00 → 75.56 | 583.5 → 339.6 | 100.00% |
| RTX 5070 Ti | Dense formulas (256 tokens) | 18,975 | 22,111 | 287.7 | 646.0 | 2.25× | 124.81 → 107.11 | 1124.4 → 599.3 | 100.00% |
| RTX 6000D | Chinese document | 32,197 | 39,937 | 365.4 | 854.4 | 2.34× | 56.12 → 45.25 | 405.6 → 222.4 | 100.00% |
| RTX 6000D | English contract | 32,226 | 40,019 | 362.2 | 849.0 | 2.34× | 56.07 → 45.15 | 408.5 → 222.6 | 100.00% |
| RTX 6000D | Dense formulas (256 tokens) | 32,762 | 36,405 | 364.5 | 843.6 | 2.31× | 72.28 → 65.05 | 821.8 → 411.6 | 100.00% |

## Both engines with MTP disabled

TF uses the official compiled `generate` path, with fused linear attention enabled.

| GPU | Input | TF prefill tok/s | NInfer prefill tok/s | TF decode tok/s | NInfer decode tok/s | Decode speedup | TF → NInfer prefill ms | TF → NInfer request ms | Draft acceptance (both) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | Chinese document | 22,525 | 24,203 | 166.4 | 460.6 | 2.77× | 80.22 → 74.66 | 858.8 → 404.0 | — |
| RTX 5070 Ti | English contract | 21,309 | 24,210 | 164.2 | 457.0 | 2.78× | 84.93 → 74.65 | 873.9 → 407.6 | — |
| RTX 5070 Ti | Dense formulas (256 tokens) | 21,684 | 22,572 | 163.4 | 439.4 | 2.69× | 109.21 → 104.91 | 1796.5 → 802.5 | — |
| RTX 6000D | Chinese document | 36,347 | 40,717 | 232.8 | 586.4 | 2.52× | 49.72 → 44.38 | 577.2 → 281.9 | — |
| RTX 6000D | English contract | 36,441 | 40,710 | 232.9 | 586.8 | 2.52× | 49.59 → 44.39 | 580.3 → 281.5 | — |
| RTX 6000D | Dense formulas (256 tokens) | 36,260 | 36,849 | 235.3 | 580.1 | 2.47× | 65.31 → 64.26 | 1202.4 → 547.8 | — |

## MTP gains within the same PyTorch loop

These use the same compiled custom greedy loop and full original BF16 proposal head. This isolates the MTP gain from changes to the generation loop.

| GPU | Input | Ordinary tok/s | MTP1 tok/s | MTP1 change | MTP3 tok/s | MTP3 change | MTP3 acceptance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | Chinese document | 203.4 | 291.0 | +43.1% | 164.0 | -19.4% | 47.52% |
| RTX 5070 Ti | English contract | 204.0 | 290.9 | +42.6% | 171.6 | -15.9% | 46.81% |
| RTX 5070 Ti | Dense formulas (256 tokens) | 204.9 | 287.5 | +40.3% | 180.9 | -11.7% | 50.17% |
| RTX 6000D | Chinese document | 254.6 | 365.5 | +43.6% | 205.3 | -19.4% | 47.52% |
| RTX 6000D | English contract | 254.6 | 362.2 | +42.3% | 214.7 | -15.7% | 46.81% |
| RTX 6000D | Dense formulas (256 tokens) | 254.8 | 364.7 | +43.2% | 225.2 | -11.6% | 50.17% |

MTP1 and MTP3 match ordinary token sequences and pass forced-rejection state recovery on both GPUs. MTP3 repeatedly applies the checkpoint's single trained MTP layer. Lower acceptance and more replay work make MTP3 slower than ordinary decoding in this PyTorch reference.

The separate native ordinary Transformers → NInfer MTP1 comparison reaches approximately 4.3–4.4× on 5070 Ti and 3.9× on 6000D; it includes both engine and MTP gains. The matched MTP1 table above compares both engines with MTP enabled.

## Reproduce

```bash
# Same compiled PyTorch loop: ordinary / MTP1 / MTP3, plus forced rejection.
python tools/xiaomi_ocr/benchmark_torch_mtp.py \
  --model-dir /path/to/Xiaomi-OCR-0 --fixtures-root /path/to/ocr-fixtures \
  --output-root results/torch-mtp --cycles 3

# Equal MTP1, exact proposal rows read directly from the .ninfer artifact.
python tools/xiaomi_ocr/benchmark_mtp.py \
  --transformers-baseline --custom-transformers-mtp \
  --hf-model /path/to/Xiaomi-OCR-0 --hf-python /path/to/python \
  --fixtures-root /path/to/ocr-fixtures \
  --artifact models/xiaomi-ocr-0-bf16-MTP.ninfer \
  --optimized-binary build/apps/ninfer-serve --output-root results \
  --lanes 1 --cycles 3 --draft-window 1 --lm-head-draft --label tf-mtp1-
```

Use `--optimized-chunk 2048` on 6000D. For the full proposal-head control,
omit `--lm-head-draft`. For native ordinary Transformers versus ordinary NInfer,
omit `--custom-transformers-mtp --lm-head-draft` and add `--compare-none`.
The fixture directory contains `manifest.json` entries with `id`, `path` and
`prompt`, plus the corresponding images. The PyTorch-only benchmark accepts an
optional `--official-summary` from the ordinary paired test to check its output
against official `generate`. Use Python 3.12, Torch 2.14.0+cu132,
Transformers 5.17.0, FLA/fla-core 0.5.2 and causal-conv1d 1.7.0 for the measured
reference. Build fused extensions for your Torch/CUDA runtime and `sm_120`;
the adapter requires all four fused linear-attention functions.

[Measurement data](../model-cards/Xiaomi-OCR-0-BF16-NInfer/transformers-mtp-results.json). Native support checks follow [Transformers generation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/generation/utils.py) and [Qwen3.5 modeling](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py).
