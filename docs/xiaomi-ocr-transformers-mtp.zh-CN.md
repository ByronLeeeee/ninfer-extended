# Xiaomi-OCR-0：Transformers 与 NInfer 的 MTP 对照

[English](xiaomi-ocr-transformers-mtp.md)

双方均启用 **MTP1**，使用相同的 **131,072 行原始 BF16 草稿输出头**，
由主模型完整词表验证。NInfer 的 decode 在 5070 Ti 上达到 **2.49–2.68 倍**，
在 6000D 上达到 **2.41–2.47 倍**。

表中的 **TF MTP1 是 Transformers + 自定义 PyTorch MTP 适配**。
Transformers 5.17.0 不加载该模型的 MTP 模块，原生 `use_mtp=True` 会拒绝 Qwen3.5
的有状态生成。适配器加载模型原有的单层 MTP，使用上游 Qwen3.5 模块、SDPA、
融合 causal-conv1d/FLA；视觉、语言和 MTP 均 fullgraph 编译。
草稿被拒绝时，恢复卷积状态、FP32 循环状态及 KV 写入位置，再重放已接受的前缀。

BF16 权重与 KV，单路、4K 上下文、贪心解码，最多输出 256 token。
每项输入、每个引擎预热两次，再交替运行三轮 A-B-B-A/B-A-A-B，取六次正式请求的中位数。
Prefill 包含视觉、语言和 MTP 提示处理；NInfer 用 CUDA events 计时，PyTorch 用同步墙钟计时。
完整请求耗时包含预处理、推理和本地 HTTP 往返。NInfer chunk 在 5070 Ti 上为 1,024，
在 6000D 上为 2,048。

## 相同 128K 草稿输出头，双方 MTP1

| 显卡 | 输入 | TF prefill tok/s | NInfer prefill tok/s | TF decode tok/s | NInfer decode tok/s | Decode 倍数 | TF → NInfer prefill ms | TF → NInfer 请求 ms | 双方草稿接受率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | 中文文书 | 19,697 | 23,100 | 265.5 | 712.6 | 2.68× | 91.74 → 78.24 | 616.0 → 328.7 | 96.49% |
| RTX 5070 Ti | 英文合同 | 19,264 | 24,175 | 278.6 | 717.3 | 2.57× | 93.83 → 74.75 | 607.8 → 322.2 | 100.00% |
| RTX 5070 Ti | 密集公式（256 token） | 19,427 | 22,041 | 283.1 | 703.9 | 2.49× | 121.90 → 107.44 | 1130.8 → 566.5 | 99.22% |
| RTX 6000D | 中文文书 | 32,722 | 40,214 | 366.2 | 905.5 | 2.47× | 55.22 → 44.93 | 405.8 → 214.3 | 96.49% |
| RTX 6000D | 英文合同 | 32,657 | 40,065 | 380.0 | 914.1 | 2.41× | 55.33 → 45.10 | 393.2 → 212.9 | 100.00% |
| RTX 6000D | 密集公式（256 token） | 32,918 | 36,506 | 380.0 | 914.9 | 2.41× | 71.94 → 64.87 | 791.4 → 386.7 | 99.22% |

两张显卡的三项输出均逐字一致。中文、英文页自然结束，分别输出 113、114 token；
公式页比较相同的 256-token 前缀。两边草稿数、接受数、接受率也一致，三项接受率分别为
**96.49%、100.00%、99.22%**。

## 完整 BF16 草稿输出头，双方 MTP1

双方使用完整的原始 BF16 输出头生成草稿。三项输出均相同，草稿接受率均为 100%。

| 显卡 | 输入 | TF prefill tok/s | NInfer prefill tok/s | TF decode tok/s | NInfer decode tok/s | Decode 倍数 | TF → NInfer prefill ms | TF → NInfer 请求 ms | 双方草稿接受率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | 中文文书 | 19,996 | 23,695 | 280.4 | 662.8 | 2.36× | 90.37 → 76.26 | 586.7 → 332.8 | 100.00% |
| RTX 5070 Ti | 英文合同 | 19,432 | 23,917 | 285.6 | 657.9 | 2.30× | 93.00 → 75.56 | 583.5 → 339.6 | 100.00% |
| RTX 5070 Ti | 密集公式（256 token） | 18,975 | 22,111 | 287.7 | 646.0 | 2.25× | 124.81 → 107.11 | 1124.4 → 599.3 | 100.00% |
| RTX 6000D | 中文文书 | 32,197 | 39,937 | 365.4 | 854.4 | 2.34× | 56.12 → 45.25 | 405.6 → 222.4 | 100.00% |
| RTX 6000D | 英文合同 | 32,226 | 40,019 | 362.2 | 849.0 | 2.34× | 56.07 → 45.15 | 408.5 → 222.6 | 100.00% |
| RTX 6000D | 密集公式（256 token） | 32,762 | 36,405 | 364.5 | 843.6 | 2.31× | 72.28 → 65.05 | 821.8 → 411.6 | 100.00% |

## 双方关闭 MTP

TF 使用官方编译后的 `generate`，四个融合线性注意力函数均已启用。

| 显卡 | 输入 | TF prefill tok/s | NInfer prefill tok/s | TF decode tok/s | NInfer decode tok/s | Decode 倍数 | TF → NInfer prefill ms | TF → NInfer 请求 ms | 双方草稿接受率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | 中文文书 | 22,525 | 24,203 | 166.4 | 460.6 | 2.77× | 80.22 → 74.66 | 858.8 → 404.0 | — |
| RTX 5070 Ti | 英文合同 | 21,309 | 24,210 | 164.2 | 457.0 | 2.78× | 84.93 → 74.65 | 873.9 → 407.6 | — |
| RTX 5070 Ti | 密集公式（256 token） | 21,684 | 22,572 | 163.4 | 439.4 | 2.69× | 109.21 → 104.91 | 1796.5 → 802.5 | — |
| RTX 6000D | 中文文书 | 36,347 | 40,717 | 232.8 | 586.4 | 2.52× | 49.72 → 44.38 | 577.2 → 281.9 | — |
| RTX 6000D | 英文合同 | 36,441 | 40,710 | 232.9 | 586.8 | 2.52× | 49.59 → 44.39 | 580.3 → 281.5 | — |
| RTX 6000D | 密集公式（256 token） | 36,260 | 36,849 | 235.3 | 580.1 | 2.47× | 65.31 → 64.26 | 1202.4 → 547.8 | — |

## 同一 PyTorch 生成循环中的 MTP 收益

以下使用相同的编译版自定义贪心生成循环与完整 BF16 草稿输出头，单独观察 MTP 的作用。

| GPU | 输入 | 普通 tok/s | MTP1 tok/s | MTP1 变化 | MTP3 tok/s | MTP3 变化 | MTP3 接受率 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTX 5070 Ti | 中文文书 | 203.4 | 291.0 | +43.1% | 164.0 | -19.4% | 47.52% |
| RTX 5070 Ti | 英文合同 | 204.0 | 290.9 | +42.6% | 171.6 | -15.9% | 46.81% |
| RTX 5070 Ti | 密集公式（256 token） | 204.9 | 287.5 | +40.3% | 180.9 | -11.7% | 50.17% |
| RTX 6000D | 中文文书 | 254.6 | 365.5 | +43.6% | 205.3 | -19.4% | 47.52% |
| RTX 6000D | 英文合同 | 254.6 | 362.2 | +42.3% | 214.7 | -15.7% | 46.81% |
| RTX 6000D | 密集公式（256 token） | 254.8 | 364.7 | +43.2% | 225.2 | -11.6% | 50.17% |

两张卡的 MTP1、MTP3 均与普通解码的 token 序列一致，强制拒绝草稿后的状态恢复也通过。MTP3 连续使用模型原有的单层 MTP；在这个 PyTorch 实现中，较低的接受率与更多状态重放让 MTP3 慢于普通解码。

另一个“官方普通 Transformers → NInfer MTP1”对照约为 5070 Ti 的 4.3–4.4 倍、6000D 的 3.9 倍，包含引擎和 MTP 的共同收益；双方启用 MTP1 时使用本文第一张表。

## 复现

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

[实测数据](../model-cards/Xiaomi-OCR-0-BF16-NInfer/transformers-mtp-results.json)。原生支持限制见 [Transformers generation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/generation/utils.py) 与 [Qwen3.5 modeling](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)。
