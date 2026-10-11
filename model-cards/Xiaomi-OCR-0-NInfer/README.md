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



Native BF16 OCR and document parsing for **[NInfer Extended](https://github.com/ByronLeeeee/ninfer-extended)**, based on
Xiaomi-OCR-0 / Qwen3.5-0.8B-Base. The artifact contains the language and vision
weights, the original MTP layer, a 65,536-row BF16 proposal head, tokenizer and
image-processing resources. Main weights, proposal weights and KV remain BF16.

## Build and run

Use **[ninfer-extended](https://github.com/ByronLeeeee/ninfer-extended) main**. MTP1 is the recommended configuration:

```bash
git clone https://github.com/ByronLeeeee/ninfer-extended.git
cd ninfer-extended
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCUDNN_ROOT=/path/to/cudnn -DCUBLAS_ROOT=/path/to/cublas
cmake --build build --target ninfer-serve -j
hf download ByronLeeee/Xiaomi-OCR-0-Ninfer xiaomi-ocr-0-bf16.ninfer --local-dir models
./build/apps/ninfer-serve models/xiaomi-ocr-0-bf16.ninfer \
  --host 127.0.0.1 --port 8080 --model-id ocr --vision \
  --max-context 32768 --kv-capacity 131072 --max-concurrency 4 \
  --kv-dtype bf16 --prefill-chunk 1024 --no-thinking --greedy \
  --spec mtp --draft-tokens 1 --lm-head-draft
```

Use `--prefill-chunk 2048` on RTX 6000D. Change `--draft-tokens 1` to
`--draft-tokens 3` to try MTP3. Both use the checkpoint's single trained MTP layer;
MTP3 applies it repeatedly. Remove `--spec mtp --draft-tokens 1 --lm-head-draft`
for ordinary BF16 decoding. Omit `--lm-head-draft` to draft with the full BF16 head.

Build requirements: 64-bit Linux (WSL2 on Windows), CUDA with `sm_120a`, C++20,
CMake ≥3.28, Ninja, FFmpeg development libraries, libcurl, PCRE2, cuDNN 9 and
cuBLAS. [Conversion and usage guide](https://github.com/ByronLeeeee/ninfer-extended/blob/main/docs/xiaomi-ocr.md).

## Compiled Transformers versus BF16 MTP1

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

[Measurement data](../../docs/results/xiaomi-ocr-mtp-bf16.json) · [Artifact manifest](https://huggingface.co/ByronLeeee/Xiaomi-OCR-0-Ninfer/resolve/main/artifact-manifest.json)
