# Xiaomi OCR measurements

BF16 weights and KV, 4K capacity, one active request. Both engines receive the same images/prompts.
Prefill includes vision and language processing; decode counts generated tokens after the first.
Timings are synchronized wall time. Compilation/cold requests are excluded.

## Prefill compilation A-B-B-A

Compiled decode is enabled in both variants. Vision tensor work and language prefill are compiled;
grid metadata stays outside the graph. Six measured requests per variant/fixture, after phase warmups.

| GPU | Page | Eager prefill ms | Compiled prefill ms | Time reduction |
|---|---|---:|---:|---:|
| RTX 5070 Ti | zh_legal | 109.20 | 76.66 | 29.8% |
| RTX 5070 Ti | en_contract | 103.04 | 75.09 | 27.1% |
| RTX 5070 Ti | dense-equations | 129.41 | 107.43 | 17.0% |
| RTX 6000D | zh_legal | 49.45 | 48.83 | 1.3% |
| RTX 6000D | en_contract | 49.50 | 48.99 | 1.0% |
| RTX 6000D | dense-equations | 65.78 | 65.32 | 0.7% |

Both GPUs captured five graphs across these workloads (vision, language shapes, decode),
with zero recorded graph breaks. All 36 measured token sequences per GPU matched the eager reference.
The 6000D gain is small enough to treat as approximately unchanged; no universal speedup is claimed.

## Compiled reference versus clean fork build

The HTTP run uses two warmups (cache priming/compilation) and three measured repetitions.
The native binary was built from the complete published source in a fresh build directory.

| GPU | Page | HF prefill tok/s | Native prefill tok/s | HF decode tok/s | Native decode tok/s | HF / native request ms |
|---|---|---:|---:|---:|---:|---:|
| RTX 5070 Ti | zh_legal | 23060 | 21924 | 173.7 | 438.4 | 839.6 / 423.1 |
| RTX 5070 Ti | en_contract | 22646 | 21201 | 169.4 | 438.7 | 829.5 / 414.8 |
| RTX 5070 Ti | dense-equations | 22197 | 19859 | 171.1 | 427.1 | 1726.0 / 810.7 |
| RTX 6000D | zh_legal | 36193 | 35306 | 233.7 | 513.8 | 579.1 / 317.2 |
| RTX 6000D | en_contract | 36154 | 35140 | 232.2 | 514.4 | 584.7 / 316.1 |
| RTX 6000D | dense-equations | 35990 | 33031 | 235.1 | 511.0 | 1206.7 / 618.1 |

## Output and memory scope

Fresh HTTP comparisons match on all three outputs on both GPUs. The two text pages finish naturally;
dense-equations is a 256-token prefix. The fresh normalized character difference is 0%.
This is agreement between implementations, not evidence of 100% OCR accuracy.
The historical 11-image full-output test had one differing handwritten-formula page
(0.1933% weighted normalized character difference); it was not repeated under the 4K capacity.

| GPU | Reference memory | Native memory | Measurement |
|---|---:|---:|---|
| RTX 5070 Ti | 2.80 GiB | 2.02 GiB | WSL GPU total increment estimate |
| RTX 6000D | 3.44 GiB | 2.26 GiB | per-process nvidia-smi |

Server Qwen and ComfyUI/OCR services remained resident and idle during isolated tests.
Video/VSR had been paused for the local test. Local WSL memory is an estimate, not per-process attribution.
These numbers are the 4K/one-lane profile and must not be reused for 32K/four-lane deployment.

All ten FP64 attention/RoPE/convolution qualification cases passed on the clean build.
Functional compiler-op schema/fake-layout checks and initial-state ownership checks also passed.
Only cache storage initialization runs outside Dynamo; every measured request executes compiled language prefill.
Actual cache length is 4096 on both GPUs; execution counters verify all 15 HTTP generations.
Storage initialization preserves the static addresses
needed by compiled decode CUDA Graphs. Skipping that priming caused a local decode regression
and is fixed in the published helper.

[Sanitized measurement data](xiaomi-ocr-results.json).
