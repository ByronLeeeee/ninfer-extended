# Xiaomi-OCR-0 · NInfer

[简体中文](README_zh-CN.md)

## Model overview

A BF16 NInfer version of [Xiaomi-OCR-0](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0), based on Qwen3.5-0.8B-Base, for OCR, document parsing, table extraction and formula recognition. It runs with **[NInfer Extended](https://github.com/ByronLeeeee/ninfer-extended)**, using native C++/CUDA kernels for vision encoding, prefill and decode.

The model file includes language, vision and MTP weights, the tokenizer and image-processing configuration. **MTP1 is recommended**; the same file also supports ordinary decoding and MTP3.

## Performance

### Prefill and decode

Warm-run medians, BF16 weights and KV, one request, 4K context and greedy decoding. Transformers uses compiled vision, prefill and decode, with CUDA Graph for decode. Prefill includes vision encoding and language processing. The formula sample generates 256 tokens.

| GPU | Input | Transformers prefill (tok/s) | NInfer MTP1 prefill (tok/s) | Transformers decode (tok/s) | NInfer MTP1 decode (tok/s) | Decode speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RTX 5070 Ti | Chinese document | 21,764 | 24,168 | 166.4 | **717.1** | 4.31× |
| RTX 5070 Ti | English contract | 22,066 | 23,383 | 163.3 | **718.6** | 4.40× |
| RTX 5070 Ti | Formulas (256 tokens) | 21,353 | 22,204 | 162.3 | **717.6** | 4.42× |
| RTX 6000D | Chinese document | 36,383 | 40,038 | 232.8 | **904.4** | 3.88× |
| RTX 6000D | English contract | 36,337 | 40,141 | 231.9 | **916.9** | 3.95× |
| RTX 6000D | Formulas (256 tokens) | 36,254 | 36,284 | 235.4 | **910.8** | 3.87× |

### Complete request latency

Includes image processing, inference and the local HTTP request.

| GPU | Input | Transformers (ms) | NInfer MTP1 (ms) | Speedup |
| --- | --- | ---: | ---: | ---: |
| RTX 5070 Ti | Chinese document | 879.0 | **333.5** | 2.64× |
| RTX 5070 Ti | English contract | 890.3 | **326.1** | 2.73× |
| RTX 5070 Ti | Formulas (256 tokens) | 1818.2 | **575.7** | 3.16× |
| RTX 6000D | Chinese document | 577.2 | **214.3** | 2.69× |
| RTX 6000D | English contract | 584.2 | **212.4** | 2.75× |
| RTX 6000D | Formulas (256 tokens) | 1202.7 | **388.8** | 3.09× |

### Transformers MTP1 comparison

The Transformers MTP1 reference uses a compiled PyTorch MTP adapter with the same proposal head as NInfer. See the [MTP comparison guide](https://github.com/ByronLeeeee/ninfer-extended/blob/main/docs/xiaomi-ocr-transformers-mtp.md) for its implementation.

| GPU | Input | Transformers MTP1 prefill (tok/s) | NInfer MTP1 prefill (tok/s) | Transformers MTP1 decode (tok/s) | NInfer MTP1 decode (tok/s) | Decode speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RTX 5070 Ti | Chinese document | 19,697 | 23,100 | 265.5 | **712.6** | 2.68× |
| RTX 5070 Ti | English contract | 19,264 | 24,175 | 278.6 | **717.3** | 2.57× |
| RTX 5070 Ti | Formulas (256 tokens) | 19,427 | 22,041 | 283.1 | **703.9** | 2.49× |
| RTX 6000D | Chinese document | 32,722 | 40,214 | 366.2 | **905.5** | 2.47× |
| RTX 6000D | English contract | 32,657 | 40,065 | 380.0 | **914.1** | 2.41× |
| RTX 6000D | Formulas (256 tokens) | 32,918 | 36,506 | 380.0 | **914.9** | 2.41× |

## Accuracy

The following results hold on both RTX 5070 Ti and RTX 6000D. CER is measured against human annotations; MTP output agreement and character differences are measured against ordinary NInfer decoding.

| Test | Ordinary BF16 | MTP1 | MTP3 |
| --- | ---: | ---: | ---: |
| 6 labelled text images, 1,117 characters: CER | 0% | 0% | 0% |
| 11 complete pages: exact output agreement | Reference | 100% (11/11) | 100% (11/11) |
| Complete pages, 7,269 characters: character difference | Reference | 0% | 0% |
| 36 mixed-batch requests: exact output agreement | Reference | 100% (36/36) | 100% (36/36) |
| Mixed batches, 27,708 characters: character difference | Reference | 0% | 0% |

The Chinese document, English contract and 256-token formula outputs used in the speed tests match Transformers exactly on both GPUs. They also match the Transformers MTP1 reference.

### MTP acceptance

Accepted draft tokens on complete pages; results are identical on both GPUs.

| Page | MTP1 | MTP3 |
| --- | ---: | ---: |
| Chinese document | 96.49% | 45.83% |
| English contract | 100.00% | 46.81% |
| Calligraphy | 78.57% | 26.19% |
| Formulas | 96.02% | 46.73% |
| Handwritten formulas | 98.61% | 47.60% |
| Receipt | 94.12% | 44.27% |

MTP1 improves all measured single-request and four-request workloads and is the recommended default.

## Usage

### Build

Requirements: Linux or WSL2, CUDA with `sm_120a` support, cuDNN 9 and cuBLAS. See the [build guide](https://github.com/ByronLeeeee/ninfer-extended#build) for dependencies.

```bash
git clone https://github.com/ByronLeeeee/ninfer-extended.git
cd ninfer-extended
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCUDNN_ROOT=/path/to/cudnn -DCUBLAS_ROOT=/path/to/cublas
cmake --build build --target ninfer-serve -j
```

### Download and start

```bash
pip install huggingface_hub
hf download ByronLeeee/Xiaomi-OCR-0-Ninfer \
  xiaomi-ocr-0-bf16-MTP.ninfer --local-dir models
./build/apps/ninfer-serve models/xiaomi-ocr-0-bf16-MTP.ninfer \
  --host 127.0.0.1 --port 8080 --model-id ocr --vision \
  --max-context 32768 --kv-capacity 131072 --max-concurrency 4 \
  --kv-dtype bf16 --prefill-chunk 1024 --no-thinking --greedy \
  --spec mtp --draft-tokens 1 --lm-head-draft
```

Use `--prefill-chunk 2048` on RTX 6000D. For MTP3, set `--draft-tokens 3`. For ordinary decoding, remove `--spec mtp --draft-tokens 1 --lm-head-draft`.

### Recognize an image

Save the input as `page.png` and run:

```python
import base64
import json
import urllib.request
from pathlib import Path

image = base64.b64encode(Path("page.png").read_bytes()).decode()
body = {
    "model": "ocr",
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}},
        {"type": "text", "text": "Extract the text in the image."}
    ]}],
    "temperature": 0,
    "max_tokens": 4096,
}
req = urllib.request.Request(
    "http://127.0.0.1:8080/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req) as response:
    print(json.load(response)["choices"][0]["message"]["content"])
```

Change the prompt for text, tables, formulas or full-page parsing. More examples are in the [OCR guide](https://github.com/ByronLeeeee/ninfer-extended/blob/main/docs/xiaomi-ocr.md).

## Model files

| File | Size | Precision |
| --- | ---: | --- |
| `xiaomi-ocr-0-bf16-MTP.ninfer` | 2.036 GB | BF16 |

License: Apache-2.0.

[Full results](gpu-results.json) · [Transformers MTP results](transformers-mtp-results.json) · [Artifact manifest](artifact-manifest.json)
