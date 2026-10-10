# Qwen3 ForcedAligner

NInfer Extended supports the original [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)
checkpoint as a native C++/CUDA BF16 model. The public `EnginePurpose::ForcedAlignment` and
`Engine::align_features()` API execute the audio encoder, language model and timestamp classifier.
Python prepares audio features and word tokens on the CPU and applies the official timestamp parser.

The model aligns an existing transcript. It computes timestamps in a single forward pass, with
no autoregressive decode or KV cache. Supported languages are Chinese, English, Cantonese, French,
German, Italian, Japanese, Korean, Portuguese, Russian and Spanish.

## Convert and build

The ASR and alignment CPU tools share one Transformers 5.17.0 environment:

```bash
python -m pip install -r tools/qwen3_asr/requirements.txt
```

Keep the original model directory, including its processor and tokenizer files.

The ready-to-use BF16 artifact is available in the
[ASR model repository](https://huggingface.co/ByronLeeee/Qwen3-ASR-1.7B-hf-Ninfer), together with
the main ASR model and a complete transcription-to-timestamps example.

```bash
python tools/qwen3_forced_aligner/convert.py \
  --model /models/Qwen3-ForcedAligner-0.6B \
  --out /models/qwen3-forced-aligner-0.6b-bf16.ninfer
cmake --build build --target ninfer-align -j
```

Conversion preserves all 708 source BF16 tensors byte for byte. Q/K/V and gate/up matrices are
concatenated by rows; the timestamp head receives 120 zero rows for the existing 5,120-row linear
geometry. Argmax considers only the original 5,000 classes. The artifact embeds the original CPU
processor/tokenizer resources. No weight quantization is applied.

## Align audio

```bash
python tools/qwen3_forced_aligner/align.py \
  --artifact /models/qwen3-forced-aligner-0.6b-bf16.ninfer \
  --engine build/apps/ninfer-align \
  --audio recording-16k.wav \
  --text "The transcript of this recording." \
  --language English --out timestamps.json
```

Supply mono 16 kHz audio. Matching `--audio`, `--text` and `--language` lists accept one to eight
independent samples. Results include timestamp classes, parsed words, audio/language GPU time,
weight/runtime bytes and graph usage. Class resolution is the checkpoint's 80 ms.

The CLI defaults to native linear operators, compensated Tensor Core causal attention, audio and
language CUDA Graphs, an 8,192-token limit, and a maximum batch of four. `--batch` can select one
to eight. `--scalar-prefill` selects the FP32 probability attention path; `--eager` disables graphs;
`--backend cublas` selects the cuBLAS linear comparison path. `--warmups` defaults to zero and
`--repeats` to one. Benchmark warmups are explicit and are not extra production predictions.

For a resident engine, use `ninfer-align --artifact MODEL.ninfer --serve --batch 4`. After a
`{"ready":true,...}` line, each input line is `{"samples":[...]}`. Samples contain `frames`,
`features_path` (contiguous little-endian FP32 `[128,frames]`), `mask_path` (I32 valid-prefix mask),
and `prompt_ids`. Output lines contain `runs` or `error`. `{"close":true}` exits. Process lifetime
owns model weights, work buffers and captured graphs.

## Execution details

The qualified reference is the official `qwen-asr 0.0.6` BF16/SDPA path. Its legacy audio attention
uses the complete sample; NInfer preserves that range and isolates samples in a batch. Native
ASR's windowed audio attention remains unchanged. Very short samples retain the official
convolution boundary width. The alignment FFN and projection/residual stages preserve the
official BF16 intermediate rounding.

Both audio and language graphs are cached by input geometry. Timestamp classification gathers
only timestamp-token rows, rather than projecting every prompt row. Alignment does not allocate
the ASR decoder's per-layer K/V buffers. Validation covers RTX 5070 Ti and RTX 6000D, real Chinese
and English recordings, sub-second audio, a repeated 60-second clip, graph reuse, and independent
batch execution. Timestamp agreement measures equivalence to the reference, rather than labeled
alignment accuracy across the eleven languages.

The tested single-sample default matches all 400 timestamp boundaries on RTX 5070 Ti and 399 of
400 on RTX 6000D; the remaining difference is one 80 ms class. Changing batch geometry can also
move a classifier tie by one class. Repeated graphs and eager execution match exactly for the
tested batch geometry.
