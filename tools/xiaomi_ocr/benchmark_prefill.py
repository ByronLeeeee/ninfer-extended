"""Bounded text-prefill compilation A/B; does not alter service defaults."""
import os
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
import argparse
import functools
import json
import statistics
import time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, CompileConfig

B = Path(__file__).resolve().parents[2]
R = B / 'out/xiaomi-prefill'
p = argparse.ArgumentParser()
p.add_argument('--mode', default='default', choices=['default', 'reduce-overhead'])
p.add_argument('--fixtures', nargs='+', default=['zh_legal', 'dense-equations'])
p.add_argument('--allow-graph-breaks', action='store_true')
p.add_argument('--functional-ops', action='store_true')
p.add_argument('--vision', action='store_true')
p.add_argument('--model-dir', default=str(B / 'models/Xiaomi-OCR-0'))
p.add_argument('--results-dir', default=str(R))
p.add_argument('--abba', action='store_true')
p.add_argument('--fixtures-dir', type=Path, required=True)
args = p.parse_args()
R = Path(args.results_dir)
R.mkdir(parents=True, exist_ok=True)
if args.functional_ops:
    import sys
    sys.path.insert(0, str(B))
    from compiled_prefill import install_prefill_boundaries
    install_prefill_boundaries()
torch.set_num_threads(1)
if args.allow_graph_breaks:
    torch._dynamo.config.cache_size_limit = 64
model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).to('cuda').eval()
processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
language = model.model.language_model
base_language = language.forward
active_language = base_language
compiled_language = torch.compile(base_language, mode=args.mode, fullgraph=not args.allow_graph_breaks, dynamic=False)

@functools.wraps(base_language)
def language_forward(*a, **kw):
    # Keep the existing official compiled decode graph unchanged. Only the
    # ordinary prefill invocation selects the separately compiled language graph.
    if torch.compiler.is_compiling():
        return base_language(*a, **kw)
    return active_language(*a, **kw)

language.forward = language_forward
if args.vision:
    from compiled_prefill import compile_vision
    base_vision, compiled_vision = compile_vision(model.model.visual, args.mode)
base_forward = model.forward
forward_times, vision_times = [], []

@functools.wraps(base_forward)
def timed_forward(*a, **kw):
    if torch.compiler.is_compiling():
        return base_forward(*a, **kw)
    begin = time.perf_counter()
    out = base_forward(*a, **kw)
    if not forward_times:
        torch.cuda.synchronize()
    forward_times.append((begin, time.perf_counter()))
    return out

model.forward = timed_forward
def vision_begin(module, a):
    vision_times.append([time.perf_counter(), None])
def vision_end(module, a, out):
    torch.cuda.synchronize()
    vision_times[-1][1] = time.perf_counter()
model.model.visual.register_forward_pre_hook(vision_begin)
model.model.visual.register_forward_hook(vision_end)

report = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'mode': args.mode, 'fullgraph': not args.allow_graph_breaks, 'dynamic': False, 'capacity': 4096, 'max_output_tokens': 256, 'scope': 'Fullgraph language prefill and vision tensor work; grid metadata eager; official compiled decode in both variants; no service defaults changed' if args.vision else 'Fullgraph language prefill; vision eager; official compiled decode in both variants', 'abba': args.abba, 'runs': [], 'summaries': [], 'errors': []}
manifest = json.loads((args.fixtures_dir / 'manifest.json').read_text())
fixtures = [next(f for f in manifest if f['id'] == name) for name in args.fixtures]
path = R / ('prefill-compile-pilot-' + args.mode + ('-partial' if args.allow_graph_breaks else '') + ('-functional' if args.functional_ops else '') + ('-vision' if args.vision else '') + '.json')
def save():
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

reference = {}
for phase, variant in enumerate((['eager_prefill', 'compiled_text_prefill', 'compiled_text_prefill', 'eager_prefill'] if args.abba else ['eager_prefill', 'compiled_text_prefill'])):
    active_language = base_language if variant == 'eager_prefill' else compiled_language
    if args.vision:
        model.model.visual.forward = base_vision if variant == 'eager_prefill' else compiled_vision
    for fixture in fixtures:
        image = Image.open(args.fixtures_dir / Path(fixture['path']).name).convert('RGB')
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': fixture['prompt']}]}]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt', enable_thinking=False).to('cuda')
        assert inputs['input_ids'].shape[1] + 256 <= 4096
        for repetition in range(4):
            forward_times.clear()
            vision_times.clear()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin = time.perf_counter()
            print(f'{variant} {fixture["id"]} run {repetition}', flush=True)
            try:
                with torch.inference_mode():
                    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False, use_cache=True, eos_token_id=248046, pad_token_id=248044, cache_implementation='static', max_cache_len=4096, compile_config=CompileConfig(mode='reduce-overhead', fullgraph=True))
                torch.cuda.synchronize()
            except Exception as error:
                report['errors'].append({'variant': variant, 'fixture': fixture['id'], 'type': type(error).__name__, 'message': str(error)[:12000]})
                save()
                print(json.dumps(report['errors'][-1], ensure_ascii=False), flush=True)
                raise
            end = time.perf_counter()
            ids = outputs[0, inputs['input_ids'].shape[1]:].tolist()
            if variant == 'eager_prefill':
                reference.setdefault(fixture['id'], ids)
            prefill_ms = (forward_times[0][1] - forward_times[0][0]) * 1000
            vision_ms = sum((b - a) * 1000 for a, b in vision_times)
            decode_ms = (end - forward_times[0][1]) * 1000
            row = {'phase': phase, 'variant': variant, 'fixture': fixture['id'], 'repeat': repetition, 'warmup': repetition == 0, 'prompt_tokens': int(inputs['input_ids'].shape[1]), 'completion_tokens': len(ids), 'prefill_ms': prefill_ms, 'vision_ms': vision_ms, 'text_prefill_ms': prefill_ms - vision_ms, 'decode_ms': decode_ms, 'decode_tps': (len(ids)-1)/(decode_ms/1000), 'generation_ms': (end-begin)*1000, 'exact_token_ids_to_eager': ids == reference[fixture['id']], 'cuda_reserved_mib': torch.cuda.memory_reserved()/2**20, 'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20, 'output_ids': ids, 'text': processor.decode(ids, skip_special_tokens=True)}
            report['runs'].append(row)
            report['dynamo_counters'] = {key: dict(value) for key, value in torch._dynamo.utils.counters.items() if key in ['stats', 'graph_break', 'frames']}
            save()
            print(json.dumps({k: v for k, v in row.items() if k not in ['output_ids', 'text']}, ensure_ascii=False), flush=True)
        samples = [row for row in report['runs'] if row['variant'] == variant and row['fixture'] == fixture['id'] and not row['warmup']]
        report['summaries'].append({'variant': variant, 'fixture': fixture['id'], **{key: statistics.median(row[key] for row in samples) for key in ['prefill_ms', 'vision_ms', 'text_prefill_ms', 'decode_tps', 'generation_ms']}, 'all_output_ids_exact': all(row['exact_token_ids_to_eager'] for row in samples)})
        save()
print(json.dumps(report['summaries'], ensure_ascii=False, indent=2), flush=True)
