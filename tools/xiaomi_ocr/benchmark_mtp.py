"""Compare ordinary and MTP decode on the same binary and BF16 artifact."""
import argparse
import base64
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import subprocess
import threading
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--lanes', type=int, choices=[1, 2, 4], required=True)
p.add_argument('--cycles', type=int, default=5)
p.add_argument('--fixtures', help='Comma-separated fixture identifiers')
p.add_argument('--label', default='')
p.add_argument('--baseline-binary', type=Path)
p.add_argument('--optimized-binary', type=Path)
p.add_argument('--output-root', type=Path, required=True)
p.add_argument('--baseline-chunk', type=int, default=1024)
p.add_argument('--optimized-chunk', type=int, default=1024)
p.add_argument('--max-tokens', type=int, default=256)
p.add_argument('--context', type=int, default=4096)
p.add_argument('--transformers-baseline', action='store_true')
p.add_argument('--custom-transformers-mtp', action='store_true', help='Use the custom compiled PyTorch MTP1 reference')
p.add_argument('--allow-output-differences', action='store_true')
p.add_argument('--artifact', required=True, type=Path)
p.add_argument('--draft-window', type=int, choices=range(1, 6), default=1)
p.add_argument('--lm-head-draft', action='store_true')
p.add_argument('--compare-none', action='store_true', help='Compare two ordinary-decode binaries')
p.add_argument('--fixtures-root', type=Path, required=True)
p.add_argument('--hf-python', type=Path)
p.add_argument('--hf-model', type=Path)
p.add_argument('--hf-app-dir', type=Path)
p.add_argument('--baseline-draft-window', type=int, choices=range(1, 6))
p.add_argument('--baseline-lm-head-draft', action='store_true')
p.add_argument('--nvidia-smi', default='/usr/lib/wsl/lib/nvidia-smi' if Path('/usr/lib/wsl/lib/nvidia-smi').exists() else 'nvidia-smi')
a = p.parse_args()
if a.custom_transformers_mtp and (not a.transformers_baseline or a.compare_none or a.draft_window != 1):
    p.error('--custom-transformers-mtp requires --transformers-baseline, MTP1 and no --compare-none')
root = Path(__file__).resolve().parent
base = a.fixtures_root.resolve()
out = (a.output_root or root) / f'{a.label}paired{a.lanes}'
out.mkdir(parents=True)
def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def telemetry(stage):
    result = subprocess.run([a.nvidia_smi,
        '--query-gpu=name,driver_version,pstate,utilization.gpu,memory.used,clocks.sm,clocks.mem,temperature.gpu,power.draw',
        '--format=csv,noheader'], capture_output=True, text=True, check=False)
    with (out / 'gpu-telemetry.jsonl').open('a') as stream:
        stream.write(json.dumps({'timestamp_utc': timestamp(), 'stage': stage,
            'returncode': result.returncode, 'gpu': result.stdout.strip(),
            'error': result.stderr.strip()}) + '\n')

started_at = timestamp()
telemetry('before_servers')
manifest = json.loads((base / 'manifest.json').read_text())
names = ['zh_legal', 'en_contract', 'dense-equations']
if not all(any(f['id'] == n for f in manifest) for n in names):
    names = [f['id'] for f in manifest]
if a.fixtures:
    names = a.fixtures.split(',')
fixtures = [next(f for f in manifest if f['id'] == name) for name in names]
binary = root.parents[1] / 'build/apps/ninfer-serve'
targets = {'baseline': (19241, a.baseline_binary or binary),
           'optimized': (19242, a.optimized_binary or binary)}
if a.transformers_baseline:
    assert a.lanes == 1 and a.context == 4096
    if not a.hf_model:
        p.error('--transformers-baseline requires --hf-model')
    baseline_module = 'mtp_baseline_server' if a.custom_transformers_mtp else 'baseline_server'
    targets['baseline'] = (19500, (a.hf_app_dir or root) / f'{baseline_module}.py')
    targets['optimized'] = (19501, targets['optimized'][1])

def opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))

def health(port):
    with opener().open(f'http://127.0.0.1:{port}/health', timeout=1) as r:
        return json.load(r)

for port, _ in targets.values():
    try:
        health(port)
    except (urllib.error.URLError, TimeoutError):
        pass
    else:
        raise RuntimeError('Pair comparison port is occupied')

logs, processes, rows, cursors = {}, {}, [], {name: 0 for name in targets}
try:
    for name, (port, binary) in targets.items():
        logs[name] = (out / f'{name}.log').open('wb')
        command = [str(binary), str(a.artifact.resolve()),
            '--host', '127.0.0.1', '--port', str(port), '--model-id', 'ocr', '--vision',
            '--max-context', str(a.context), '--kv-capacity', str(a.context * a.lanes),
            '--max-concurrency', str(a.lanes), '--kv-dtype', 'bf16', '--prefill-chunk',
            str(a.baseline_chunk if name == 'baseline' else a.optimized_chunk),
            '--no-thinking', '--greedy', '--no-prefix-reuse', '--media-cache-mib', '0',
            '--request-log-jsonl', str(out / f'{name}-requests.jsonl')]
        window = a.draft_window if name == 'optimized' and not a.compare_none else a.baseline_draft_window if name == 'baseline' else None
        if window:
            command += ['--spec', 'mtp', '--draft-tokens', str(window)]
            if a.lm_head_draft if name == 'optimized' else a.baseline_lm_head_draft:
                command.append('--lm-head-draft')
        environment = os.environ.copy()
        if a.transformers_baseline and name == 'baseline':
            environment.update(XIAOMI_OCR_MODEL_PATH=str(a.hf_model or base / 'models/Xiaomi-OCR-0'), HF_HUB_OFFLINE='1',
                XIAOMI_BENCH_TIMING_MODE='wall', XIAOMI_TORCH_NUM_THREADS='1',
                XIAOMI_REQUIRE_FUSED_LINEAR='1', XIAOMI_COMPILE_DECODE='1',
                XIAOMI_COMPILE_PREFILL='1', XIAOMI_COMPILE_CACHE_FLOOR='4096',
                TORCHINDUCTOR_COMPILE_THREADS='1')
            if a.custom_transformers_mtp:
                env['XIAOMI_TF_MTP_WINDOW'] = '1'
                env.pop('XIAOMI_TF_PROPOSAL_ARTIFACT', None)
                if a.lm_head_draft:
                    env['XIAOMI_TF_PROPOSAL_ARTIFACT'] = str(a.artifact.resolve())
            command = [str(a.hf_python or sys.executable)] + [
                '-m', 'uvicorn', f'{baseline_module}:app', '--app-dir', str(a.hf_app_dir or root),
                '--host', '127.0.0.1', '--port', str(port)]
        processes[name] = subprocess.Popen(command, stdout=logs[name], stderr=logs[name], env=environment)
        for _ in range(300):
            if processes[name].poll() is not None:
                raise RuntimeError((out / f'{name}.log').read_text()[-3000:])
            try:
                information = health(port)
                if a.transformers_baseline and name == 'baseline':
                    assert information['compile_prefill'] and information['compile_decode']
                    assert all(v['fused'] for v in information['linear_backends'].values())
                    (out / 'transformers-audit.json').write_text(json.dumps(information, indent=2))
                break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(.2)
        else:
            raise RuntimeError('Server readiness timeout')
        (out / f'{name}-health.json').write_text(json.dumps(information, indent=2))
        telemetry(f'after_{name}_ready')
    bodies = {}
    for f in fixtures:
        path = base / Path(f['path']).name
        mime = 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg'
        bodies[f['id']] = json.dumps({'model': 'ocr', 'temperature': 0, 'presence_penalty': 0,
            'frequency_penalty': 0, 'top_p': 1, 'top_k': 0, 'min_p': 0, 'seed': 123, 'max_tokens': a.max_tokens,
            'messages': [{'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': 'data:' + mime + ';base64,' + base64.b64encode(path.read_bytes()).decode()}},
                {'type': 'text', 'text': f['prompt']}]}]}).encode()
    def request(name, fixture, cycle, warmup):
        port = targets[name][0]
        barrier = threading.Barrier(a.lanes)
        def worker(_):
            barrier.wait()
            req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', bodies[fixture],
                headers={'Content-Type': 'application/json'})
            try:
                with opener().open(req, timeout=180) as r:
                    return json.load(r)
            except urllib.error.HTTPError as error:
                raise RuntimeError(f'{name} {fixture}: HTTP {error.code}: {error.read().decode()}') from error
        begin = time.perf_counter()
        request_started = timestamp()
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.lanes) as pool:
            responses = list(pool.map(worker, range(a.lanes)))
        elapsed = time.perf_counter() - begin
        if a.transformers_baseline and name == 'baseline':
            result = responses[0]
            stage = result['timings']
            assert stage['compiled_decode_active']
            assert stage['prefill_compiler_stats']['compiled_language_calls'] > 0
            assert stage['prefill_compiler_stats']['actual_cache_max_length'] == 4096
            done = [{'event': 'request_done', 'result': result['usage'],
                'speculative': result.get('speculative', {}),
                'compiled_decode_active': stage['compiled_decode_active'],
                'prefill_compiler_stats': stage['prefill_compiler_stats'], 'timings_seconds': {
                'vision': stage['vision_ms'] / 1000, 'prefill': stage['text_prefill_ms'] / 1000,
                'decode': stage['decode_ms'] / 1000}}]
        else:
            records = [json.loads(line) for line in (out / f'{name}-requests.jsonl').read_text().splitlines()]
            done = [r for r in records if r['event'] == 'request_done'][cursors[name]:]
        cursors[name] += len(done)
        assert len(done) == a.lanes
        texts = [r['choices'][0]['message']['content'] for r in responses]
        assert len(set(texts)) == 1
        row = {'engine': name, 'fixture': fixture, 'cycle': cycle, 'warmup': warmup,
            'started_at_utc': request_started, 'finished_at_utc': timestamp(),
            'output_text': texts[0],
            'finish_reasons': [r['choices'][0]['finish_reason'] for r in responses],
            'completion_tokens': [r['usage']['completion_tokens'] for r in responses],
            'output_sha256': hashlib.sha256(texts[0].encode()).hexdigest(),
            'decode_tps': statistics.median((r['result']['completion_tokens'] - 1) / r['timings_seconds']['decode'] for r in done),
            'prefill_tps': statistics.median(r['result']['prompt_tokens'] / (r['timings_seconds']['vision'] + r['timings_seconds']['prefill']) for r in done),
            'total_tps': sum(r['usage']['completion_tokens'] for r in responses) / elapsed,
            'vision_ms': statistics.median(r['timings_seconds']['vision'] * 1000 for r in done),
            'language_ms': statistics.median(r['timings_seconds']['prefill'] * 1000 for r in done),
            'prefill_ms': statistics.median((r['timings_seconds']['vision'] + r['timings_seconds']['prefill']) * 1000 for r in done),
            'burst_ms': elapsed * 1000, 'request_metrics': done}
        rows.append(row)
        with (out / 'raw.jsonl').open('a') as s:
            s.write(json.dumps(row) + '\n')
    for f in fixtures:
        for _ in range(2):
            for name in targets:
                request(name, f['id'], -1, True)
    for cycle in range(a.cycles):
        telemetry(f'before_cycle_{cycle + 1}')
        for f in fixtures:
            order = ['baseline', 'optimized', 'optimized', 'baseline']
            if cycle % 2:
                order = ['optimized', 'baseline', 'baseline', 'optimized']
            for name in order:
                request(name, f['id'], cycle, False)
        print(f'Completed paired {a.lanes}-lane cycle {cycle + 1}', flush=True)
        telemetry(f'after_cycle_{cycle + 1}')
    summary = []
    for f in fixtures:
        values = {}
        for name in targets:
            selected = [r for r in rows if r['fixture'] == f['id'] and r['engine'] == name and not r['warmup']]
            assert len(selected) == 2 * a.cycles
            hashes = {r['output_sha256'] for r in selected}
            assert len(hashes) == 1
            values[name] = {key: statistics.median(r[key] for r in selected) for key in ['decode_tps', 'prefill_tps', 'total_tps', 'burst_ms', 'vision_ms', 'language_ms', 'prefill_ms']}
            values[name]['output_sha256'] = hashes.pop()
            values[name]['completion_tokens'] = selected[0]['completion_tokens']
            specs = [m.get('speculative', {'drafted_tokens': 0, 'accepted_tokens': 0,
                'rounds': 0, 'accepted_per_position': [], 'fallback_steps': 0})
                for r in selected for m in r['request_metrics']]
            proposed = sum(s['drafted_tokens'] for s in specs)
            accepted = sum(s['accepted_tokens'] for s in specs)
            rounds = sum(s['rounds'] for s in specs)
            values[name]['speculative'] = {
                'drafted_tokens': proposed, 'accepted_tokens': accepted, 'rounds': rounds,
                'acceptance_rate': accepted / proposed if proposed else None,
                'tokens_per_round': 1 + accepted / rounds if rounds else None,
                'accepted_per_position': [sum(s['accepted_per_position'][i] for s in specs)
                    for i in range(a.draft_window)] if name == 'optimized' and not a.compare_none else [],
                'fallback_steps': sum(s.get('fallback_steps', 0) for s in specs)}
        output_exact = values['baseline']['output_sha256'] == values['optimized']['output_sha256']
        assert output_exact or a.allow_output_differences
        pairs = []
        for cycle in range(a.cycles):
            current = [r for r in rows if r['fixture'] == f['id'] and r['cycle'] == cycle and not r['warmup']]
            med = {name: statistics.median(r['decode_tps'] for r in current if r['engine'] == name) for name in targets}
            pairs.append((med['optimized'] / med['baseline'] - 1) * 100)
        summary.append({'fixture': f['id'], **values, 'output_exact': output_exact,
            'paired_cycle_decode_gain_pct': pairs,
            **{key + '_gain_pct': (values['optimized'][key] / values['baseline'][key] - 1) * 100 for key in ['decode_tps', 'prefill_tps', 'total_tps']}})
    report = {'started_at_utc': started_at, 'finished_at_utc': timestamp(),
        'lanes': a.lanes, 'context_per_request': a.context, 'weights': 'bf16', 'kv': 'bf16',
        'chunks': {'baseline': a.baseline_chunk, 'optimized': a.optimized_chunk}, 'max_tokens': a.max_tokens,
        'sampling': {'temperature': 0, 'presence_penalty': 0, 'frequency_penalty': 0,
                     'top_p': 1, 'top_k': 0, 'min_p': 0, 'seed': 123},
        'method': f'Both real servers stay resident; two warmups per input and engine; {a.cycles} short cycles per fixture with alternating A-B-B-A/B-A-A-B order, {2 * a.cycles} measured bursts per engine',
        'binary_sha256': {name: hashlib.sha256(binary.read_bytes()).hexdigest() for name, (_, binary) in targets.items()},
                'baseline_kind': 'compiled Transformers + custom PyTorch MTP1' if a.custom_transformers_mtp else 'compiled Transformers BF16' if a.transformers_baseline else
            f'MTP{a.baseline_draft_window}' if a.baseline_draft_window else 'ordinary decode, same NInfer binary and artifact',
        'optimized_kind': 'ordinary decode regression' if a.compare_none else f'MTP{a.draft_window}',
        'proposal_head': 'optimized' if a.lm_head_draft else 'full BF16',
        'artifact': str(a.artifact.resolve()),
        'formal_output_count': sum(not r['warmup'] for r in rows) * a.lanes,
        'summary': summary}
    (out / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
finally:
    for process in processes.values():
        process.terminate()
    for process in processes.values():
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    for log in logs.values():
        log.close()
