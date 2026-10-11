"""Controlled reference benchmark; keeps native support and custom MTP distinct."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--model-dir', type=Path, required=True)
p.add_argument('--fixtures-root', type=Path, required=True)
p.add_argument('--output-root', type=Path, required=True)
p.add_argument('--official-summary', type=Path)
p.add_argument('--proposal-artifact', type=Path)
p.add_argument('--fixtures', default='zh_legal,en_contract,dense-equations')
p.add_argument('--max-tokens', type=int, default=256)
p.add_argument('--cycles', type=int, default=3)
p.add_argument('--smoke', action='store_true')
p.add_argument('--label', default='custom-torch-mtp')
a = p.parse_args()
result_dir = a.output_root / a.label
result_dir.mkdir(parents=True)
from qwen_mtp_torch import Reference
import torch
import transformers
manifest = json.loads((a.fixtures_root/'manifest.json').read_text())
fixtures = [next(f for f in manifest if f['id']==name) for name in a.fixtures.split(',')]
reference = Reference(a.model_dir)
if a.proposal_artifact:
    from indexed_proposal import install
    install(reference, a.proposal_artifact)
rows = []
def telemetry(stage):
    smi = '/usr/lib/wsl/lib/nvidia-smi' if Path('/usr/lib/wsl/lib/nvidia-smi').exists() else 'nvidia-smi'
    r = subprocess.run([smi,'--query-gpu=name,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw',
                        '--format=csv,noheader'],capture_output=True,text=True)
    with (result_dir/'telemetry.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(stage=stage,gpu=r.stdout.strip()))+'\n')
def messages(fixture):
    return [dict(role='user',content=[dict(type='image',image=str(a.fixtures_root/Path(fixture['path']).name)),
                                   dict(type='text',text=fixture['prompt'])])]
def run(f, mode, phase, force_reject=False, max_tokens=None):
    telemetry(f"{phase}-{f['id']}-{mode}-before")
    t0 = time.perf_counter()
    inputs = reference.prepare(messages(f))
    torch.cuda.synchronize()
    prepared = time.perf_counter()
    r = reference.generate(inputs, window=mode, max_tokens=max_tokens or a.max_tokens,
                           force_reject=force_reject)
    r.update(fixture=f['id'],mode=mode,phase=phase,e2e_ms=(time.perf_counter()-t0)*1000,
             preprocessing_ms=(prepared-t0)*1000)
    r['output_sha256'] = hashlib.sha256(r['output_text'].encode()).hexdigest()
    with (result_dir/'runs.jsonl').open('a') as stream:
        stream.write(json.dumps(r,ensure_ascii=False)+'\n')
    rows.append(r)
    print(json.dumps({k:r[k] for k in ('fixture','mode','phase','completion_tokens','decode_tps',
        'prefill_ms','mtp_prefill_ms','e2e_ms','acceptance_rate','rollback_replays')},ensure_ascii=False),flush=True)
    return r

telemetry('before')
for f in fixtures:
    print('WARMUP '+f['id'],flush=True)
    for repetition in range(2):
        for mode in (0,1,3):
            run(f,mode,'warmup')
    # A forced-rejection run exercises actual snapshot/restore rather than
    # merely testing an all-accepted trajectory.
    regular = run(f,0,'rollback-control',max_tokens=16)
    rejected = run(f,1,'rollback-forced-reject',force_reject=True,max_tokens=16)
    assert rejected['rollback_replays'] > 0
    assert regular['output_ids']==rejected['output_ids'], (f['id'],'rollback output differs')
    if not a.smoke:
        for cycle in range(a.cycles):
            order = (0,1,3,3,1,0) if cycle%2==0 else (3,1,0,0,1,3)
            for mode in order:
                run(f,mode,'formal')
                time.sleep(.15)
telemetry('after')
source = [x for x in rows if x['phase']==('warmup' if a.smoke else 'formal')]
summary = []
official_path = a.official_summary
official = {x['fixture']:x['baseline'] for x in json.loads(official_path.read_text())['summary']} if official_path and official_path.exists() else {}
for f in fixtures:
    modes = {}
    for mode in (0,1,3):
        selected = [r for r in source if r['fixture']==f['id'] and r['mode']==mode]
        modes[str(mode)] = {k:statistics.median(r[k] for r in selected) for k in (
            'main_prefill_ms','vision_ms','mtp_prefill_ms','prefill_ms','decode_ms','generation_ms',
            'decode_tps','prefill_tps','e2e_ms','preprocessing_ms','completion_tokens','drafted_tokens',
            'accepted_tokens','rollback_replays','rounds','cuda_allocated_mib')}
        modes[str(mode)].update(output_sha256=selected[0]['output_sha256'],
            repeated_output_exact=all(r['output_ids']==selected[0]['output_ids'] for r in selected),
            acceptance_rate=statistics.median(r['acceptance_rate'] for r in selected) if mode else None)
    summary.append(dict(fixture=f['id'],modes=modes,
        mtp1_output_exact=modes['0']['output_sha256']==modes['1']['output_sha256'],
        mtp3_output_exact=modes['0']['output_sha256']==modes['3']['output_sha256'],
        ordinary_matches_official=modes['0']['output_sha256']==official[f['id']]['output_sha256'] if official else None))
report = dict(scope='Transformers + explicit custom PyTorch MTP adapter; not native use_mtp',
    torch=torch.__version__,transformers=transformers.__version__,weights='bf16',kv='bf16',context=4096,
    target='upstream Qwen3.5 + fused FLA/causal-conv + SDPA; compiled fullgraph',
    proposal_head=reference.proposal_metadata if a.proposal_artifact else 'full original BF16 vocabulary',mtp_tensors=reference.mtp.loaded_tensor_count,
    backends=reference.backends,prefill_compiler_stats=reference.prefill_stats,
    rollback='snapshot conv/recurrent/attention frontiers; restore and replay accepted prefix on rejection',
    rollback_control_passed=True,method=f'2 warmups, {a.cycles} alternating balanced cycles; {2*a.cycles} formal runs per mode and fixture',
    smoke=a.smoke,summary=summary)
(result_dir/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)
