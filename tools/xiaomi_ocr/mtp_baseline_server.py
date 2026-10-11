"""HTTP benchmark wrapper for Transformers + the explicit PyTorch MTP adapter."""
import base64
import io
import json
import os
from pathlib import Path
import sys
import threading
import time

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException

from qwen_mtp_torch import Reference

model_path = Path(os.environ['XIAOMI_OCR_MODEL_PATH'])
window = int(os.environ.get('XIAOMI_TF_MTP_WINDOW', '1'))
assert window in (0, 1, 3)
reference = Reference(model_path)
proposal_artifact = os.environ.get('XIAOMI_TF_PROPOSAL_ARTIFACT')
proposal_rows = 0
if proposal_artifact:
    from indexed_proposal import install
    install(reference, Path(proposal_artifact))
    proposal_rows = reference.proposal_metadata['rows']
app = FastAPI()
lock = threading.Lock()

@app.get('/health')
def health():
    return dict(status='ready',engine='transformers-custom-mtp',dtype='bfloat16',
        linear_backends=reference.backends,compile_prefill=True,compile_decode=True,
        speculative_backend='custom-pytorch-mtp',draft_window=window,
        proposal_head=f'original BF16 shortlist: {proposal_rows} rows' if proposal_rows else 'full original BF16 vocabulary',static_cache_floor=4096)

@app.post('/v1/chat/completions')
def generate(body:dict):
    if body.get('stream'):
        raise HTTPException(400,'Benchmark wrapper uses complete responses')
    with lock,torch.inference_mode():
        began=time.perf_counter()
        messages=[]
        for message in body['messages']:
            content=[]
            if isinstance(message['content'],str):
                content=[dict(type='text',text=message['content'])]
            else:
                for item in message['content']:
                    if item['type']=='text':
                        content.append(item)
                    elif item['type']=='image_url':
                        url=item['image_url']['url']
                        content.append(dict(type='image',image=Image.open(io.BytesIO(
                            base64.b64decode(url.split(',',1)[1]))).convert('RGB')))
            messages.append(dict(role=message['role'],content=content))
        inputs=reference.prepare(messages)
        torch.cuda.synchronize()
        prepared=time.perf_counter()
        result=reference.generate(inputs,window=window,max_tokens=body.get('max_tokens',256))
        counters=torch._dynamo.utils.counters
        graph_breaks=dict(counters['graph_break'])
        assert not graph_breaks,graph_breaks
        timing=dict(preprocessing_ms=(prepared-began)*1000,
            vision_ms=result['vision_ms'],prefill_ms=result['prefill_ms'],
            text_prefill_ms=result['prefill_ms']-result['vision_ms'],
            main_prefill_ms=result['main_prefill_ms'],mtp_prefill_ms=result['mtp_prefill_ms'],
            decode_ms=result['decode_ms'],generation_ms=result['generation_ms'],
            e2e_ms=(time.perf_counter()-began)*1000,decode_tps=result['decode_tps'],
            prefill_tps=result['prefill_tps'],compiled_decode_active=counters['stats']['unique_graphs']>0,
            prefill_compiler_stats=dict(reference.prefill_stats),compiler_graph_breaks=graph_breaks)
        speculative={k:result[k] for k in ('drafted_tokens','accepted_tokens','rounds',
            'acceptance_rate','accepted_per_position','rollback_replays')}
        speculative['tokens_per_round']=(result['completion_tokens']-1)/result['rounds'] if result['rounds'] else None
        speculative['fallback_steps']=result['tail_steps']
        return dict(id='torch-mtp-'+str(time.time_ns()),object='chat.completion',model='ocr',
            choices=[dict(index=0,message=dict(role='assistant',content=result['output_text']),
                finish_reason='stop' if result['output_ids'][-1]==248046 else 'length')],
            usage=dict(prompt_tokens=result['prompt_tokens'],completion_tokens=result['completion_tokens'],
                total_tokens=result['prompt_tokens']+result['completion_tokens']),timings=timing,
            speculative=speculative,output_token_ids=result['output_ids'],
            input_token_ids=inputs['input_ids'][0].tolist())
