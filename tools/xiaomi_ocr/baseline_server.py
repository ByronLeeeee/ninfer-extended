"""Official Transformers BF16 generation with selectable stage instrumentation."""
import base64, io, json, threading, time, functools, os, inspect, importlib
from pathlib import Path
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from transformers import AutoModelForImageTextToText, AutoProcessor, CompileConfig

MODEL=os.environ.get('XIAOMI_OCR_MODEL_PATH', str(Path(__file__).resolve().parents[2] / 'models/Xiaomi-OCR-0'))
if os.environ.get('XIAOMI_TORCH_NUM_THREADS'):
    torch.set_num_threads(int(os.environ['XIAOMI_TORCH_NUM_THREADS']))

def linear_backend_info():
    module=importlib.import_module('transformers.models.qwen3_5.modeling_qwen3_5')
    result={}
    for name in ['causal_conv1d_fn','causal_conv1d_update','torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule']:
        function=getattr(module,name); seen=set()
        while inspect.isfunction(function) and id(function) not in seen:
            seen.add(id(function)); closure=inspect.getclosurevars(function).nonlocals
            if 'is_new_implementation' in closure:
                implementation=closure['implementation']
                result[name]={'fused':closure['is_new_implementation'],'implementation_module':implementation.__module__}
                break
            function=getattr(function,'__wrapped__',None)
        else: result[name]={'fused':None,'implementation_module':None}
    return result

linear_backends=linear_backend_info()
if os.environ.get('XIAOMI_REQUIRE_FUSED_LINEAR')=='1' and not all(v['fused'] for v in linear_backends.values()):
    raise RuntimeError('Optimized original benchmark requires all four fused linear-attention kernels: '+json.dumps(linear_backends))
processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
compile_prefill_enabled=os.environ.get('XIAOMI_COMPILE_PREFILL')=='1'
prefill_compiler_stats={}
if compile_prefill_enabled:
    from compiled_prefill import compile_prefill
    prefill_compiler_stats=compile_prefill(model)
app=FastAPI(); lock=threading.Lock(); records=[]; vision_events=[]
timing_mode=os.environ.get('XIAOMI_BENCH_TIMING_MODE', 'cuda-events')
compile_decode=os.environ.get('XIAOMI_COMPILE_DECODE')=='1'
if compile_decode and timing_mode!='wall':
    raise RuntimeError('Compiled decode requires wall timing outside the compiled graph')
compile_cache_floor=int(os.environ.get('XIAOMI_COMPILE_CACHE_FLOOR', '4096'))
forward_wall=[]; vision_wall=[]
original_forward=model.forward

def event(): return torch.cuda.Event(enable_timing=True)
@functools.wraps(original_forward)
def timed_forward(*args,**kwargs):
    # Official generate compiles decode, while prefill calls the ordinary forward.
    # Keep timing/state mutation out of the compiled graph itself.
    if torch.compiler.is_compiling():
        return original_forward(*args,**kwargs)
    began=time.perf_counter()
    start,end=(event(),event()) if timing_mode!='wall' else (None,None)
    if start is not None: start.record()
    result=original_forward(*args,**kwargs)
    if end is not None: end.record()
    elif not records: torch.cuda.synchronize()
    forward_wall.append((began,time.perf_counter()))
    records.append((start,end)); return result
model.forward=timed_forward
def vision_begin(module,args):
    if timing_mode=='wall': vision_wall.append([time.perf_counter(),None])
    else:
        start,end=event(),event(); start.record(); vision_events.append((start,end))
def vision_end(module,args,result):
    if timing_mode=='wall':
        torch.cuda.synchronize(); vision_wall[-1][1]=time.perf_counter()
    else: vision_events[-1][1].record()
model.model.visual.register_forward_pre_hook(vision_begin)
model.model.visual.register_forward_hook(vision_end)

@app.get('/health')
def health(): return {'status':'ready','engine':'transformers','dtype':'bfloat16','linear_backends':linear_backends,'torch_num_threads':torch.get_num_threads(),'torch_num_interop_threads':torch.get_num_interop_threads(),'compile_decode':compile_decode,'compile_mode':'reduce-overhead' if compile_decode else None,'compile_prefill':compile_prefill_enabled,'prefill_scope':'vision and language fullgraph; grid metadata eager' if compile_prefill_enabled else None,'static_cache_floor':compile_cache_floor if compile_decode else None}
@app.get('/v1/models')
def models(): return {'data':[{'id':'xiaomi-ocr-0-bf16'}]}
@app.post('/v1/chat/completions')
def generate(body:dict):
    if body.get('stream'): raise HTTPException(400,'Use non-streaming for CUDA-event timings')
    with lock,torch.inference_mode():
        t0=time.perf_counter(); messages=[]
        for message in body['messages']:
            content=[]
            if isinstance(message['content'],str): content=[{'type':'text','text':message['content']}]
            else:
                for item in message['content']:
                    if item['type']=='text': content.append(item)
                    elif item['type']=='image_url':
                        url=item['image_url']['url']
                        if not url.startswith('data:image/'): raise HTTPException(400,'Local data images required')
                        img=Image.open(io.BytesIO(base64.b64decode(url.split(',',1)[1]))).convert('RGB')
                        content.append({'type':'image','image':img})
            messages.append({'role':message['role'],'content':content})
        inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',enable_thinking=False).to('cuda')
        torch.cuda.synchronize(); preprocessing_ms=(time.perf_counter()-t0)*1000
        records.clear(); vision_events.clear(); forward_wall.clear(); vision_wall.clear(); torch.cuda.reset_peak_memory_stats(); began=time.perf_counter()
        compile_options={'cache_implementation':'static','max_cache_len':compile_cache_floor,'compile_config':CompileConfig(mode='reduce-overhead',fullgraph=True)} if compile_decode else {}
        outputs=model.generate(**inputs,max_new_tokens=body.get('max_tokens',4096),do_sample=False,use_cache=True,eos_token_id=248046,pad_token_id=248044,**compile_options)
        torch.cuda.synchronize(); finished=time.perf_counter(); generate_ms=(finished-began)*1000
        ids=outputs[0,inputs['input_ids'].shape[1]:].tolist()
        text=processor.decode(ids,skip_special_tokens=True)
        if timing_mode=='wall':
            prefill_ms=(forward_wall[0][1]-forward_wall[0][0])*1000
            decode_ms=(finished-forward_wall[0][1])*1000
            vision_ms=sum((b-a)*1000 for a,b in vision_wall)
        else:
            prefill_ms=records[0][0].elapsed_time(records[0][1])
            decode_ms=sum(a.elapsed_time(b) for a,b in records[1:])
            vision_ms=sum(a.elapsed_time(b) for a,b in vision_events)
        timing={'preprocessing_ms':preprocessing_ms,'vision_ms':vision_ms,'prefill_ms':prefill_ms,'text_prefill_ms':prefill_ms-vision_ms,'prefill_tokens':inputs['input_ids'].shape[1],'decode_ms':decode_ms,'decode_tokens':max(0,len(ids)-1),'generation_ms':generate_ms,'e2e_ms':(time.perf_counter()-t0)*1000}
        timing['prefill_tps']=timing['prefill_tokens']/(prefill_ms/1000)
        timing['decode_tps']=timing['decode_tokens']/(decode_ms/1000) if decode_ms else None
        timing['cuda_allocated_mib']=torch.cuda.memory_allocated()/2**20
        timing['cuda_reserved_mib']=torch.cuda.memory_reserved()/2**20
        timing['cuda_peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
        timing['timing_mode']=timing_mode
        timing['compiled_decode_active']=compile_decode and hasattr(model,'_compiled_call')
        timing['prefill_compiler_stats']=dict(prefill_compiler_stats)
        return {'id':'hf-'+str(time.time_ns()),'object':'chat.completion','model':'xiaomi-ocr-0-bf16','choices':[{'index':0,'message':{'role':'assistant','content':text},'finish_reason':'stop' if ids[-1]==248046 else 'length'}],'usage':{'prompt_tokens':timing['prefill_tokens'],'completion_tokens':len(ids),'total_tokens':timing['prefill_tokens']+len(ids)},'timings':timing,'output_token_ids':ids,'input_token_ids':inputs['input_ids'][0].tolist(),'image_grid_thw':inputs.get('image_grid_thw',torch.empty(0)).tolist()}
