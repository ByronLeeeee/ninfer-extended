"""BF16 Transformers/PyTorch MTP reference for the Qwen3.5 OCR checkpoint.

This is an explicit benchmark adapter, not Transformers' native use_mtp path.
The target uses upstream Qwen3.5 layers, fused FLA/causal-conv kernels and SDPA.
Rejected speculative tokens restore conv/FP32 recurrent state and replay the
accepted prefix; changing only the KV length would be incorrect for this model.
"""
import copy
import functools
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForImageTextToText, AutoProcessor, StaticCache
from transformers.models.qwen3_5 import modeling_qwen3_5 as q
from transformers.masking_utils import create_causal_mask


def fused_backends():
    import inspect
    result = {}
    for name in ('causal_conv1d_fn', 'causal_conv1d_update',
                 'torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule'):
        fn = getattr(q, name)
        seen = set()
        while inspect.isfunction(fn) and id(fn) not in seen:
            seen.add(id(fn))
            variables = inspect.getclosurevars(fn).nonlocals
            if 'is_new_implementation' in variables:
                result[name] = dict(fused=variables['is_new_implementation'],
                    implementation_module=variables['implementation'].__module__)
                break
            fn = getattr(fn, '__wrapped__', None)
        else:
            result[name] = dict(fused=None)
    return result


class MtpHead(torch.nn.Module):
    def __init__(self, model, checkpoint):
        super().__init__()
        cfg = copy.deepcopy(model.config.text_config)
        cfg.layer_types = ['full_attention']
        cfg.num_hidden_layers = 1
        self.config = cfg
        with torch.device('cuda'):
            self.fc = torch.nn.Linear(2 * cfg.hidden_size, cfg.hidden_size, bias=False,
                                      dtype=torch.bfloat16)
            self.pre_fc_norm_embedding = q.Qwen3_5RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
            self.pre_fc_norm_hidden = q.Qwen3_5RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
            self.layers = torch.nn.ModuleList([q.Qwen3_5DecoderLayer(cfg, 0)])
            self.norm = q.Qwen3_5RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
            self.rotary_emb = q.Qwen3_5TextRotaryEmbedding(cfg)
        self.to(dtype=torch.bfloat16)
        weights = {}
        index_path = checkpoint / 'model.safetensors.index.json'
        if index_path.exists():
            mapping = json.loads(index_path.read_text())['weight_map']
            files = sorted(set(mapping[k] for k in mapping if k.startswith('mtp.')))
        else:
            files = ['model.safetensors']
        for filename in files:
            with safe_open(checkpoint / filename, framework='pt', device='cpu') as stream:
                for key in stream.keys():
                    if key.startswith('mtp.'):
                        weights[key[4:]] = stream.get_tensor(key)
        missing, unexpected = self.load_state_dict(weights, strict=False)
        assert not unexpected, unexpected
        assert not [k for k in missing if not k.startswith('rotary_emb.')], missing
        self.loaded_tensor_count = len(weights)
        self.eval()
        # Tie the checkpoint's original embedding/output head, with no quantization.
        self.embedding = model.model.language_model.embed_tokens
        self.lm_head = model.lm_head
        self.cache = StaticCache(cfg, max_cache_len=4096)
        self.cache.early_initialization(1, cfg.num_key_value_heads, cfg.head_dim,
                                       torch.bfloat16, torch.device('cuda'))

    def stem(self, previous_hidden, next_embeddings):
        return self.fc(torch.cat((self.pre_fc_norm_embedding(next_embeddings),
                                  self.pre_fc_norm_hidden(previous_hidden)), dim=-1))

    def decode(self, previous_hidden, next_embeddings, positions):
        x = self.stem(previous_hidden, next_embeddings)
        mask = create_causal_mask(config=self.config, inputs_embeds=x,
            attention_mask=None, past_key_values=self.cache, position_ids=None)
        x = self.layers[0](x, position_embeddings=self.rotary_emb(x, positions),
                           attention_mask=mask, past_key_values=self.cache, use_cache=True)
        hidden = self.norm(x)
        token = self.lm_head(hidden[:, -1:]).argmax(-1)
        return hidden, token

    def prefill(self, previous_hidden, next_embeddings, positions):
        # The earlier MTP prompt columns supply KV only. Their attention/MLP
        # outputs are unused, so compute a complete block only for the last one.
        x = self.stem(previous_hidden[:, :-1], next_embeddings[:, :-1])
        attn = self.layers[0].self_attn
        y = self.layers[0].input_layernorm(x)
        shape = (1, x.shape[1], -1, attn.head_dim)
        key = attn.k_norm(attn.k_proj(y).view(shape)).transpose(1, 2)
        value = attn.v_proj(y).view(shape).transpose(1, 2)
        cos, sin = self.rotary_emb(x, positions[:, :, :-1])
        _, key = q.apply_rotary_pos_emb(key, key, cos, sin)
        self.cache.update(key, value, 0)
        return self.decode(previous_hidden[:, -1:], next_embeddings[:, -1:],
                           positions[:, :, -1:])


class Reference:
    def __init__(self, checkpoint, compile_mode='reduce-overhead'):
        torch.set_num_threads(1)
        self.backends = fused_backends()
        assert all(x['fused'] for x in self.backends.values()), self.backends
        self.processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(checkpoint,
            dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).to('cuda').eval()
        from compiled_prefill import compile_prefill
        self.prefill_stats = compile_prefill(self.model)
        self.language = self.model.model.language_model
        self.cache = StaticCache(self.model.config, max_cache_len=4096)
        cfg = self.model.config.text_config
        self.cache.early_initialization(1, cfg.num_key_value_heads, cfg.head_dim,
                                       torch.bfloat16, torch.device('cuda'))
        self.live = []
        for index, layer in enumerate(self.cache.layers):
            if hasattr(layer, 'recurrent_states'):
                attn = self.language.layers[index].linear_attn
                layer.lazy_initialization(
                    conv_states=torch.empty((1, attn.conv_dim, attn.conv_kernel_size),
                        dtype=torch.bfloat16, device='cuda'),
                    recurrent_states=torch.empty((1, attn.num_v_heads, attn.head_k_dim, attn.head_v_dim),
                        dtype=torch.float32, device='cuda'), conv_kernel_size=attn.conv_kernel_size)
                self.live += [layer.conv_states[0], layer.recurrent_states[0]]
            else:
                self.live.append(layer.cumulative_length)
        self.saved = [torch.empty_like(x) for x in self.live]
        for x in self.saved:
            torch._dynamo.mark_static_address(x)
        self.mtp = MtpHead(self.model, checkpoint)
        self.target = torch.compile(self.target_eager, fullgraph=True, dynamic=False, mode=compile_mode)
        self.propose = torch.compile(self.mtp.decode, fullgraph=True, dynamic=False, mode=compile_mode)
        self.mtp_prefill = torch.compile(self.mtp.prefill, fullgraph=True, dynamic=False)
        self.save_state = torch.compile(functools.partial(self.copy_state, self.live, self.saved),
            fullgraph=True, dynamic=False, mode=compile_mode)
        self.restore_state = torch.compile(functools.partial(self.copy_state, self.saved, self.live),
            fullgraph=True, dynamic=False, mode=compile_mode)

    @staticmethod
    def copy_state(source, destination):
        torch._foreach_copy_(destination, source)

    def target_eager(self, ids, positions):
        result = self.language(input_ids=None, inputs_embeds=self.language.embed_tokens(ids),
            position_ids=positions, attention_mask=None, past_key_values=self.cache, use_cache=True)
        hidden = result.last_hidden_state
        return hidden, self.model.lm_head(hidden).argmax(-1)

    def target_step(self, ids, frontier, delta):
        positions = (torch.arange(ids.shape[1], device='cuda') + frontier + delta).view(1, 1, -1).expand(3, 1, -1)
        torch.compiler.cudagraph_mark_step_begin()
        hidden, tokens = self.target(ids, positions)
        # Keep returned values independent of the next CUDA Graph replay.
        return hidden.clone(), tokens.clone()

    def draft_step(self, hidden, ids, positions):
        torch.compiler.cudagraph_mark_step_begin()
        result, token = self.propose(hidden.contiguous(), self.language.embed_tokens(ids), positions.contiguous())
        return result.clone(), token.clone()

    def prepare(self, messages):
        return self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors='pt', enable_thinking=False).to('cuda')

    @torch.inference_mode()
    def generate(self, inputs, window=0, max_tokens=256, force_reject=False):
        self.cache.reset()
        self.mtp.cache.reset()
        self.model.model.rope_deltas = None
        torch.cuda.synchronize()
        began = time.perf_counter()
        emb = self.language.embed_tokens(inputs['input_ids'])
        vision_begin = time.perf_counter()
        image_outputs = self.model.model.get_image_features(inputs['pixel_values'],
            inputs['image_grid_thw'], return_dict=True)
        image_emb = torch.cat(image_outputs.pooler_output, dim=0).to(emb.dtype)
        image_mask, _ = self.model.model.get_placeholder_mask(inputs['input_ids'],
            inputs_embeds=emb, image_features=image_emb)
        emb = emb.masked_scatter(image_mask, image_emb)
        torch.cuda.synchronize()
        vision_ms = (time.perf_counter() - vision_begin) * 1000
        positions = self.model.model.compute_3d_position_ids(input_ids=inputs['input_ids'],
            inputs_embeds=emb, image_grid_thw=inputs['image_grid_thw'],
            attention_mask=inputs['attention_mask'], past_key_values=self.cache,
            mm_token_type_ids=inputs['mm_token_type_ids'])
        hidden = self.language(input_ids=None, inputs_embeds=emb, position_ids=positions,
            attention_mask=inputs['attention_mask'], past_key_values=self.cache,
            use_cache=True).last_hidden_state
        anchor = self.model.lm_head(hidden[:, -1:]).argmax(-1)
        torch.cuda.synchronize()
        main_prefill_end = time.perf_counter()
        frontier = inputs['input_ids'].shape[1]
        delta = int(self.model.model.rope_deltas.item())
        drafts = []
        if window:
            shifted = torch.cat((emb[:, 1:], self.language.embed_tokens(anchor)), dim=1)
            torch.compiler.cudagraph_mark_step_begin()
            draft_hidden, draft = self.mtp_prefill(hidden, shifted, positions)
            draft_hidden, draft = draft_hidden.clone(), draft.clone()
            drafts = [draft]
            for i in range(1, window):
                pos = torch.full((3, 1, 1), frontier + i - 1 + delta, device='cuda', dtype=torch.long)
                draft_hidden, draft = self.draft_step(draft_hidden, draft, pos)
                drafts.append(draft)
        torch.cuda.synchronize()
        prefill_end = time.perf_counter()
        output = [int(anchor.item())]
        drafted = accepted_total = replay_count = rounds = tail_steps = 0
        accepted_positions = [0] * window
        while len(output) < max_tokens and output[-1] != 248046:
            # Finish a short remaining budget with ordinary single-token calls.
            # This avoids counting draft tokens that cannot be emitted.
            if window == 0 or max_tokens - len(output) <= window:
                _, next_token = self.target_step(anchor, frontier, delta)
                frontier += 1
                anchor = next_token
                output.append(int(anchor.item()))
                tail_steps += 1
                continue
            proposal = torch.cat(drafts, dim=1)
            if force_reject:
                proposal = (proposal + 1) % self.model.config.text_config.vocab_size
            ids = torch.cat((anchor, proposal), dim=1)
            self.save_state()
            target_hidden, target_tokens = self.target_step(ids, frontier, delta)
            candidate_ids = proposal[0].tolist()
            verified_ids = target_tokens[0].tolist()
            accepted = 0
            for prediction, candidate in zip(verified_ids, candidate_ids):
                if prediction != candidate:
                    break
                accepted += 1
            rounds += 1
            drafted += window
            accepted_total += accepted
            for i in range(accepted):
                accepted_positions[i] += 1
            if accepted < window:
                self.restore_state()
                self.target_step(ids[:, :accepted + 1].contiguous(), frontier, delta)
                replay_count += 1
            licensed = verified_ids[:accepted + 1]
            remaining = max_tokens - len(output)
            for token in licensed[:remaining]:
                output.append(token)
                if token == 248046:
                    break
            if output[-1] == 248046 or len(output) == max_tokens:
                break
            # Speculative AR entries have attention-only state. Rewind its KV
            # frontier before realigning it with target-verified hidden states.
            self.mtp.cache.layers[0].cumulative_length.fill_(frontier)
            aligned_ids = target_tokens[:, :accepted + 1].contiguous()
            aligned_positions = (torch.arange(accepted + 1, device='cuda') + frontier + delta).view(1, 1, -1).expand(3, 1, -1)
            draft_hidden, draft = self.draft_step(target_hidden[:, :accepted + 1].contiguous(),
                                                 aligned_ids, aligned_positions)
            draft_hidden = draft_hidden[:, -1:].contiguous()
            frontier += accepted + 1
            drafts = [draft]
            for i in range(1, window):
                pos = torch.full((3, 1, 1), frontier + i - 1 + delta, device='cuda', dtype=torch.long)
                draft_hidden, draft = self.draft_step(draft_hidden, draft, pos)
                drafts.append(draft)
            anchor = target_tokens[:, accepted:accepted + 1].contiguous()
        torch.cuda.synchronize()
        end = time.perf_counter()
        return dict(output_ids=output, output_text=self.processor.decode(output, skip_special_tokens=True),
            prompt_tokens=inputs['input_ids'].shape[1], completion_tokens=len(output),
            main_prefill_ms=(main_prefill_end-began)*1000, vision_ms=vision_ms,
            mtp_prefill_ms=(prefill_end-main_prefill_end)*1000,
            prefill_ms=(prefill_end-began)*1000, decode_ms=(end-prefill_end)*1000,
            generation_ms=(end-began)*1000, decode_tps=(len(output)-1)/(end-prefill_end),
            prefill_tps=inputs['input_ids'].shape[1]/(prefill_end-began),
            drafted_tokens=drafted, accepted_tokens=accepted_total,
            acceptance_rate=accepted_total/drafted if drafted else None,
            accepted_per_position=accepted_positions, rounds=rounds,
            rollback_replays=replay_count, tail_steps=tail_steps,
            force_reject=force_reject, cuda_allocated_mib=torch.cuda.memory_allocated()/2**20,
            prefill_compiler_stats=dict(self.prefill_stats))
