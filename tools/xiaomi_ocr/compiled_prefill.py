"""Inference-only compiler boundaries for Qwen3.5 fused prefill kernels.

The CUDA/Triton implementations remain causal-conv1d and flash-linear-attention.
Functional custom operators hide their Python launchers and non-contiguous out=
buffers from Dynamo, while preserving the real output layouts and state dtype.
"""
import torch
from causal_conv1d import causal_conv1d_fn as _conv
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _chunk


@torch.library.custom_op('xiaomi_ocr::causal_prefill', mutates_args=())
def causal_prefill(x: torch.Tensor, weight: torch.Tensor,
                   bias: torch.Tensor | None, activation: str | None) -> torch.Tensor:
    return _conv(x, weight, bias, activation=activation)


@causal_prefill.register_fake
def _fake_conv(x, weight, bias, activation):
    # causal-conv1d retains channel-last layout for a channel-last input.
    if x.stride(1) == 1:
        return torch.empty((x.shape[0], x.shape[2], x.shape[1]),
                           device=x.device, dtype=x.dtype).transpose(1, 2)
    return torch.empty(x.shape, device=x.device, dtype=x.dtype)


@torch.library.custom_op('xiaomi_ocr::gdn_prefill', mutates_args=())
def gdn_prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                g: torch.Tensor, beta: torch.Tensor,
                initial_state: torch.Tensor | None,
                normalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
    return _chunk(q, k, v, g, beta, initial_state=initial_state,
                  output_final_state=True, use_qk_l2norm_in_kernel=normalize)


@gdn_prefill.register_fake
def _fake_gdn(q, k, v, g, beta, initial_state, normalize):
    return (torch.empty(v.shape, dtype=q.dtype, device=q.device),
            torch.empty((q.shape[0], v.shape[2], k.shape[-1], v.shape[-1]),
                        dtype=torch.float32, device=q.device))


def install_prefill_boundaries():
    """Install for cached, equal-length Qwen3.5 inference; decode stays upstream."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module
    if getattr(module, '_xiaomi_prefill_boundaries', False):
        return
    original_conv = module.causal_conv1d_fn
    original_chunk = module.torch_chunk_gated_delta_rule

    def conv(x, weight, bias=None, activation=None, **kwargs):
        if torch.compiler.is_compiling():
            return causal_prefill(x, weight, bias, activation)
        return original_conv(x, weight, bias, activation=activation, **kwargs)

    def chunk(q, k, v, g, beta, initial_state=None, output_final_state=False,
              use_qk_l2norm_in_kernel=False, cu_seqlens=None, **kwargs):
        if torch.compiler.is_compiling() and cu_seqlens is None and output_final_state:
            return gdn_prefill(q, k, v, g, beta, initial_state, use_qk_l2norm_in_kernel)
        return original_chunk(q, k, v, g=g, beta=beta, initial_state=initial_state,
                              output_final_state=output_final_state,
                              use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                              cu_seqlens=cu_seqlens, **kwargs)

    module.causal_conv1d_fn = conv
    module.torch_chunk_gated_delta_rule = chunk
    module._xiaomi_prefill_boundaries = True


def compile_vision(visual, mode='default'):
    """Compile vision tensor work; ragged-grid metadata is prepared outside Dynamo."""
    import functools
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    base = visual.forward
    compiled = torch.compile(base, fullgraph=True, dynamic=False, mode=mode)

    @functools.wraps(base)
    def forward(hidden_states, grid_thw, **kwargs):
        if torch.compiler.is_compiling():
            return base(hidden_states, grid_thw, **kwargs)
        indices, weights = q.get_vision_interpolation_indices_and_weights(
            grid_thw, visual.num_grid_per_side, visual.interpolation_mode,
            visual.interpolation_align_corners, visual.config.spatial_merge_size)
        positions = q.get_vision_position_ids(grid_thw, visual.spatial_merge_size)
        cu, maximum = q.get_vision_attention_seqlens(grid_thw, visual.config)
        return compiled(hidden_states, grid_thw, interp_indices=indices,
                        interp_weights=weights, position_ids=positions,
                        cu_seqlens=cu, max_seqlen=maximum, **kwargs)
    return base, forward


def compile_prefill(model, mode='default'):
    """Enable separate fullgraph vision and cached language prefill regions."""
    import functools
    install_prefill_boundaries()
    language = model.model.language_model
    base = language.forward
    compiled = torch.compile(base, fullgraph=True, dynamic=False, mode=mode)
    stats = {'cache_storage_initializations': 0, 'compiled_language_calls': 0,
             'cache_type': None, 'actual_cache_max_length': None}

    @functools.wraps(base)
    def forward(*args, **kwargs):
        # Official generate compiles single-token decode separately. Inline the
        # original language implementation inside that existing decode graph.
        if torch.compiler.is_compiling():
            return base(*args, **kwargs)
        cache = kwargs.get('past_key_values')
        stats['cache_type'] = type(cache).__name__
        if cache is None:
            raise RuntimeError('Compiled Xiaomi prefill requires cached inference')
        # Initialize storage only, outside Dynamo. Unlike an eager prefill this
        # does no model arithmetic and does not advance the cache state.
        hidden = kwargs.get('inputs_embeds')
        if hidden is None:
            raise RuntimeError('Compiled multimodal prefill requires inputs_embeds')
        cache.early_initialization(hidden.shape[0], language.config.num_key_value_heads,
                                   language.config.head_dim, hidden.dtype, hidden.device)
        for index, layer_cache in enumerate(cache.layers):
            if hasattr(layer_cache, 'is_conv_states_initialized'):
                attn = language.layers[index].linear_attn
                conv = None if layer_cache.is_conv_states_initialized[0] else torch.empty(
                    (hidden.shape[0], attn.conv1d.weight.shape[0], attn.conv_kernel_size),
                    dtype=hidden.dtype, device=hidden.device)
                recurrent = None if layer_cache.is_recurrent_states_initialized[0] else torch.empty(
                    (hidden.shape[0], attn.num_v_heads, attn.head_k_dim, attn.head_v_dim),
                    dtype=torch.float32, device=hidden.device)
                if conv is not None or recurrent is not None:
                    layer_cache.lazy_initialization(conv_states=conv, recurrent_states=recurrent,
                                                     conv_kernel_size=attn.conv_kernel_size)
                    stats['cache_storage_initializations'] += 1
        stats['actual_cache_max_length'] = cache.get_max_length()
        stats['compiled_language_calls'] += 1
        return compiled(*args, **kwargs)

    language.forward = forward
    _, model.model.visual.forward = compile_vision(model.model.visual, mode)
    return stats
