"""Check functional compiler schemas, real/fake layouts and cache-state ownership."""
import torch
from causal_conv1d import causal_conv1d_fn
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from compiled_prefill import causal_prefill, gdn_prefill

torch.manual_seed(42)
with torch.inference_mode():
    for tokens in [1, 63, 65]:
        x = torch.randn(1, tokens, 6144, dtype=torch.bfloat16, device='cuda').transpose(1, 2)
        w = torch.randn(6144, 4, dtype=x.dtype, device='cuda') * 0.1
        args = (x, w, None, 'silu')
        torch.library.opcheck(causal_prefill, args, test_utils=('test_schema', 'test_faketensor'))
        assert torch.equal(causal_prefill(*args), causal_conv1d_fn(x, w, activation='silu'))
    for tokens in [63, 65]:
        q, k, v = [torch.randn(1, tokens, 16, 128, dtype=torch.bfloat16, device='cuda') * 0.1 for _ in range(3)]
        g = -torch.rand(1, tokens, 16, dtype=torch.float32, device='cuda') * 0.1
        beta = torch.rand(1, tokens, 16, dtype=q.dtype, device='cuda')
        for state in [None, torch.randn(1, 16, 128, 128, device='cuda') * 0.1]:
            before = None if state is None else state.clone()
            args = (q, k, v, g, beta, state, True)
            torch.library.opcheck(gdn_prefill, args, test_utils=('test_schema', 'test_faketensor'))
            actual = gdn_prefill(*args)
            reference = chunk_gated_delta_rule(q, k, v, g, beta, initial_state=state,
                                               output_final_state=True, use_qk_l2norm_in_kernel=True)
            assert all(torch.equal(a, b) for a, b in zip(actual, reference))
            assert state is None or torch.equal(state, before)
print('PASS: compiler schemas/layouts; unchanged fused outputs; initial state ownership')
