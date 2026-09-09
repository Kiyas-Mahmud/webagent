"""Real CUDA equivalence with the pinned, unmodified bitsandbytes kernel."""
from copy import deepcopy
import pytest
from web_agent.runtime.qwen2vl_pc01 import _install_immutable_linear4bit, PC01RuntimeError

@pytest.mark.parametrize('input_dtype', ['float16', 'float32'])
@pytest.mark.parametrize('bias_enabled', [True, False])
def test_frozen_bias_preserves_upstream_outputs_and_entire_state(input_dtype, bias_enabled):
    torch = pytest.importorskip('torch')
    bnb = pytest.importorskip('bitsandbytes')
    if not torch.cuda.is_available():
        pytest.skip('real CUDA kernel required')
    torch.manual_seed(42)
    reference = bnb.nn.Linear4bit(64, 32, bias=bias_enabled, compute_dtype=torch.float16,
                                quant_type='nf4', compress_statistics=True).cuda()
    if reference.bias is not None:
        reference.bias.data = reference.bias.data.float()
    reference.requires_grad_(False).eval()
    candidate = deepcopy(reference)
    before = {k: v.clone() for k, v in candidate.state_dict().items()}
    _install_immutable_linear4bit(candidate)
    _install_immutable_linear4bit(candidate)
    x = torch.randn(2, 4, 64, device='cuda', dtype=getattr(torch, input_dtype))
    with torch.inference_mode():
        expected = reference(x)
        actual = candidate(x)
        repeat = candidate(x)
    assert torch.equal(actual, expected)
    assert torch.equal(repeat, expected)
    after = candidate.state_dict()
    assert before.keys() == after.keys()
    for key in before:
        assert before[key].dtype == after[key].dtype
        assert torch.equal(before[key], after[key]), key
    if bias_enabled:
        assert reference.bias.dtype == torch.float16
        assert candidate.bias.dtype == torch.float32
    candidate.train()
    with pytest.raises(PC01RuntimeError, match='frozen CUDA'):
        candidate(x)
