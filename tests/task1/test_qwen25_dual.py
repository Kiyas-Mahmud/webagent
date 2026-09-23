import copy
from types import SimpleNamespace

import pytest
import torch

from .test_assessment import rows
from web_agent.eval.task1.core import encode, make_views, request_key, sha
from web_agent.eval.task1.qwen25_backends import (
    QWEN25_BACKEND,
    QWEN25_CHECKPOINT_HASH,
    QWEN25_REVISION,
    Qwen25Generator,
    Qwen25RowBackend,
)
from web_agent.eval.task1.qwen25_dual import check_generation_parity
from web_agent.eval.task1.qwen25_profiles import (
    CONTRASTS,
    PROFILE,
    ROWS,
    ROW_IDS,
    row_spec,
)


def test_qwen25_profile_has_seven_explicit_rows_and_nine_contrasts():
    assert len(ROWS) == len(set(ROW_IDS)) == 7
    assert len(CONTRASTS) == 9
    assert [row.id for row in ROWS if row.output_mode == 'heads'] == ['qwen25_heads']
    assert not any('internvl' in str(row).lower() for row in ROWS)


def test_cli_dispatches_explicit_qwen25_profile(monkeypatch):
    from web_agent.eval.task1 import cli, qwen25_dual

    seen = []
    monkeypatch.setattr(qwen25_dual, 'main', lambda argv: seen.append(argv))
    argv = ['check', '--profile', PROFILE]
    cli.main(argv)
    assert seen == [argv]


def test_unknown_profile_does_not_silently_fallback():
    from web_agent.eval.task1 import cli

    with pytest.raises(ValueError, match='Unknown Task 1 profile'):
        cli.main(['check', '--profile', 'unknown'])


def test_selected_qwen25_assets_are_frozen_constants():
    assert QWEN25_REVISION == 'cc594898137f460bfe9f0759e9844b3ce807cfb5'
    assert QWEN25_CHECKPOINT_HASH == '71f867cc357d0410337a9b23398b109cc95d6c25c2ffc971138b7b8e3aafef2c'
    assert QWEN25_BACKEND == 'Qwen2.5-VL-7B-Instruct'


def test_wrong_qwen25_variant_rejected_before_prediction():
    with pytest.raises(ValueError, match='variant'):
        Qwen25RowBackend(SimpleNamespace(variant='base'), row_spec('browser_use_trained'))


def test_qwen25_processor_requires_two_dynamic_image_grids(monkeypatch, rows):
    import web_agent.eval.task1.qwen25_backends as backend

    generator = Qwen25Generator.__new__(Qwen25Generator)
    generator.root = '/unused'
    seen = {}
    first, second = object(), object()
    monkeypatch.setattr(backend, 'images', lambda request, root: [first, second])

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            seen['messages'] = messages
            return 'chat'

        def __call__(self, **kwargs):
            seen.update(kwargs)
            return {
                'input_ids': torch.tensor([[1]]),
                'attention_mask': torch.tensor([[1]]),
                'pixel_values': torch.zeros(28, 1176),
                'image_grid_thw': torch.tensor([[1, 4, 4], [1, 3, 4]]),
            }

    generator.processor = Processor()
    generator.batch('browser_use', make_views(rows, ['t:0'])[0][0])
    assert seen['images'] == [first, second]
    assert [part['type'] for part in seen['messages'][0]['content']] == ['image', 'image', 'text']
    assert 'images_kwargs' not in seen


def test_qwen25_rejects_collapsed_image_grid(monkeypatch, rows):
    import web_agent.eval.task1.qwen25_backends as backend

    generator = Qwen25Generator.__new__(Qwen25Generator)
    generator.root = '/unused'
    monkeypatch.setattr(backend, 'images', lambda request, root: [object(), object()])
    generator.processor = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: 'chat',
        __call__=None,
    )

    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return 'chat'

        def __call__(self, **kwargs):
            return {
                'input_ids': torch.tensor([[1]]),
                'attention_mask': torch.tensor([[1]]),
                'pixel_values': torch.zeros(16, 1176),
                'image_grid_thw': torch.tensor([[1, 4, 4]]),
            }

    generator.processor = Processor()
    with pytest.raises(ValueError, match='two ordered'):
        generator.batch('browser_use', make_views(rows, ['t:0'])[0][0])


def test_qwen25_base_and_trained_inputs_must_match(rows):
    requests = make_views(rows, ['t:0'])[0]
    identity = {'effective_generation': {'max_new_tokens': 128}}
    record = {
        'input_tensor_hashes': {'x': 'same'},
        'prompt': 'same',
        'serialized_chat': 'same',
        'base_parameter_signature_sha256': 'same',
        'generation_config_sha256': sha(encode(identity['effective_generation'])),
    }
    records = {
        system: {request_key(request): copy.deepcopy(record) for request in requests}
        for system in ROW_IDS
    }
    check_generation_parity(records, requests, identity)
    records['webvoyager_trained'][request_key(requests[0])]['serialized_chat'] = 'changed'
    with pytest.raises(ValueError, match='input parity'):
        check_generation_parity(records, requests, identity)
