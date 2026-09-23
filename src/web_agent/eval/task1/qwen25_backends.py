"""Frozen Qwen2.5-VL-7B base, trained decoder, and trained-head routes."""
from __future__ import annotations

import copy

from .backends import (
    base_parameter_signature,
    images,
    prompt,
    require_local_images,
    tensor_fingerprints,
)
from .core import PHASES, encode, file_hash, parse_response, sha, validate_request

QWEN25_REVISION = 'cc594898137f460bfe9f0759e9844b3ce807cfb5'
QWEN25_CHECKPOINT_HASH = '71f867cc357d0410337a9b23398b109cc95d6c25c2ffc971138b7b8e3aafef2c'
QWEN25_BACKEND = 'Qwen2.5-VL-7B-Instruct'


def _validate_saved_config(saved):
    if saved.get('epoch') != 0:
        raise ValueError('Expected selected Qwen2.5 epoch 0 checkpoint')
    cfg = saved['config']
    bb = cfg['backbone']
    expected = {
        'family': 'qwen2_vl',
        'revision': QWEN25_REVISION,
        'load_in_4bit': True,
        'dtype': 'bfloat16',
        'qlora': True,
        'lora_rank': 16,
        'lora_alpha': 32,
        'attn_implementation': 'sdpa',
        'min_pixels': 50176,
        'max_pixels': 200704,
        'vlm_hidden_dim': 3584,
        'preserve_spatial_tokens': True,
        'forward_logits_to_keep': 1,
    }
    for key, value in expected.items():
        if bb.get(key) != value:
            raise ValueError(f'Unexpected Qwen2.5 checkpoint contract: {key}')
    targets = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
    if tuple(bb.get('lora_target_modules', ())) != targets:
        raise ValueError('Unexpected Qwen2.5 LoRA target modules')
    if not all(cfg['data'].get(key) for key in
               ('causal_routing', 'executed_action_post', 'recovery_transitions')):
        raise ValueError('Qwen2.5 causal data contract mismatch')
    return cfg


class Qwen25Assessor:
    """Unchanged trained heads from the selected PC-02 checkpoint."""

    def __init__(self, config):
        require_local_images(config['image_root'])
        import torch
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        from web_agent.data.gold_dataset import GoldDataset
        from web_agent.models.model import WebAgentModel
        from web_agent.train.gold_stages import build_processor

        if file_hash(config['checkpoint']) != QWEN25_CHECKPOINT_HASH:
            raise ValueError('Qwen2.5 checkpoint hash mismatch')
        self.torch = torch
        self.root = config['image_root']
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        saved = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)
        cfg = copy.deepcopy(_validate_saved_config(saved))
        cfg['backbone']['vlm_model'] = config['qwen25_base']
        cfg['data']['root'] = self.root
        self.cfg = cfg
        self.processor = build_processor(cfg)
        self.model = WebAgentModel(cfg)
        set_peft_model_state_dict(self.model.encoder.model, saved['lora'])
        actual = get_peft_model_state_dict(self.model.encoder.model)
        if set(actual) != set(saved['lora']):
            raise ValueError('Qwen2.5 LoRA key mismatch')
        for key, value in actual.items():
            if not torch.equal(value.cpu(), saved['lora'][key].to(dtype=value.dtype)):
                raise ValueError('Qwen2.5 LoRA restoration mismatch: ' + key)
        for attr, key in [
            ('adapter', 'adapter'),
            ('task_adapters', 'task_adapters'),
            ('failure_head', 'failure'),
            ('action_head', 'action'),
            ('memory_head', 'memory'),
            ('recovery_outcome_head', 'recovery_outcome'),
        ]:
            module = getattr(self.model, attr)
            module.load_state_dict(saved[key], strict=True)
            module.to('cuda')
        self.model.eval()
        self.model.requires_grad_(False)
        self.dataset = GoldDataset([], cfg, self.processor)
        del saved

    def batch(self, request):
        from web_agent.data.gold_dataloader import _collate_stream

        ims = images(request, self.root)
        inp = {key: request[key] for key in ('task_description', 'website_domain')}
        row = {}
        for prefix, phase, image_view, action in [
            ('pre_', 'pre', ims[:1], ''),
            ('post_', 'post', ims, request['executed_action']['type']),
        ]:
            values = self.dataset._vlm_inputs(
                inp, image_view, phase, executed_action=action,
            )
            row.update({prefix + key: value for key, value in values.items()})
        batch = {}
        for prefix in ('pre_', 'post_'):
            batch.update(_collate_stream([row], prefix, self.cfg))
        if request['phase'] == PHASES[1]:
            values = self.dataset._vlm_inputs(
                inp, ims, 'recovery',
                executed_action=request['executed_action']['type'],
            )
            row.update({'recovery_' + key: value for key, value in values.items()})
            batch.update(_collate_stream([row], 'recovery_', self.cfg))
            batch['recovery_row_indices'] = self.torch.tensor([0], dtype=self.torch.long)
        return batch

    def predict(self, request):
        from web_agent.labels import EXECUTION_OUTCOME_INV, FAILURE_TYPE_INV

        batch = self.batch(request)
        with self.torch.inference_mode():
            predictions = self.model(batch)
        if request['phase'] == PHASES[0]:
            raw = {key: predictions[key].float().cpu().tolist()
                   for key in ('outcome', 'failure_type')}
            result = {
                'outcome_label': EXECUTION_OUTCOME_INV[
                    int(predictions['outcome'].argmax(-1).item())
                ],
                'failure_type_4': FAILURE_TYPE_INV[
                    int(predictions['failure_type'].argmax(-1).item())
                ],
            }
        else:
            raw = {'recovery_outcome': predictions['recovery_outcome'].float().cpu().tolist()}
            result = {
                'outcome_label': 'SUCCESS'
                if predictions['recovery_outcome'].item() > 0 else 'FAILURE'
            }
        if not all(self.torch.isfinite(predictions[key]).all() for key in raw):
            raise ValueError('Nonfinite trained Qwen2.5 predictions')
        return {
            'status': 'valid',
            'predictions': result,
            'raw_logits': raw,
            'model_calls': 1,
        }

    def parity(self, requests, source_rows):
        """Reproduce original GoldDataset inputs and logits on disjoint development cases."""
        from web_agent.data.gold_dataloader import causal_gold_collate
        from web_agent.data.gold_dataset import GoldDataset

        by_id = {row['meta']['sample_id']: row for row in source_rows}
        report = []
        for request in requests:
            dataset = GoldDataset(
                [by_id[request['case_id']]], self.cfg, self.processor,
                trajectory_records=source_rows,
            )
            original = causal_gold_collate([dataset[0]], self.cfg)
            clean = self.batch(request)
            prefixes = ('pre_', 'post_') if request['phase'] == PHASES[0] else ('recovery_',)
            for key, value in clean.items():
                if key.startswith(prefixes) and not self.torch.equal(value, original[key]):
                    raise ValueError('Qwen2.5 processor parity failed: ' + key)
            original = {
                key: value for key, value in original.items()
                if key.startswith(('pre_', 'post_', 'recovery_'))
            }
            with self.torch.inference_mode():
                first = self.model(original)
                second = self.model(clean)
            keys = ('outcome', 'failure_type') if request['phase'] == PHASES[0] else ('recovery_outcome',)
            for key in keys:
                self.torch.testing.assert_close(first[key], second[key], rtol=1e-5, atol=1e-5)
            report.append({'case_id': request['case_id'], 'phase': request['phase'], 'status': 'PASS'})
        return report


def qwen25_generation_config(config):
    from transformers import GenerationConfig

    generation = GenerationConfig.from_pretrained(
        config['qwen25_base'], local_files_only=True,
    )
    generation.do_sample = False
    generation.max_new_tokens = 128
    generation.temperature = 0.0
    generation.top_p = 1.0
    generation.use_cache = True
    generation.return_dict_in_generate = False
    generation.output_scores = False
    if generation.pad_token_id is None:
        eos = generation.eos_token_id
        if eos is None:
            raise ValueError('No Qwen2.5 EOS/PAD token available')
        generation.pad_token_id = eos[0] if isinstance(eos, list) else eos
    return generation


class Qwen25Generator:
    """Identical prompts and tensors on the frozen base or verified trained LoRA."""

    def __init__(self, config, variant):
        if variant not in ('base', 'trained'):
            raise ValueError('Unsupported Qwen2.5 weight variant')
        require_local_images(config['image_root'])
        import torch
        from peft.tuners.lora.layer import LoraLayer

        if file_hash(config['checkpoint']) != QWEN25_CHECKPOINT_HASH:
            raise ValueError('Qwen2.5 checkpoint hash mismatch')
        self.torch = torch
        self.root = config['image_root']
        self.variant = variant
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        if variant == 'trained':
            self.owner = Qwen25Assessor(config)
            self.model = self.owner.model.encoder.model
            self.processor = self.owner.processor
        else:
            from peft import prepare_model_for_kbit_training
            from web_agent.models.encoders.vlm import VLMEncoder
            from web_agent.train.gold_stages import build_processor

            saved = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)
            cfg = copy.deepcopy(_validate_saved_config(saved))
            del saved
            cfg['backbone']['vlm_model'] = config['qwen25_base']
            cfg['backbone']['qlora'] = False
            cfg['data']['root'] = self.root
            self.processor = build_processor(cfg)
            self.owner = VLMEncoder(cfg)
            self.model = self.owner.model
            prepare_model_for_kbit_training(
                self.model, use_gradient_checkpointing=False,
            )
        self.base_signature = base_parameter_signature(self.model)
        self.base_signature_sha256 = sha(encode(self.base_signature))
        self.model.eval()
        self.model.requires_grad_(False)
        self.generation_config = qwen25_generation_config(config)
        self.layers = {
            name: module for name, module in self.model.named_modules()
            if isinstance(module, LoraLayer)
        }
        if (variant == 'base' and self.layers) or (variant == 'trained' and not self.layers):
            raise ValueError('Incorrect Qwen2.5 adapter routing')
        self.assert_adapter_state()

    def assert_adapter_state(self):
        if any(parameter.requires_grad for parameter in self.model.parameters()) or self.model.training:
            raise ValueError('Inference model must remain frozen/eval')
        for name, layer in self.layers.items():
            if layer.disable_adapters or layer.merged or list(layer.active_adapters) != ['default']:
                raise ValueError('Inactive, merged or unexpected Qwen2.5 LoRA adapter: ' + name)

    def batch(self, method, request):
        validate_request(request)
        text = prompt(method, request)
        messages = [{'role': 'user', 'content': [
            {'type': 'image'}, {'type': 'image'}, {'type': 'text', 'text': text},
        ]}]
        chat = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        batch = self.processor(
            text=[chat], images=images(request, self.root), return_tensors='pt',
        )
        required = {'input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw'}
        if not required.issubset(batch):
            raise ValueError('Qwen2.5 processor omitted required model inputs')
        if batch['image_grid_thw'].ndim != 2 or batch['image_grid_thw'].shape[0] != 2:
            raise ValueError('Expected exactly two ordered Qwen2.5 image grids')
        return text, chat, batch

    def predict(self, method, request):
        self.assert_adapter_state()
        text, chat, batch = self.batch(method, request)
        fingerprints = tensor_fingerprints(batch)
        batch = {
            key: value.to(device=self.model.device, dtype=self.torch.bfloat16)
            if value.is_floating_point() else value.to(self.model.device)
            for key, value in batch.items()
        }
        seen = set()
        calls = 0

        def active_hook(name):
            def hook(layer, args):
                nonlocal calls
                if layer.disable_adapters or layer.merged or list(layer.active_adapters) != ['default']:
                    raise ValueError('Qwen2.5 LoRA became inactive during generation: ' + name)
                seen.add(name)
                calls += 1
            return hook

        handles = [
            layer.register_forward_pre_hook(active_hook(name))
            for name, layer in self.layers.items()
        ]
        try:
            with self.torch.inference_mode():
                output = self.model.generate(
                    **batch, generation_config=self.generation_config,
                )
        finally:
            for handle in handles:
                handle.remove()
        if self.variant == 'trained' and not seen:
            raise ValueError('Trained Qwen2.5 generation did not execute LoRA layers')
        length = batch['input_ids'].shape[1]
        tokens = output[:, length:][0].tolist()
        raw = self.processor.decode(
            tokens, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        eos = self.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        return dict(
            parse_response(raw, request['phase']),
            raw_response=raw,
            prompt=text,
            serialized_chat=chat,
            input_tensor_hashes=fingerprints,
            generation_config_sha256=sha(encode(self.generation_config.to_dict())),
            generated_tokens=tokens,
            generated_token_count=len(tokens),
            limit_hit=len(tokens) == self.generation_config.max_new_tokens,
            ended_with_eos=bool(tokens and tokens[-1] in eos),
            model_calls=1,
            base_parameter_signature_sha256=self.base_signature_sha256,
            backend=QWEN25_BACKEND,
            weight_variant=self.variant,
            output_mode='generation',
            method=method,
            adapter_evidence={
                'configured_layers': len(self.layers),
                'invoked_layers': len(seen),
                'forward_calls': calls,
                'active_adapter': 'default' if self.layers else None,
                'restoration_verified': self.variant == 'trained',
            },
        )


class Qwen25RowBackend:
    def __init__(self, generator, row):
        if generator.variant != row.weight_variant:
            raise ValueError('Qwen2.5 row weight variant mismatch')
        self.generator = generator
        self.row = row

    def predict(self, request):
        if self.row.output_mode == 'generation':
            result = self.generator.predict(self.row.method, request)
        else:
            if self.row.id != 'qwen25_heads' or self.generator.variant != 'trained':
                raise ValueError('Incorrect Qwen2.5 head routing')
            result = self.generator.owner.predict(request)
            result.update(
                backend=QWEN25_BACKEND,
                weight_variant='trained',
                output_mode='heads',
                method=None,
            )
        result['row_id'] = self.row.id
        return result
