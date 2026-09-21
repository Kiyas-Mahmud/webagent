import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from .test_assessment import rows, Fake
from web_agent.eval.task1.core import make_views,execute,read,request_key,write_new,encode,sha
from web_agent.eval.task1.profiles import ROWS,ROW_IDS,CONTRASTS,row_spec,PROFILE
from web_agent.eval.task1.backends import InternVLGenerator,InternVLRowBackend,tensor_fingerprints
from web_agent.eval.task1.dual import check_generation_parity,check_record_routes,existing_records
from web_agent.eval.task1.reporting import score


def test_seven_explicit_rows_and_nine_contrasts():
    assert len(ROWS)==len(set(ROW_IDS))==7
    assert len(CONTRASTS)==9
    assert [r.id for r in ROWS if r.output_mode=='heads']==['internvl_heads']
    assert not any('qwen' in str(r).lower() for r in ROWS)
    with pytest.raises(ValueError):row_spec('browser_use')


def test_cli_profile_dispatch_cannot_default_to_qwen(monkeypatch):
    from web_agent.eval.task1 import cli,dual
    seen=[]
    monkeypatch.setattr(dual,'main',lambda argv:seen.append(argv))
    argv=['check','--profile',PROFILE]
    cli.main(argv)
    assert seen==[argv]


def test_wrong_variant_rejected_before_prediction():
    with pytest.raises(ValueError,match='variant'):
        InternVLRowBackend(SimpleNamespace(variant='base'),row_spec('browser_use_trained'))


def test_fingerprints_bind_values_shape_dtype_and_order():
    a=torch.tensor([[1,2],[3,4]],dtype=torch.int64)
    x=tensor_fingerprints({'a':a})
    assert x!=tensor_fingerprints({'a':a.flip(0)})
    assert x!=tensor_fingerprints({'a':a.float()})
    assert x!=tensor_fingerprints({'a':a.reshape(1,4)})


def fake_generator(variant='trained', invoke=True):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.layer=torch.nn.Linear(2,2)
            self.layer.disable_adapters=False;self.layer.merged=False;self.layer.active_adapters=['default']
            self.device=torch.device('cpu')
        def generate(self,input_ids,**kwargs):
            if invoke:self.layer(torch.ones(1,2))
            return torch.cat([input_ids,torch.tensor([[3,4]])],dim=1)
    g=InternVLGenerator.__new__(InternVLGenerator);g.torch=torch;g.variant=variant
    g.model=Model().eval().requires_grad_(False);g.base_signature_sha256='test-signature'
    g.layers={'layer':g.model.layer} if variant=='trained' else {}
    g.generation_config=SimpleNamespace(max_new_tokens=128,eos_token_id=4,to_dict=lambda:{'max_new_tokens':128})
    g.processor=SimpleNamespace(decode=lambda *args,**kwargs:'{"outcome_label":"SUCCESS","reason":"observed","failure_type_4":"NONE"}')
    g.batch=lambda *args:('same prompt','same chat',{'input_ids':torch.tensor([[1,2]]),'pixel_values':torch.ones(2,3,2,2)})
    return g


def test_trained_generate_observes_adapter_hooks_and_preserves_tokens(rows):
    g=fake_generator();r=make_views(rows,['t:0'])[0][0]
    result=InternVLRowBackend(g,row_spec('browser_use_trained')).predict(r)
    assert result['adapter_evidence']['invoked_layers']==1
    assert result['generated_tokens']==[3,4]
    assert result['generated_token_count']==2 and result['ended_with_eos'] and not result['limit_hit']
    assert result['row_id']=='browser_use_trained'
    assert not g.model.layer._forward_pre_hooks


def test_trained_generate_without_adapter_execution_fails(rows):
    with pytest.raises(ValueError,match='did not execute'):
        fake_generator(invoke=False).predict('browser_use',make_views(rows,['t:0'])[0][0])


@pytest.mark.parametrize('field,value',[('disable_adapters',True),('merged',True),('active_adapters',[])])
def test_adapter_disabled_or_merged_is_not_silent_fallback(field,value):
    g=fake_generator();setattr(g.model.layer,field,value)
    with pytest.raises(ValueError,match='LoRA'):g.assert_adapter_state()


def test_freeze_required_for_generation():
    g=fake_generator();g.model.train()
    with pytest.raises(ValueError,match='frozen/eval'):g.assert_adapter_state()


def test_infrastructure_resume_stops_before_any_new_call(rows,tmp_path):
    requests=make_views(rows,['t:0'])[0]
    b=Fake(fail=True)
    with pytest.raises(RuntimeError):execute(requests,'internvl_heads',b,tmp_path,{})
    before=b.calls
    with pytest.raises(RuntimeError,match='Review required'):execute(requests,'internvl_heads',b,tmp_path,{})
    assert b.calls==before


def test_seven_row_scoring_does_not_import_old_rows(rows):
    requests,targets,_=make_views(rows,['t:0'])
    records={s:{request_key(r):{'status':'valid','predictions':targets[request_key(r)],'model_calls':1,'latency_seconds':1} for r in requests} for s in ROW_IDS}
    report=score(requests,targets,records,systems=ROW_IDS,contrasts=CONTRASTS)
    assert report['status']=='complete'
    assert set(report['systems']['interaction_assessment'])==set(ROW_IDS)
    assert len(report['paired_comparisons'])==27
    del records['internvl_heads'][request_key(requests[0])]
    assert score(requests,targets,records,systems=ROW_IDS,contrasts=CONTRASTS)['status']=='incomplete'


def test_empty_valid_denominators_not_reported_as_zero(rows):
    requests,targets,_=make_views(rows,['t:0'])
    records={s:{request_key(r):{'status':'parse_error','predictions':{}} for r in requests} for s in ROW_IDS}
    report=score(requests,targets,records,systems=ROW_IDS,contrasts=CONTRASTS)
    m=report['systems']['interaction_assessment']['browser_use_base']['endpoints']['outcome_label']
    assert m['mcc'] is None and m['all_case_accuracy']==0


def test_identical_outputs_allowed_but_input_or_generation_changes_fail(rows):
    requests=make_views(rows,['t:0'])[0];identity={'effective_generation':{'max_new_tokens':128}}
    record={'input_tensor_hashes':{'x':'same'},'prompt':'same','serialized_chat':'same','base_parameter_signature_sha256':'same',
            'generation_config_sha256':sha(encode(identity['effective_generation']))}
    records={s:{request_key(r):copy.deepcopy(record) for r in requests} for s in ROW_IDS}
    check_generation_parity(records,requests,identity)
    records['agent_s2_trained'][request_key(requests[0])]['input_tensor_hashes']={'x':'different'}
    with pytest.raises(ValueError,match='input parity'):check_generation_parity(records,requests,identity)


def test_two_image_processor_call_and_order(monkeypatch,rows):
    import web_agent.eval.task1.backends as backend
    g=InternVLGenerator.__new__(InternVLGenerator);g.root='/unused';seen={}
    first,second=object(),object()
    monkeypatch.setattr(backend,'images',lambda request,root:[first,second])
    class Processor:
        def apply_chat_template(self,messages,**kwargs):
            seen['messages']=messages;return 'chat'
        def __call__(self,**kwargs):
            seen.update(kwargs);return {'input_ids':torch.tensor([[1]]),'pixel_values':torch.zeros(2,3,448,448)}
    g.processor=Processor()
    g.batch('browser_use',make_views(rows,['t:0'])[0][0])
    assert seen['images']==[first,second]
    assert seen['images_kwargs']=={'crop_to_patches':False}
    assert [c['type'] for c in seen['messages'][0]['content']]==['image','image','text']


def test_peft_base_preparation_matches_trained_frozen_precision():
    from peft import prepare_model_for_kbit_training
    layer = torch.nn.Linear(4,4).to(torch.bfloat16)
    prepare_model_for_kbit_training(layer,use_gradient_checkpointing=False)
    assert all(p.dtype==torch.float32 and not p.requires_grad for p in layer.parameters())


def test_signature_compares_base_and_lora_wrapper_and_detects_precision():
    from web_agent.eval.task1.backends import base_parameter_signature
    plain=torch.nn.Sequential(torch.nn.Linear(2,2))
    wrapped=torch.nn.Sequential(torch.nn.Module())
    wrapped[0].base_layer=torch.nn.Linear(2,2)
    wrapped[0].lora_A=torch.nn.Linear(2,1,bias=False)
    assert base_parameter_signature(plain)==base_parameter_signature(wrapped)
    plain.to(torch.bfloat16)
    assert base_parameter_signature(plain)!=base_parameter_signature(wrapped)


def test_pad_resolution_is_frozen_explicitly(monkeypatch):
    import transformers
    from web_agent.eval.task1.backends import internvl_generation_config
    monkeypatch.setattr(transformers.GenerationConfig,'from_pretrained',lambda *a,**k:transformers.GenerationConfig(eos_token_id=151645))
    config=internvl_generation_config({'internvl_base':'unused'})
    assert config.pad_token_id==151645 and config.use_cache and config.max_new_tokens==128
