"""Persistent local GPU worker, separate from the native Browser Use environment."""
import base64
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
import traceback

from web_agent.eval.task1.core import file_hash
from web_agent.eval.task2.assessment import ExecutedTransition, Observation, InternVLTransitionAssessor


def transition_from_dict(value):
    value = dict(value)
    for key in ('before','after'):value[key] = Observation(**value[key])
    return ExecutedTransition(**value)


def native_messages(request):
    from PIL import Image
    messages, images = [], []
    for message in request['messages']:
        role = message['role']
        if role not in ('system','user','assistant'):raise ValueError('Unsupported native message role')
        content = message.get('content')
        parts = [{'type':'text','text':content}] if isinstance(content,str) else list(content or [])
        normalized = []
        for part in parts:
            if part['type'] == 'text':normalized.append({'type':'text','text':part['text']})
            elif part['type'] == 'image_url':
                url = part['image_url']['url']
                if not url.startswith('data:image/') or ';base64,' not in url:
                    raise ValueError('Only inline local screenshots allowed')
                data = base64.b64decode(url.split(';base64,',1)[1], validate=True)
                with Image.open(io.BytesIO(data)) as image:images.append(image.convert('RGB'))
                normalized.append({'type':'image'})
            else:raise ValueError('Unsupported content part: '+part['type'])
        if message.get('tool_calls'):
            normalized.append({'type':'text','text':json.dumps({'tool_calls':message['tool_calls']})})
        messages.append({'role':role,'content':normalized})
    # Preserve native prompts/history. Only describe the local structured-output interface.
    contract = 'Return exactly one JSON object matching this response schema. No Markdown. Choose exactly one immediate action.\n'+json.dumps(request['output_schema'])
    messages.append({'role':'user','content':[{'type':'text','text':contract}]})
    if request.get('previous_proposal_rejection'):
        messages.append({'role':'user','content':[{'type':'text','text':
            'PREVIOUS PROPOSAL WAS NOT EXECUTED. Interface feedback:\n'+json.dumps(request['previous_proposal_rejection'])}]})
    if request.get('advice'):
        messages.append({'role':'user','content':[{'type':'text','text':'ADVISORY TRAINED ASSESSMENT (not evaluator truth):\n'+json.dumps(request['advice'])}]})
    return messages, images


def load_memory_model():
    from web_agent.runtime.checkpoint_inference import ValidationSelectedCheckpoint
    from web_agent.runtime.qwen2vl_pc01 import PC01RuntimeArtifacts, _load_selected_model
    from web_agent.data.gold_dataset import GoldDataset
    from web_agent.models.encoders.vlm_contract import get_vlm_contract
    from web_agent.memory.label_backed_store import LabelBackedMemoryStore
    from web_agent.memory.label_experience import LabelExperienceMaterial
    cfg = json.loads(Path('configs/eval/table2/miniwob_hybrid_dev_v3.json').read_text())
    export = Path(cfg['asset_paths']['export']); base = Path(cfg['asset_paths']['base_snapshot'])
    m = json.loads((export/'pc01_export_manifest.json').read_text())
    checkpoint = Path(cfg['asset_paths']['checkpoint'])
    if file_hash(checkpoint) != m['checkpoint_sha256']:raise ValueError('Memory encoder checkpoint changed')
    report = json.loads((Path(cfg['full_run_directory'])/'report.json').read_text())
    selection = ValidationSelectedCheckpoint(manifest_id='task2-original-memory-query',model_seed=42,
        checkpoint_path=checkpoint,selected_checkpoint_sha256=m['checkpoint_sha256'],
        resolved_config_sha256=m['resolved_config_payload_sha256'],processor_contract_sha256=m['processor_contract_sha256'],
        validation_rows_read=int(report['val_rows']))
    artifacts = PC01RuntimeArtifacts(resolved_config_path=export/'resolved_config.json',
        processor_contract_path=export/'processor_contract.json',processor_source=base,
        e0_resolved_config_path=export/'e0_resolved_config.json',e0_processor_contract_path=export/'e0_processor_contract.json',
        e0_backbone_path=base,export_manifest_path=export/'pc01_export_manifest.json',
        training_action_value_evidence_path=export/'training_action_value_evidence.json')
    model, processor, torch, config, contract = _load_selected_model(selection=selection,artifacts=artifacts)
    dataset = GoldDataset.__new__(GoldDataset)
    dataset.cfg=config; dataset.processor=processor; dataset.vlm_contract=get_vlm_contract(config)
    store = LabelBackedMemoryStore(cfg['asset_paths']['memory'])
    if store.checkpoint_sha256 != m['checkpoint_sha256']:raise ValueError('Query/store space mismatch')
    material = LabelExperienceMaterial(cfg['memory_material'],store=store,include_source_context=True)
    return model,dataset,store,material


def main():
    import torch
    from PIL import Image
    from web_agent.eval.task1.backends import InternVLGenerator, tensor_fingerprints
    config = json.loads(Path('configs/eval/task1/internvl_dual_lab_paths.json').read_text())
    actor = None; assessor = None; memory = None
    for line in sys.stdin:
        try:
            q = json.loads(line); op=q['op']
            if op == 'initialize':
                actor=InternVLGenerator(config,'base');actor.generation_config.max_new_tokens=512
                assessor=InternVLTransitionAssessor.load(config)
                memory=load_memory_model()
                value={'status':'READY','actor':'InternVL-base','assessor':'InternVL-trained-heads',
                       'memory_query_checkpoint':memory[2].checkpoint_sha256,'model_calls':0}
            elif op == 'reset':
                torch.manual_seed(42); torch.cuda.manual_seed_all(42)
                value = {'seed':42}
            elif op == 'generate':
                if actor is None:
                    actor = InternVLGenerator(config,'base')
                    actor.generation_config.max_new_tokens=512
                actor.assert_adapter_state()
                messages,images = native_messages(q['request'])
                chat = actor.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                batch = actor.processor(text=[chat], images=images or None,
                    images_kwargs={'crop_to_patches':False},return_tensors='pt')
                if images and (batch['pixel_values'].shape[0] != len(images) or tuple(batch['pixel_values'].shape[-2:]) != (448,448)):
                    raise ValueError('Native screenshot patch contract mismatch')
                hashes = tensor_fingerprints(batch)
                batch={k:v.to(device=actor.model.device,dtype=torch.bfloat16) if v.is_floating_point() else v.to(actor.model.device) for k,v in batch.items()}
                with torch.inference_mode():output=actor.model.generate(**batch,generation_config=actor.generation_config)
                tokens=output[0,batch['input_ids'].shape[1]:].tolist()
                value={'raw_response':actor.processor.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False),
                       'generated_tokens':tokens,'token_count':len(tokens),'limit_hit':len(tokens)==512,
                       'input_tensor_hashes':hashes,'image_count':len(images),'model_calls':1,
                       'backend':'InternVL3.5-8B-HF','weight_variant':'base','generation_config':actor.generation_config.to_dict()}
            elif op in ('assess','parity'):
                if assessor is None:assessor=InternVLTransitionAssessor.load(config)
                assessor.backend.root=q['image_root']
                transition=transition_from_dict(q['transition'])
                value=assessor.assess(transition)
                if op=='parity':
                    original=assessor.backend.predict(transition.head_request(Path(q['image_root'])))
                    for key,logits in original['raw_logits'].items():
                        torch.testing.assert_close(torch.tensor(logits),torch.tensor(value['raw_logits'][key]),rtol=1e-5,atol=1e-5)
                    value={'status':'PASS','original_logits':original['raw_logits'],'bridge_logits':value['raw_logits'],'model_calls':2}
            elif op=='memory':
                if memory is None:memory=load_memory_model()
                model,dataset,store,material=memory
                transition=transition_from_dict(q['transition']); root=Path(q['image_root'])
                transition.validate(root)
                ims=[]
                for obs in (transition.before,transition.after):
                    with Image.open(root/obs.image) as im:ims.append(im.convert('RGB'))
                encoded=dataset._vlm_inputs({'task_description':transition.task_description,'website_domain':transition.website_domain},ims,'post',executed_action=transition.action_type,action_value='')
                batch={}
                for key,v in encoded.items():
                    if key=='image_count':batch['post_image_counts']=v
                    else:batch['post_'+key]=v.unsqueeze(0) if key in dataset.vlm_contract.sequence_input_keys else v
                with torch.inference_mode():vector=model.memory_embedding(batch)[0].float().cpu().numpy()
                hits=store.query(vector,excluded_task_ids=q.get('excluded_task_ids',()))
                from web_agent.memory.context_applicability import context_exclusion
                examples=[]; candidates=[]
                for hit in hits:
                    item=material.for_hit(hit).to_dict()
                    reason='BELOW_THRESHOLD' if not hit.admitted else context_exclusion(item, q['applicability_transition'])
                    candidates.append(dict(asdict(hit),exclusion_reason=reason))
                    if not reason:examples.append(item)
                value={'candidates':candidates,'examples':examples,'model_calls':1,
                       'query_checkpoint_sha256':store.checkpoint_sha256,'threshold':store.threshold,
                       'query_tensor_hashes':tensor_fingerprints(batch),'write_enabled':False}
            elif op=='close':
                break
            else:raise ValueError('Unknown model operation')
            print('TASK2:'+json.dumps({'ok':True,'value':value},allow_nan=False),flush=True)
        except Exception as exc:
            print('TASK2:'+json.dumps({'ok':False,'error':str(exc),'traceback':traceback.format_exc()}),flush=True)


if __name__=='__main__':main()
