"""Reuse the registered PC-01 loaders; no model or decoding changes."""

def load_models(config):
    from pathlib import Path
    import json
    from web_agent.runtime.checkpoint_inference import ValidationSelectedCheckpoint
    from web_agent.runtime.qwen2vl_pc01 import PC01RuntimeArtifacts
    export=Path(config['asset_paths']['export'])
    base=Path(config['asset_paths']['base_snapshot'])
    full=Path(config['full_run_directory'])
    m=json.loads((export/'pc01_export_manifest.json').read_text())
    report=json.loads((full/'report.json').read_text())
    selection=ValidationSelectedCheckpoint(manifest_id='pc01-miniwob-one-action-feasibility',model_seed=42,checkpoint_path=full/'checkpoints/Y_QWEN2VL_2B_GOLD_V2_8_DGX_FULL_SEED42/best_e6_outcome-mcc0.624.ckpt',selected_checkpoint_sha256=m['checkpoint_sha256'],resolved_config_sha256=m['resolved_config_payload_sha256'],processor_contract_sha256=m['processor_contract_sha256'],validation_rows_read=int(report['val_rows']))
    a=PC01RuntimeArtifacts(resolved_config_path=export/'resolved_config.json',processor_contract_path=export/'processor_contract.json',processor_source=base,e0_resolved_config_path=export/'e0_resolved_config.json',e0_processor_contract_path=export/'e0_processor_contract.json',e0_backbone_path=base,export_manifest_path=export/'pc01_export_manifest.json',training_action_value_evidence_path=export/'training_action_value_evidence.json')
    
    import hashlib
    from web_agent.runtime.qwen2vl_pc01 import load_pc01_runtime_backends, E0_ACTION_PROMPT, E0_PARSER_ID, E0_PARSER_VERSION, PARAMETER_PROVIDER_PROMPT, REGISTERED_GENERATION_KWARGS
    from web_agent.runtime.checkpoint_inference import ValidationSelectedBackbone
    from web_agent.runtime.action_parameters import build_registered_hybrid_parameter_provider
    e0=ValidationSelectedBackbone(manifest_id='pc01-miniwob-parameter-feasibility',backbone_id='Qwen/Qwen2-VL-2B-Instruct',backbone_revision=m['model_revision'],backbone_path=base,backbone_sha256=m['base_snapshot_directory_payload_sha256'],resolved_config_sha256=m['files']['e0_resolved_config.json']['sha256'],processor_contract_sha256=m['processor_contract_sha256'],base_prompt_sha256=hashlib.sha256(E0_ACTION_PROMPT.encode()).hexdigest(),parser_id=E0_PARSER_ID,parser_version=E0_PARSER_VERSION,validation_rows_read=int(report['val_rows']))
    
    bundle=load_pc01_runtime_backends(selected=selection,e0=e0,artifacts=a)
    provider=build_registered_hybrid_parameter_provider(fallback_resolver=bundle.parameter_fallback_resolver,fallback_policy_id='pc01-unadapted-base-parameters',fallback_policy_version='v1',prompt_text=PARAMETER_PROVIDER_PROMPT,decoding_parameters=REGISTERED_GENERATION_KWARGS)
    
    return bundle, provider, m
