"""Replay saved training vectors through H3; no model weights or new embeddings.

Uses a scripted causal fixture and copies existing vectors as query outputs.
This verifies retrieval/wiring only, not live memory usefulness.
"""
import hashlib
import json
from pathlib import Path
import sys
from dataclasses import replace
from threading import Lock
from types import SimpleNamespace as NS

import numpy as np
import torch

from web_agent.memory.hybrid_runtime import build_hybrid_memory
from web_agent.runtime.model_calls import ModelCallLedger, activate_model_call_ledger, deactivate_model_call_ledger
from tests.table2.test_hybrid_memory import inputs


def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
    store_path=Path('/home/aiub/kiyas/table2-evidence/p4-local-embeddings-v1')
    material_path=Path('/home/aiub/kiyas/table2-evidence/p4-local-label-memory-v1/memory-items.jsonl')
    expected_manifest='4b577b79e2f8e04424572aa25223a81fa15a8f1f3cb483ef181dda982aa072d6'
    expected_material='f06beefbd503ad68e89f26bc46fe2bb51a1ea32c87632f275118275efe0d31ab'
    manifest=json.loads((store_path/'manifest.json').read_text())
    def hashes():
        return {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [store_path/'manifest.json',material_path]+[store_path/name for name in manifest['files']]}
    before=hashes()
    vectors=np.load(store_path/'embeddings.npy',mmap_mode='r')
    current=[0]
    runtime=NS(checkpoint_sha256=manifest['checkpoint_sha256'],lock=Lock(),torch=torch,
        batch=NS(transition=lambda *args:{}),model=NS(memory_embedding=lambda batch:
            torch.from_numpy(np.array(vectors[current[0]],copy=True)).unsqueeze(0)))
    adapter=build_hybrid_memory(system='H3',episode_id='episode',runtime=runtime,store_directory=store_path,
        material_path=material_path,evidence_dir=out/'receipts',expected_manifest_sha256=expected_manifest,
        expected_material_sha256=expected_material)
    assert adapter.store._vectors.flags.writeable is False
    rows=json.loads((store_path/'index.json').read_text())
    events=[];ledger=ModelCallLedger('episode',sink=events.append);token=activate_model_call_ledger(ledger)
    replay=[]
    try:
        for number,index in enumerate((0,987,1973),1):
            current[0]=index
            query,shadow,request=inputs(number)
            query=replace(query,checkpoint_sha256=manifest['checkpoint_sha256'])
            request=replace(request,checkpoint_sha256=manifest['checkpoint_sha256'])
            result=adapter.apply_post_failure(query=query,shadow_decision=shadow,rng=None,embedding_request=request)
            # Independent stable cosine top-three replay of the returned query.
            vector=np.asarray(result.query_result.normalized_query_embedding,dtype=np.float32)
            scores=np.clip(vectors@(vector/np.linalg.norm(vector)),-1,1)
            allowed=np.array([i for i,row in enumerate(rows) if row['source_task']!=query.task_id])
            order=allowed[np.argsort(-scores[allowed],kind='stable')][:3]
            assert result.query_result.candidate_ids==tuple(rows[i]['memory_id'] for i in order)
            assert np.allclose(result.query_result.scores,scores[order],rtol=0,atol=1e-7)
            hits=adapter.store.query(vector,excluded_task_ids={query.task_id})
            assert [h.admitted for h in hits]==[bool(scores[i]>=manifest['admission_threshold']) for i in order]
            for item in result.final_decision.memory_experiences:
                assert item.recovery_action_value is None and item.reflection is None
            replay.append({'source_vector_index':index,'candidate_ids':list(result.query_result.candidate_ids),
                'scores':list(result.query_result.scores),'context_ids':list(result.query_result.context_candidate_ids),
                'shadow_preserved':result.shadow_decision.sha256==shadow.sha256,
                'empty_admission_preserves_object':result.final_decision is shadow if not result.final_decision.memory_experiences else None})
    finally:deactivate_model_call_ledger(token)
    records=adapter.recovery_context_material._records
    assert len(records)==1974
    assert all(x.recovery_action_value is None and x.reflection is None for x in records.values())
    after=hashes();assert before==after
    result={'status':'PASS','evidence_type':'SAVED_TRAIN_VECTOR_REPLAY','count':len(records),'dimension':768,
        'threshold':adapter.store.threshold,'actual_model_inferences':0,'new_embeddings':0,
        'scripted_embedding_dispatches':ledger.count,'live_episodes':0,'memory_writes':0,
        'missing_corrective_values':1974,'missing_reflections':1974,'source_files_unchanged':True,
        'before_sha256':before,'after_sha256':after,'replay':replay,'dispatches':events}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: frozen 1974-vector H3 replay; unchanged files; no new embeddings or actual inference')


if __name__=='__main__':main()
