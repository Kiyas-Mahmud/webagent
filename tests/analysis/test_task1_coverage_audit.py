import importlib.util
from pathlib import Path
import pytest
from web_agent.eval.task1.core import parse_response
s=importlib.util.spec_from_file_location('audit',Path('scripts/analysis/task1_coverage_audit.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
P='interaction_assessment'

def test_missing_fields_are_not_broken_json_or_repaired():
 raw='{"outcome_label":"ABSTAIN"}'
 x=m.inspect_response(raw,P)
 assert x['syntax']=='json' and set(x['defects'])=={'missing:reason','missing:failure_type_4'}
 assert parse_response(raw,P)['status']=='parse_error'

def test_abstention_bypasses_invalid_category_in_frozen_parser():
 raw='{"outcome_label":"ABSTAIN","reason":"unknown","failure_type_4":"ABSTAIN"}'
 assert m.inspect_response(raw,P)['defects']==['invalid_failure_label']
 assert parse_response(raw,P)=={'status':'abstention','predictions':{}}

def test_decisive_output_does_not_bypass_invalid_category():
 raw='{"outcome_label":"FAILURE","reason":"unknown","failure_type_4":"ABSTAIN"}'
 assert m.inspect_response(raw,P)['defects']==['invalid_failure_label']
 assert parse_response(raw,P)['status']=='parse_error'

def test_fenced_json_is_never_repaired():
 raw='```json\n{"outcome_label":"ABSTAIN","reason":"unknown"}\n```'
 assert m.inspect_response(raw,'recovery_assessment')['defects']==['not_bare_json']
 assert parse_response(raw,'recovery_assessment')['status']=='parse_error'

def test_duplicate_keys_preserved_as_contract_failure():
 raw='{"outcome_label":"SUCCESS","outcome_label":"FAILURE","reason":"evidence"}'
 assert 'duplicate_keys' in m.inspect_response(raw,'recovery_assessment')['defects']
 assert parse_response(raw,'recovery_assessment')['status']=='parse_error'

def test_no_failure_category_required_for_recovery():
 raw='{"outcome_label":"ABSTAIN","reason":"unknown"}'
 assert m.inspect_response(raw,'recovery_assessment')['defects']==[]
 assert parse_response(raw,'recovery_assessment')['status']=='abstention'
