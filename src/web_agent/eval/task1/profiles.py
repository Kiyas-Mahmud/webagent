"""Explicit immutable row identities for the follow-up comparison."""
from dataclasses import asdict, dataclass

PROFILE = 'internvl_dual_v2'

@dataclass(frozen=True)
class Row:
    id: str
    method: str | None
    weight_variant: str
    output_mode: str

METHODS = ('browser_use', 'agent_s2', 'webvoyager')
ROWS = tuple(Row(f'{m}_{variant}', m, variant, 'generation')
             for variant in ('base', 'trained') for m in METHODS) + (
    Row('internvl_heads', None, 'trained', 'heads'),)
ROW_IDS = tuple(r.id for r in ROWS)
CONTRASTS = tuple((f'{m}_base', f'{m}_trained') for m in METHODS) + tuple(
    (r.id, 'internvl_heads') for r in ROWS if r.output_mode == 'generation')

def profile_spec():
    return {'name': PROFILE, 'rows': [asdict(r) for r in ROWS],
            'contrasts': [list(p) for p in CONTRASTS], 'seed': 42,
            'study': 'follow-up recorded-transition validation comparison',
            'pilot_requests_per_row': 360, 'development_requests_per_row': 18}

def row_spec(identifier):
    try:
        return next(r for r in ROWS if r.id == identifier)
    except StopIteration:
        raise ValueError('Unknown InternVL comparison row: ' + identifier) from None
