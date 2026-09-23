"""Explicit immutable rows for the selected Qwen2.5-VL-7B follow-up."""
from dataclasses import asdict, dataclass

PROFILE = 'qwen25_dual_v1'


@dataclass(frozen=True)
class Row:
    id: str
    method: str | None
    weight_variant: str
    output_mode: str


METHODS = ('browser_use', 'agent_s2', 'webvoyager')
ROWS = tuple(
    Row(f'{method}_{variant}', method, variant, 'generation')
    for variant in ('base', 'trained') for method in METHODS
) + (Row('qwen25_heads', None, 'trained', 'heads'),)
ROW_IDS = tuple(row.id for row in ROWS)
CONTRASTS = tuple((f'{method}_base', f'{method}_trained') for method in METHODS) + tuple(
    (row.id, 'qwen25_heads') for row in ROWS if row.output_mode == 'generation'
)


def profile_spec():
    return {
        'name': PROFILE,
        'rows': [asdict(row) for row in ROWS],
        'contrasts': [list(pair) for pair in CONTRASTS],
        'seed': 42,
        'study': 'selected-model follow-up recorded-transition validation comparison',
        'pilot_requests_per_row': 360,
        'development_requests_per_row': 18,
    }


def row_spec(identifier):
    try:
        return next(row for row in ROWS if row.id == identifier)
    except StopIteration:
        raise ValueError('Unknown Qwen2.5 comparison row: ' + identifier) from None
