"""Reject mixing embedding spaces, even when their dimensions are identical."""
import json
from pathlib import Path
from web_agent.eval.task1.core import file_hash


def verify_memory_query_identity(store_directory, *, expected_manifest_sha256,
                                 query_checkpoint_sha256, query_dimension):
    root = Path(store_directory)
    if file_hash(root / 'manifest.json') != expected_manifest_sha256:
        raise ValueError('Memory manifest identity changed')
    manifest = json.loads((root / 'manifest.json').read_text())
    if query_checkpoint_sha256 != manifest['checkpoint_sha256']:
        raise ValueError('MEMORY_EMBEDDING_SPACE_MISMATCH: use the original store encoder; equal dimensions are insufficient')
    if query_dimension != manifest['dimension']:
        raise ValueError('Memory query dimension mismatch')
    for name, digest in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != digest:
            raise ValueError('Memory artifact identity changed: ' + name)
    return {'status': 'ASSET_IDENTITY_PASS', 'checkpoint_sha256': query_checkpoint_sha256,
            'dimension': query_dimension, 'count': manifest['count'],
            'admission_threshold': manifest['admission_threshold'], 'write_enabled': False,
            'live_query_parity_verified': False}
