"""Evaluation on 3 test splits + metrics (Failure-F1, recovery SR, etc.)."""

import hashlib
from pathlib import Path


PROCESS_BROKER_IMPORT_SOURCE_SHA256 = hashlib.sha256(
    Path(__file__).resolve().read_bytes()
).hexdigest()
