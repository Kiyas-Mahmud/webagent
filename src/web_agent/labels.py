"""Label integer maps — SINGLE SOURCE OF TRUTH (PROJECT_SPECIFICATION section 3.4).

Imported by dataset, eval, and baselines. Do NOT redefine these anywhere else.
Wrong label encoding is the #1 red-flag bug (loss won't drop) — keep it here only.

Action head supports 5 classes but SCROLL/NAVIGATE have NO training data in the
source (Mind2Web recorded only CLICK/TYPE/SELECT). The unused ids stay reserved;
do not synthetically fill them. Recovery head supports 6 classes; ABORT has no
training rows but is reserved for the production Decision Combiner.
"""

EXECUTION_OUTCOME = {"SUCCESS": 0, "FAILURE": 1}

FAILURE_TYPE = {
    "NONE": 0,
    "PERCEPTION_ERROR": 1,
    "ACTION_MISMATCH": 2,
    "LOOP_DETECTED": 3,
}

ACTION_TYPE = {
    "CLICK": 0,
    "TYPE": 1,
    "SELECT": 2,
    "SCROLL": 3,     # synthetic: reserved; gold v12: has data
    "NAVIGATE": 4,   # reserved (no data in either source)
    "PRESS_KEY": 5,  # gold v12 keyboard action (Enter/Tab/...); additive, synthetic has none
}

RECOVERY_STRATEGY = {
    "NONE": 0,
    "RETRY": 1,
    "REPLAN": 2,
    "BACKTRACK": 3,
    "ALTERNATIVE_TARGET": 4,
    "ABORT": 5,  # no training data — reserved for Decision Combiner
}

MEMORY_FLAG = {False: 0, True: 1}

# Number of output classes per head (drives the Linear out-features).
NUM_OUTCOME = len(EXECUTION_OUTCOME)        # 2
NUM_FAILURE_TYPE = len(FAILURE_TYPE)        # 4
NUM_ACTION_TYPE = len(ACTION_TYPE)          # 5
NUM_RECOVERY = len(RECOVERY_STRATEGY)       # 6

# Reverse maps for decoding predictions back to string labels.
EXECUTION_OUTCOME_INV = {v: k for k, v in EXECUTION_OUTCOME.items()}
FAILURE_TYPE_INV = {v: k for k, v in FAILURE_TYPE.items()}
ACTION_TYPE_INV = {v: k for k, v in ACTION_TYPE.items()}
RECOVERY_STRATEGY_INV = {v: k for k, v in RECOVERY_STRATEGY.items()}
