# Existing-data improvement audit result

## Run identity

- Kaggle kernel: `kiyasmahmud/web-gold-existing-data-improvement`
- Accepted kernel version: 3
- Execution mode: CPU
- Dataset source: attached `kiyasmahmud/web-gold-40k`
- Splits read: train and validation
- Test rows read: 0
- Source records mutated: false
- Source images copied or extracted by the audit: false
- Review-package SHA-256:
  `48534e7e815135b5530514887835be2b046da1efc1416c2d3d7124dc425e1770`

The ZIP passed an integrity read of all 11 members.

## BBox result

| Split | Records | BBox-labelled | Valid | Invalid/masked |
| --- | ---: | ---: | ---: | ---: |
| Train | 23,499 | 8,761 | 6,719 | 2,042 |
| Validation | 7,861 | 2,841 | 2,349 | 492 |
| Total | 31,360 | 11,602 | 9,068 | 2,534 |

Invalid bbox share among bbox-labelled train/validation rows:
`2,534 / 11,602 = 21.84%`.

The mask manifest and bbox review CSV contain the same 2,534 unique
`split:sample_id` keys. Every manifest row has `bbox_mask=0`; there are no test
rows and no automatic corrections.

### Geometry evidence

| Reason | Rows |
| --- | ---: |
| Bottom boundary overflow | 2,228 |
| Y origin outside image | 2,136 |
| Negative origin | 177 |
| Right boundary overflow | 132 |
| X origin outside image | 89 |
| Non-positive size | 6 |

Rows may have multiple reasons. The dominant combination is
`bottom_boundary_overflow|y_origin_outside` on 2,132 rows.

The invalid labels have only 289 distinct bbox values across 2,534 rows.
Repeated examples include y coordinates of 3,475, 5,675, 9,770, 10,214, and
11,013 against 720-pixel viewport images. This is systematic document/page
geometry paired with viewport screenshots, not isolated rounding noise.

Invalid rows by action:

| Action | Invalid rows | Share of invalid |
| --- | ---: | ---: |
| SELECT | 1,618 | 63.85% |
| TYPE | 552 | 21.78% |
| CLICK | 364 | 14.36% |

Every invalid row has `action_coordinates=null`. The exported schema also has
no verified document-to-viewport scroll offset. Therefore neither point
supervision nor deterministic coordinate conversion is available. Masking is
the only defensible automatic action. Any corrected bbox requires direct replay
or screenshot evidence and independent confirmation.

## Weak-class coverage result

Validation support:

| Target class | Validation rows |
| --- | ---: |
| `ACTION:SCROLL` | 1,293 |
| `ACTION:SELECT` | 1,292 |
| `ACTION:NAVIGATE` | 1,499 |
| `FAILURE:LOOP_DETECTED` | 240 |
| `RECOVERY:BACKTRACK` | 200 |
| `RECOVERY:RETRY` | 0 |
| `RECOVERY:ABORT` | 0 |

The first five classes have enough existing examples for label-quality review
before any collection decision. `RETRY` and `ABORT` are absent from both train
and validation. Real targeted collection is required only if the registered
paper scope claims they are learned strategies.

All existing attempted recoveries have a causal next-step transition:

- Train: 5,518 attempted, 5,518 proper transitions, coverage 1.0.
- Validation: 1,858 attempted, 1,858 proper transitions, coverage 1.0.

## Reviewer package

- BBox queue: 2,534 rows from 2,336 tasks.
- Weak-class queue: 1,977 rows from 977 complete tasks.
- Weak-class focus rows: 1,674; the remaining 303 rows are trajectory context.
- BBox assignments: A 1,163, B 1,128, A+B 243.
- Weak-task assignments: A 418, B 453, A+B 106.
- Reviewer decisions currently filled: 0.
- Proposed bbox corrections currently filled: 0.

The package includes correct before/after action montages and causal
failure-state/post-recovery-state montages for `BACKTRACK`.

## Decision

The automated phase passes and the 2,534 invalid bbox labels are safely
documented and masked. This does not mean the underlying labels were corrected.

The next permitted work is human review:

1. Reviewer A and Reviewer B complete their assigned queues.
2. Every correction, rejection, and ambiguity receives a second decision.
3. Corrections are applied only through the source/export ledger.
4. Counts, split integrity, leakage gates, and the bbox audit are rerun.
5. A controlled 5k mini experiment is permitted only after review adjudication.

Do not start headline/full training from this audit result alone.
