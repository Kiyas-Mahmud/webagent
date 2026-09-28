# Real-website task suite for the agent (`web-tasks-v1`)

2026-09-28. Development of the agent system; no with/without comparison yet.

## Why real websites

P1 (failure/recovery heads) is trained on the user's own dataset
(`web_agent_gold_v16_500_domain_40k`: 305 real domains, 1280×720 screenshots).
On MiniWoB it labelled 100% of 122 live steps FAILURE. The runtime path is
correct — 240 validation rows through the exact Task 2 bridge give accuracy
0.825 / MCC 0.647 — but in the training data SUCCESS steps change a median 36%
of pixels versus 0.19% for MiniWoB widgets. On real pages P1 behaves as trained.

## Suite

`configs/eval/task2/web_tasks_v1.json` — 20 tasks on 15 sites taken from the
dataset's domains (Wikipedia, Wiktionary, arXiv, Debian, Creative Commons,
GOV.UK, TED, jQuery, EPA, IETF, Stanford, Yale, europa.eu).

- Each task: start URL, goal text, allowed domains, completion rule
  `url_contains` (checked on the current URL after every step, never shown to
  the agent). Every target URL was fetched (HTTP 200) and matched its rule.
- Browser at **1280×720**, the training screenshot size; normal Chrome
  user-agent (headless Chromium's default is refused by several sites).
- Excluded after engineering checks, before any comparison: DuckDuckGo (serves a
  bot-block page after one automated search), Opera download (main button starts
  a file download). `debian-distrib` goal reworded for the same reason.

## Components

| Piece | File |
|---|---|
| Live environment (reset, observe, score) | `src/web_agent/eval/task2/web_worker.py` |
| Agent loop (unchanged; per-task allowed domains, real site domain to P1) | `src/web_agent/eval/task2/live.py` |
| Runner | `scripts/run_agent.py --suite web --task all|<ids> --mode baseline|ours` |

## First live check (3 tasks, ours)

| Task | P1 verdicts | Result |
|---|---|---|
| Wikipedia "Alan Turing" | TYPE → SUCCESS (P fail 0.32), CLICK → SUCCESS (0.36) | completed |
| Debian | SCROLL → SUCCESS (0.00, 0.01); clicks failed natively (ISO download button) | not completed |
| DuckDuckGo | FAILURE on the bot-block page — correct | site blocked; task removed |

## Development analysis run: all 20 tasks, ours (2026-09-28)

Run: `.task2-assets/demo/20260928T122723-web-20tasks-ours-s0`. **12/19 completed**,
1 infrastructure error (TED: Playwright screenshot waited for web fonts; now falls
back to Chrome's own `Page.captureScreenshot`, verified on TED).

P1 on real sites versus MiniWoB:

| Steps | Real sites: labelled SUCCESS | median P(failure) | MiniWoB |
|---|---:|---:|---:|
| Interaction, completed episodes (n=28) | 57% | 0.34 | 0% (0.97) |
| Interaction, failed episodes (n=57) | 18% | 1.00 | 0% (0.97) |
| Recovery, completed episodes (n=5) | 80% | – | 0% |
| Recovery, failed episodes (n=24) | 8% | – | 0% |

P1 separates good from bad steps on real pages; it still flags some correct steps
(43% in completed episodes), consistent with its "visible page change" cue.

Failures: 4 genuine (agent wandered until the step budget or gave up: Debian,
EPA, IETF, Stanford) and 3 actor format errors (base Qwen produced invalid native
JSON; affects baseline and ours alike). The base agent therefore fails a real
share of tasks — there is headroom for the with/without comparison.

## Limits

Live sites change over time; a later with/without comparison must run both
systems on the same day, interleaved per task.
