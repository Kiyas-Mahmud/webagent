# ESWA related work and evaluation review

Prepared 2026-09-13. Literature analysis and proposed next research, not a new
execution protocol or authorization to change the frozen experiment.

We already have an executable BrowserGym + MiniWoB hybrid agent. The research
gap is demonstrating useful planning, recovery and memory interventions, and
comparing them fairly. Building another wrapper alone would not address it.

Expert Systems with Applications covers original research on the design,
development, testing and application of expert and intelligent systems. Its
scope does not impose a universal browser-agent requirement. Our own claims
about browser completion require browser execution evidence.
[Elsevier journal scope, ESWA entry](https://shop.elsevier.com/subjects/journals/physical-sciences-and-engineering/computer-science/artificial-intelligence/artificial-intelligence-expert-systems-and-knowledge-based-systems).

## 1. Relevant ESWA papers and what their evaluation establishes

This is a focused review of three methodologically related papers, not a
systematic review or an exhaustive browser-agent bibliography. The accessible
publisher abstracts, introductions and section previews were inspected; full
experimental tables were not accessible. Exact repetition counts, confidence
intervals and statistical procedures remain unverified where not stated below.
The papers cover different domains; their success percentages cannot be ranked
against our MiniWoB percentages. The publisher assigns SDPA to the December
2026 issue; its article preview is already available at this review date.

| Paper | Agent design | Verified evaluation | Lesson for our study |
|---|---|---|---|
| **PenExpert: A multi-agent hybrid LLM–expert system framework for autonomous penetration testing** (2026), DOI 10.1016/j.eswa.2026.133284 | Separates state parsing, planning and execution; combines expert rules, procedural retrieval and failure history. | AutoPenBench, single-host Vulnhub and a multi-host setting; compares PentestGPT, VulnBot, RedTeamLLM and STT-reasoning. Reports completion, penetration depth, token/time cost and capability-matrix/RAG ablations. | Compare complete execution at several task complexities; isolate the knowledge component and account for cost. Its retrieved procedures contain more actionable material than our labels. [Publisher](https://www.sciencedirect.com/science/article/pii/S0957417426021937). |
| **Seeing the details to plan ahead: Grounding LLM tree planning in fine-grained perception for embodied agents** (SDPA, 2026), DOI 10.1016/j.eswa.2026.133107 | Updates a semantic map from perception; grounds a hierarchical task tree and backtracking in that state. | ALFRED Seen/Unseen evaluation, success rates, path-length-weighted metrics, comparisons including Prompter and CAPEAM, and component ablations. The abstract reports 55.97% Seen and 50.75% Unseen success. | Test perception-to-action consistency and recovery from failed steps; separate completion from efficiency and state exactly what generalization was tested. [Publisher](https://www.sciencedirect.com/science/article/abs/pii/S095741742602018X). |
| **CDAFlow: Enhancing LLM clinical decision-making through agentic workflow** (2026), DOI 10.1016/j.eswa.2026.131806 | Uses state transitions, progress evaluation, filtered knowledge rules, and short-/long-term memory. | ClinicalBench: 1,500 cases across 24 departments; comparisons with collaborative-expert and single-agent approaches; evaluates workflow subprocesses and reports an Ace metric. Its exact definition was not verified, so it is not treated as browser task success. | Evaluate intermediate decisions and final outcomes separately; knowledge must be applicable to the current state. Clinical benchmark results are not evidence of browser performance or deployed clinical benefit. [Publisher](https://www.sciencedirect.com/science/article/abs/pii/S0957417426007190). |

These examples motivate the proposal below; they do not establish an ESWA
minimum sample size, mandatory architecture, or guarantee of publication.
Before using detailed claims from these papers in the manuscript, obtain their
full texts through university access and verify the cited experimental tables.

## 2. Where our current evidence stands

The [completed H-study](TABLE2_HYBRID_FINAL_RESULTS_V1.md) already contains 120
episodes, 30 matched task/reset blocks, a frozen protocol and an independent
PASS evidence audit. H0 completed 8/30; H1, H2 and H3 completed 9/30 each.
Neither recovery nor memory added a completion. These are finished results,
not a missing evaluation that needs an unchanged rerun.

The logs identify the immediate barriers:

- All 112 executed actions were CLICK. Text-entry and login preparation were
  not demonstrated.
- H2 and H3 each generated two recovery proposals. All four were malformed and
  rejected before execution; no final-run post-recovery assessment occurred.
- H3 made two retrieval queries in one episode. Both abstained after
  applicability filtering. No example reached generation. The other 29 H3
  episodes did not query memory.
- The existing 1,974-item store lacks corrective values and reflections.
  Storage-decision accuracy does not establish useful retrieved experience.
- H0–H3 are internal ablations. They do not support superiority over external
  competing agents. The six evaluation families were previously observed.

The infrastructure and evidence preservation are strengths. The unproven part
is the proposed mechanism's contribution to completing tasks. The logs locate
failure stages; they do not by themselves establish why the frozen generator
prefers CLICK or produces invalid recovery JSON.

## 3. Preserve all four pillars, with measurable claims

| Pillar | Role in the hybrid agent | Evidence to report |
|---|---|---|
| P1: failure-aware resilience | Learned diagnosis and strategy inform a concrete recovery proposal; execute it and retain learned assessment. | Failure detection, valid recovery proposals, executed recoveries, post-recovery assessment, and H2−H1 completion difference. |
| P2: multimodal decision | Current screenshot and page information, with trained advice and causal history, inform the next decision. | Correct input timing, observed-state consistency and action decisions; separate modality ablations would be needed to isolate P2's own gain. |
| P3: action and grounding | Ground the model-selected action in a current control, preserve its value and execute through Playwright. | Action-type coverage, parsing/target/parameter validity, execution and grounding errors; separate ablations are needed to isolate P3. |
| P4: corrective memory | Frozen retrieval supplies applicable advisory experience to H3 after preserving H2's decision. | Query → candidates → admission → generation exposure → action change → completion, including abstention and H3−H2. |

H0 is the base actor; H1 adds trained P2/P3 advice; H2 adds P1; H3 adds P4.
The generator still selects actions. These identities must stay distinct from
historical E0–E3. Keeping four modules connected does not establish four gains.

## 4. Proposed evaluation package for a stronger manuscript

This is our proposed design, not a protocol copied from one paper. Implement
only a bounded development revision before considering another final study.

1. **Establish a usable action loop on development tasks.** Inspect the actual
   context and generated output for text preparation and recovery-format
   failures. Test a declared interface revision that lets the frozen generator
   choose one valid action and its arguments. Log TYPE → observed state → next
   model-selected action if it occurs. Do not insert a task solution in the
   executor. Schema validity and correct task decisions are separate outcomes.
2. **Exercise recovery in a separate mechanism check.** Include predefined,
   recoverable disturbances on development tasks, with matched conditions and
   budgets across systems. Retain every outcome. Report this separately from
   natural task completion; injected failures cannot stand in for the primary
   unmodified benchmark. Stop action after environment termination.
3. **Decide what memory claim is actually testable.** Under the current frozen
   store, test selective retrieval honestly and accept absent exposure or no
   benefit. A richer experience store would require genuine recorded
   trajectories from a separate permitted preparation split, a new manifest
   and approval of that scope change. It requires no weight training in
   principle, but it is a different memory experiment. Never invent missing
   corrections or populate it from final evaluation outcomes.
4. **Add fair comparison before claiming superiority.** Retain H0–H3 and add
   a reproducible external/reference agent implementation suited to the same
   browser tasks. Match model access, observation information, tools, resets
   and budgets where possible; disclose differences and implementation
   fidelity. A custom baseline must not be presented as the published system.
5. **Freeze fresh evaluation once development is finished.** Choose task
   coverage by capabilities and application relevance before outcomes; include
   both simple and multi-step tasks, preserving all failures. New resets within
   familiar families support only that scope of generalization. Prespecify
   sample size for the desired sensitivity and resources; do not repeatedly
   add episodes until a p-value passes. Preserve the existing final study as
   historical evidence.
6. **Report effectiveness and cost together.** Keep raw full-completion
   scoring, paired counts/differences, improved/worsened pairs and the two
   primary contrasts with multiplicity correction. Report uncertainty,
   denominators, exclusions, parsing/grounding failures, recovery and memory
   exposure, calls and time. Mechanism-check results and end-to-end results
   belong in separately labelled tables.

For an applied browser-agent claim, a later representative workflow case study
could complement MiniWoB. It should use the same declared agent and report
successes and failures; a hand-scripted demonstration is not autonomous-agent
evaluation. This would be an additional scope decision, not a prerequisite
invented to prevent completion of the current study.

## 5. Bounded next decision

The immediate useful work is a development investigation of **valid action
generation and multi-step preparation** with the existing model. A second full
benchmark now would not resolve that cause. Memory cannot help the measured
pipeline when no applicable advice reaches the planner, and recovery cannot
help when its proposals never reach execution.

If a bounded revision still fails, close it with the observed limitations and
reconsider the manuscript claim with the supervisor. A rigorous transfer/failure
analysis is a possible research direction; acceptance still depends on novelty,
evidence and journal review. The current evidence does not support a claim that
all four pillars improve browser completion.

This review changes no checkpoint, prompts, runtime, embeddings, thresholds,
packages, datasets, completed results or completed TODO statuses. No training
or evaluation was launched.
