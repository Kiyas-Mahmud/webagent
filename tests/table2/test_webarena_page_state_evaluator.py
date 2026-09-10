from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from web_agent.benchmarks.base import AdapterExecution, EnvironmentAdapter
from web_agent.benchmarks.webarena import FrozenWebArenaObservationSnapshot
from web_agent.eval.table2.live_page_broker_assembly import (
    assert_process_wide_broker_assembly,
    process_runtime_page_publisher,
    process_sealed_page_evaluator_capability,
)
from web_agent.eval.table2 import webarena_page_state_evaluator as evaluator_module
from web_agent.eval.table2.production_runner import SealedEvaluatorBinding
from web_agent.eval.table2.common import read_jsonl
from web_agent.eval.table2.package_validator import _validate_final_evidence
from web_agent.eval.table2.sealed_page_broker import (
    BrowserGymStartStateApplicationReceipt,
    BrowserGymValidationDisabledBoundary,
)
from web_agent.eval.table2.sealed_verifier import (
    SealedVerifierSink,
    SealedVerifierWriter,
    verify_sealed_stream,
)
from web_agent.eval.table2.webarena_page_state_evaluator import (
    COMPILE_REPORT_STATUS,
    EVALUATOR_ID,
    EVALUATOR_VERSION,
    WebArenaPageStateEvaluatorError,
    build_page_state_evaluator_compile_report,
    compile_page_state_evaluator_config,
    create_sealed_webarena_page_state_evaluator,
    evaluate_compiled_page_state,
    live_page_session_id,
    require_page_state_evaluator_campaign_ready,
    validate_page_state_evaluator_compile_report,
)
from web_agent.runtime.contracts import (
    EpisodeSummary,
    ConcreteAction,
    Observation,
    ObservationStage,
    OpaqueTerminalSignal,
    RuntimeStartState,
    SystemID,
    TaskSpecification,
    TerminalReason,
    VerifierReceiptBinding,
    canonical_sha256,
)
from web_agent.runtime.event_log import EpisodeEventLogs


REFERENCE_FIXTURE = (
    Path(__file__).parent / "fixtures" / "webarena_page_state_reference.json"
)
REFERENCE_GENERATOR = (
    Path(__file__).parent
    / "fixtures"
    / "generate_webarena_page_state_reference.py"
)
SHA_A = "a" * 64


class _FixturePage:
    def __init__(
        self,
        *,
        url: str,
        content: str = "<html></html>",
        dom: dict[str, Any] | None = None,
        mutate_on_evaluate: bool = False,
    ) -> None:
        self.url = url
        self._content = content
        self.dom = dict(dom or {})
        self.mutate_on_evaluate = mutate_on_evaluate

    def content(self) -> str:
        return self._content

    def title(self) -> str:
        return "fixture"

    def evaluate(self, script: str, selector: str) -> Any:
        if self.mutate_on_evaluate:
            self.url = self.url.rstrip("/") + "/mutated"
        property_name = next(
            (name for name in ("outerText", "value", "selectedIndex") if name in script),
            None,
        )
        if property_name is None:
            raise AssertionError("evaluator supplied unknown JavaScript")
        key = f"{selector}|{property_name}"
        if key not in self.dom:
            raise RuntimeError("querySelector target is absent")
        return self.dom[key]

    def digest(self) -> str:
        return canonical_sha256(
            {"url": self.url, "content": self._content, "dom": self.dom}
        )


def _start_state(index: int) -> RuntimeStartState:
    return RuntimeStartState(
        sites=("shopping",),
        start_url=f"https://shop.test/start/{index}",
        require_login=True,
        storage_state=f".auth/shop-{index}.json",
        geolocation=None,
        require_reset=False,
    )


def _url_config(reference_url: str) -> dict[str, Any]:
    return {
        "eval_types": ["url_match"],
        "reference_answers": None,
        "reference_url": reference_url,
        "program_html": [],
        "url_note": "GOLD in PRED",
    }


def _task_row(index: int, config: dict[str, Any]) -> dict[str, Any]:
    start = _start_state(index).to_webarena_mapping()
    return {
        "task_id": f"webarena.{index}",
        "upstream_index": index,
        "benchmark_task_id": str(index),
        "benchmark_task_version": "libwebarena-0.0.4:test.raw.json",
        "instruction": "fixture task",
        "start_state": start,
        "task_config": {
            **start,
            "task_id": index,
            "intent": "fixture task",
            "eval": deepcopy(config),
        },
        "evaluator": {
            "evaluator_id": EVALUATOR_ID,
            "evaluator_version": EVALUATOR_VERSION,
            "config": deepcopy(config),
        },
    }


def _task(index: int) -> TaskSpecification:
    start = _start_state(index)
    return TaskSpecification(
        task_id=f"webarena.{index}",
        goal="fixture task",
        benchmark_id="webarena",
        benchmark_version="0.14.3",
        start_state_id=start.start_state_sha256,
        site=start.sites[0],
        start_url=start.start_url,
        runtime_start_state=start,
    )


def _sink(tmp_path: Path, *, episode_id: str, task_id: str, system_id: str):
    return SealedVerifierSink(
        tmp_path,
        block_id=f"task_{task_id.replace('.', '-')}__seed_42__repeat_0",
        attempt_id=0,
        system_id=system_id,
        episode_id=episode_id,
        matched_seed=42,
        task_id=task_id,
        repeat_id=0,
    )


def _publish(
    *,
    page: _FixturePage,
    episode_id: str,
    task: TaskSpecification,
    cleanup_calls: list[str],
) -> str:
    assert task.runtime_start_state is not None
    session_id = live_page_session_id(
        episode_id=episode_id,
        task_id=task.task_id,
        start_state_sha256=task.runtime_start_state.start_state_sha256,
    )
    process_runtime_page_publisher().publish(
        session_id=session_id,
        episode_id=episode_id,
        task_id=task.task_id,
        page=page,
        page_state_digest=lambda current: current.digest(),
        close_live_page=lambda: cleanup_calls.append("closed"),
        validation_boundary=BrowserGymValidationDisabledBoundary(
            boundary_id="fixture-validation-disabled",
            boundary_version="v1",
            source_sha256=SHA_A,
        ),
        start_state_application=BrowserGymStartStateApplicationReceipt(
            episode_id=episode_id,
            task_id=task.task_id,
            runtime_start_state=task.runtime_start_state,
            applied_start_state_sha256=(
                task.runtime_start_state.start_state_sha256
            ),
            applier_id="fixture-six-field-applier",
            applier_version="v1",
            applier_source_sha256=SHA_A,
        ),
    )
    return session_id


def _bound(
    binding,
    *,
    task: TaskSpecification,
    episode_id: str,
    receipt_kind: str,
    sequence: int,
    action_id: str | None = None,
):
    stage = (
        ObservationStage.RESET
        if receipt_kind == "after_reset"
        else (
            ObservationStage.POST_RECOVERY
            if receipt_kind == "after_recovery_action"
            else ObservationStage.POST_ACTION
        )
    )
    observation = Observation(
        observation_id=f"{episode_id}:observation:{sequence}",
        episode_id=episode_id,
        stage=stage,
        screenshot_sha256=canonical_sha256([episode_id, sequence]),
        url="https://runtime-observation.test/redacted",
        title=f"causal-{sequence}",
        page_state={},
        prior_action_id=action_id,
    )
    receipt = VerifierReceiptBinding(
        receipt_kind=receipt_kind,
        observation_id=observation.observation_id,
        observation_sha256=observation.record_sha256,
        action_id=action_id,
        action_sha256=(canonical_sha256(action_id) if action_id is not None else None),
    )
    snapshot = FrozenWebArenaObservationSnapshot(
        observation=observation,
        environment_state_digester_id="fixture-digester",
        environment_state_digester_version="v1",
        environment_state_sha256=SHA_A,
    )
    return binding.terminal_signal_mapper(snapshot, task, receipt)


def _action_payload(
    action_id: str,
    *,
    status: str,
    executor_step: int,
    attempt_id: str | None = None,
) -> dict[str, Any]:
    return {
        "action": {
            "action_id": action_id,
            "source_decision_id": "decision",
            "action_type": "CLICK",
            "parameters": {},
            "bbox": None,
            "recovery_attempt_id": attempt_id,
        },
        "action_sha256": canonical_sha256(action_id),
        "execution": {
            "action_id": action_id,
            "status": status,
            "executor_step": executor_step,
            "state_changed": status == "executed",
            "environment_error": False,
            "error_kind": None if status == "executed" else "element_not_found",
            "message": "",
            "internal_retry_count": 0,
        },
        **({"attempt_id": attempt_id} if attempt_id is not None else {}),
    }


def test_process_uses_one_disjoint_broker_pair() -> None:
    assert_process_wide_broker_assembly()
    assert process_runtime_page_publisher() is process_runtime_page_publisher()
    assert (
        process_sealed_page_evaluator_capability()
        is process_sealed_page_evaluator_capability()
    )
    assert not hasattr(process_runtime_page_publisher(), "evaluate_bound")
    assert not hasattr(process_sealed_page_evaluator_capability(), "publish")


class _ProcessEntrypointAdapter(EnvironmentAdapter):
    benchmark_id = "webarena"
    benchmark_version = "0.14.3"

    def __init__(self) -> None:
        self.binding = None
        self.writer = None

    def reset(self, task, *, episode_id, seed):
        raise NotImplementedError

    def observe(self, *, stage, prior_action_id=None):
        raise NotImplementedError

    def execute(self, action: ConcreteAction) -> AdapterExecution:
        raise NotImplementedError

    def terminal_signal(self, task, binding=None):
        assert self.binding is not None
        return self.binding.terminal_signal_mapper(None, task, binding)

    def close(self) -> None:
        return None

    def process_broker_bind_sealed_evaluator(self, *, evaluator, evidence_writer):
        assert self.binding is None
        self.binding = evaluator
        self.writer = evidence_writer

    def process_broker_sealed_evaluator(self, *, evidence_writer):
        if self.binding is not None:
            assert evidence_writer is self.writer
        return self.binding

    @staticmethod
    def process_broker_manual_rescue_sidecar_identity():
        return {
            "schema_version": (
                "table2-process-broker-manual-rescue-sidecar-identity-v1"
            ),
            "record_type": "ProcessBrokerManualRescueSidecarIdentity",
            "relative_path": "manual_rescue_guard.child.jsonl",
            "record_count": 0,
            "tail_sha256": None,
            "content_sha256": hashlib.sha256(b"").hexdigest(),
        }


def test_exported_child_callbacks_have_exact_shapes_and_fail_closed_without_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_parameters = {
        evaluator_module.evaluate_environment_transition: (
            "task",
            "adapter",
            "receipt_binding",
            "evidence_writer",
        ),
        evaluator_module.finalize_environment_episode: (
            "task",
            "adapter",
            "episode_summary",
            "episode_runtime_dir",
            "evidence_writer",
        ),
    }
    for callback, names in expected_parameters.items():
        parameters = tuple(inspect.signature(callback).parameters.values())
        assert tuple(item.name for item in parameters) == names
        assert all(
            item.kind is inspect.Parameter.KEYWORD_ONLY
            and item.default is inspect.Parameter.empty
            for item in parameters
        )

    index = 777
    task = _task(index)
    adapter = _ProcessEntrypointAdapter()
    sink = _sink(
        tmp_path,
        episode_id="entrypoint:E2:webarena.777:repeat-0:seed-42",
        task_id=task.task_id,
        system_id="E2",
    )
    writer = SealedVerifierWriter(sink)
    observation = Observation(
        observation_id="entrypoint-reset",
        episode_id="entrypoint:E2:webarena.777:repeat-0:seed-42",
        stage=ObservationStage.RESET,
        screenshot_sha256=SHA_A,
        url=task.start_url,
        title="fixture",
        page_state={},
    )
    receipt_binding = VerifierReceiptBinding(
        receipt_kind="after_reset",
        observation_id=observation.observation_id,
        observation_sha256=observation.record_sha256,
    )
    with pytest.raises(
        WebArenaPageStateEvaluatorError,
        match="bootstrap is not registered",
    ):
        evaluator_module.evaluate_environment_transition(
            task=task,
            adapter=adapter,
            receipt_binding=receipt_binding,
            evidence_writer=writer,
        )

    row = _task_row(index, _url_config("https://shop.test/done"))
    content = dict(row)
    row["source_content_sha256"] = canonical_sha256(content)
    task = replace(
        task,
        metadata={
            "task_partition": "normal",
            "upstream_index": index,
            "benchmark_task_id": str(index),
            "source_content_sha256": row["source_content_sha256"],
        },
    )

    def fake_factory(task_row, supplied_writer, *, manual_rescue_identity_provider):
        assert task_row == row
        assert supplied_writer is writer
        assert manual_rescue_identity_provider()["record_count"] == 0
        return SealedEvaluatorBinding(
            benchmark_version="0.14.3",
            evaluator_id=EVALUATOR_ID,
            evaluator_version=EVALUATOR_VERSION,
            terminal_signal_mapper=lambda _snapshot, _task, _receipt: (
                OpaqueTerminalSignal(
                    event_id="entrypoint-bound",
                    token_sha256="b" * 64,
                    terminate=False,
                )
            ),
            finalize_episode_evidence=lambda _summary, _writer, _runtime: (
                OpaqueTerminalSignal(
                    event_id="entrypoint-final",
                    token_sha256="c" * 64,
                    terminate=True,
                )
            ),
        )

    monkeypatch.setattr(
        evaluator_module,
        "_load_process_broker_evaluator_task_row",
        lambda *, task: row,
    )
    monkeypatch.setattr(
        evaluator_module,
        "create_sealed_webarena_page_state_evaluator",
        fake_factory,
    )
    transition = evaluator_module.evaluate_environment_transition(
        task=task,
        adapter=adapter,
        receipt_binding=receipt_binding,
        evidence_writer=writer,
    )
    assert transition.event_id == "entrypoint-bound"
    summary = EpisodeSummary(
        episode_id=observation.episode_id,
        protocol_id="table2-pc01-pilot-v1",
        system_id=SystemID.E2,
        task_id=task.task_id,
        repeat_id=0,
        model_seed=42,
        valid_for_primary=True,
        terminal_reason=TerminalReason.TIMEOUT,
        executor_steps=0,
        normal_actions=0,
        recovery_actions=0,
        recovery_attempts=0,
        failure_incidents=0,
        memory_queries=0,
        memory_interventions=0,
        elapsed_seconds=1.0,
    )
    final = evaluator_module.finalize_environment_episode(
        task=task,
        adapter=adapter,
        episode_summary=summary,
        episode_runtime_dir=tmp_path,
        evidence_writer=writer,
    )
    assert final.event_id == "entrypoint-final"


def test_reference_fixture_is_replayable_and_source_attested() -> None:
    fixture = json.loads(REFERENCE_FIXTURE.read_text(encoding="utf-8"))
    assert fixture["schema_version"] == "table2-webarena-page-state-reference-v1"
    assert (
        fixture["classification"]
        == "generated_pinned_upstream_executable_reference"
    )
    assert (
        fixture["reference_role"]
        == "local_compatibility_parity_only_not_external_review"
    )
    payload_sha256 = fixture.pop("payload_sha256")
    assert canonical_sha256(fixture) == payload_sha256
    provenance = fixture["provenance"]
    assert provenance["generator_source_sha256"] == hashlib.sha256(
        REFERENCE_GENERATOR.read_bytes()
    ).hexdigest()
    assert (
        provenance["wheel_sha256"]
        == "9ebee3b4371502c4f0f7e727a72e5846235d6750d420db9a3b8a168107654feb"
    )
    assert (
        provenance["wheel_member_sha256"]["webarena/test.raw.json"]
        == "7b50386fd69163dbc05d615d834df4c6ed2c35596e97a1b10d17451c02537652"
    )
    assert (
        provenance["imported_evaluator_source_sha256"]
        == provenance["wheel_member_sha256"][
            "webarena/evaluation_harness/evaluators.py"
        ]
    )


def test_supported_semantics_match_generated_upstream_reference_fixture() -> None:
    fixture = json.loads(REFERENCE_FIXTURE.read_text(encoding="utf-8"))
    for case in fixture["cases"]:
        page = _FixturePage(**case["page"])
        compiled = compile_page_state_evaluator_config(case["config"])
        result = evaluate_compiled_page_state(page, compiled)
        assert list(result.vector) == case["expected_vector"], case["case_id"]

    pinned = next(
        case
        for case in fixture["cases"]
        if case["case_id"] == "exact_pinned_task_758_program_html"
    )
    assert pinned["source"]["kind"] == "exact_pinned_task_eval"
    assert pinned["source"]["task_index"] == 758
    # These are the exact unusual upstream bytes.  The compatibility port must
    # pass the parsed selector through unchanged; it must never add the `]`.
    assert pinned["config"]["program_html"][2]["locator"] == (
        "document.querySelector('[name=\"route_to\"').value"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda cfg: cfg.update(
            {
                "eval_types": ["string_match"],
                "reference_answers": {"exact_match": "secret"},
                "reference_url": None,
            }
        ),
        lambda cfg: cfg.update({"unknown": True}),
        lambda cfg: cfg.update({"url_note": "PRED in GOLD"}),
        lambda cfg: cfg.update(
            {"reference_url": "https://user:password@shop.test/done"}
        ),
        lambda cfg: cfg.update(
            {"reference_url": "https://shop.test/done?session_token=secret"}
        ),
        lambda cfg: cfg.update(
            {
                "eval_types": ["program_html", "url_match"],
                "program_html": [
                    {
                        "url": "last",
                        "locator": "document.querySelector(\"main\").outerText",
                        "required_contents": {"must_include": ["done"]},
                    }
                ],
            }
        ),
    ],
)
def test_url_config_rejects_string_unknown_extra_and_credentials(mutation) -> None:
    config = _url_config("https://shop.test/done")
    mutation(config)
    with pytest.raises(WebArenaPageStateEvaluatorError):
        compile_page_state_evaluator_config(config)


@pytest.mark.parametrize(
    "target_change",
    [
        {"url": "func:shopping_get_latest_order_url()"},
        {"url": "https://shop.test/other"},
        {"locator": "func:helper(__page__)"},
        {"locator": "eval:document.body.innerText"},
        {"locator": "document.body.outerText"},
        {"locator": " document.querySelector(\"main\").outerText"},
        {"locator": "document.querySelector(r'main').outerText"},
        {"locator": "document.querySelector(u'main').outerText"},
        {"locator": "document.querySelector('ma' 'in').outerText"},
        {"locator": "document.querySelector('''main''').outerText"},
        {"locator": r'document.querySelector("#a\/b").outerText'},
        {"prep_actions": ["document.body.click()"]},
        {"required_contents": {"fuzzy_match": ["value"]}},
    ],
)
def test_program_html_rejects_navigation_code_mutation_and_unknown_rules(
    target_change,
) -> None:
    target = {
        "url": "last",
        "locator": "document.querySelector(\"main\").outerText",
        "required_contents": {"must_include": ["value"]},
    }
    target.update(target_change)
    config = {
        "eval_types": ["program_html"],
        "reference_answers": None,
        "reference_url": None,
        "program_html": [target],
    }
    with pytest.raises(WebArenaPageStateEvaluatorError):
        compile_page_state_evaluator_config(config)


def test_factory_rejects_official_claim_and_task_config_disagreement(tmp_path) -> None:
    row = _task_row(901, _url_config("https://shop.test/done"))
    sink = _sink(
        tmp_path,
        episode_id="factory:E2:webarena.901:repeat-0:seed-42",
        task_id="webarena.901",
        system_id="E2",
    )
    row["evaluator"]["evaluator_version"] = "official-v1"
    with pytest.raises(WebArenaPageStateEvaluatorError, match="pending external review"):
        create_sealed_webarena_page_state_evaluator(row, SealedVerifierWriter(sink))

    row = _task_row(902, _url_config("https://shop.test/done"))
    row["task_config"]["eval"]["reference_url"] = "https://shop.test/other"
    with pytest.raises(WebArenaPageStateEvaluatorError, match="differs"):
        create_sealed_webarena_page_state_evaluator(row, SealedVerifierWriter(sink))


def test_compile_report_binds_all_50_rows_but_never_claims_readiness() -> None:
    rows = [
        _task_row(index, _url_config(f"https://shop.test/done/{index}"))
        for index in range(1000, 1050)
    ]
    report = build_page_state_evaluator_compile_report(rows)

    assert report["status"] == COMPILE_REPORT_STATUS
    assert report["implementation_classification"] == EVALUATOR_VERSION
    assert report["task_count"] == 50
    assert report["criterion_count"] == 50
    assert report["all_task_configs_compiled"] is True
    assert report["external_review_status"] == "PENDING"
    assert report["campaign_ready"] is False
    assert report["handoff_eligible"] is False
    assert report["freeze_eligible"] is False
    assert report["production_eligible"] is False
    unsigned = dict(report)
    report_sha256 = unsigned.pop("report_sha256")
    assert report_sha256 == canonical_sha256(unsigned)
    assert validate_page_state_evaluator_compile_report(
        report, task_rows=rows
    ) == report
    with pytest.raises(WebArenaPageStateEvaluatorError, match="not campaign-ready"):
        require_page_state_evaluator_campaign_ready(report, task_rows=rows)


def test_compile_report_fails_closed_on_incomplete_duplicate_or_uncompiled_row() -> None:
    rows = [
        _task_row(index, _url_config(f"https://shop.test/done/{index}"))
        for index in range(1100, 1150)
    ]
    with pytest.raises(WebArenaPageStateEvaluatorError, match="exactly 50"):
        build_page_state_evaluator_compile_report(rows[:-1])

    duplicate = deepcopy(rows)
    duplicate[-1]["benchmark_task_id"] = duplicate[0]["benchmark_task_id"]
    with pytest.raises(WebArenaPageStateEvaluatorError, match="duplicate"):
        build_page_state_evaluator_compile_report(duplicate)

    unsupported = deepcopy(rows)
    unsupported[-1]["evaluator"]["config"]["eval_types"] = ["string_match"]
    unsupported[-1]["task_config"]["eval"] = deepcopy(
        unsupported[-1]["evaluator"]["config"]
    )
    with pytest.raises(WebArenaPageStateEvaluatorError, match="forbidden"):
        build_page_state_evaluator_compile_report(unsupported)


def test_compile_report_recomputation_rejects_report_or_task_row_tampering() -> None:
    rows = [
        _task_row(index, _url_config(f"https://shop.test/done/{index}"))
        for index in range(1200, 1250)
    ]
    report = build_page_state_evaluator_compile_report(rows)
    tampered_report = deepcopy(report)
    tampered_report["campaign_ready"] = True
    with pytest.raises(WebArenaPageStateEvaluatorError, match="recomputation"):
        validate_page_state_evaluator_compile_report(
            tampered_report, task_rows=rows
        )

    tampered_rows = deepcopy(rows)
    tampered_rows[-1]["evaluator"]["config"]["reference_url"] = (
        "https://shop.test/different"
    )
    tampered_rows[-1]["task_config"]["eval"] = deepcopy(
        tampered_rows[-1]["evaluator"]["config"]
    )
    with pytest.raises(WebArenaPageStateEvaluatorError, match="recomputation"):
        validate_page_state_evaluator_compile_report(
            report, task_rows=tampered_rows
        )


def test_bound_evaluation_is_oracle_blind_and_seals_only_hashed_criteria(
    tmp_path,
) -> None:
    index = 903
    episode = f"causal:E2:webarena.{index}:repeat-0:seed-42"
    task = _task(index)
    page = _FixturePage(url="https://shop.test/start")
    cleanup: list[str] = []
    session = _publish(
        page=page, episode_id=episode, task=task, cleanup_calls=cleanup
    )
    sink = _sink(
        tmp_path,
        episode_id=episode,
        task_id=task.task_id,
        system_id="E2",
    )
    evaluator = create_sealed_webarena_page_state_evaluator(
        _task_row(index, _url_config("https://shop.test/done?view=all")),
        SealedVerifierWriter(sink),
    )
    first = _bound(
        evaluator,
        task=task,
        episode_id=episode,
        receipt_kind="after_reset",
        sequence=1,
    )
    second = _bound(
        evaluator,
        task=task,
        episode_id=episode,
        receipt_kind="after_normal_action",
        sequence=2,
        action_id="causal-action",
    )
    assert first.terminate is second.terminate is False
    records = verify_sealed_stream(sink.path)
    assert (
        records[0]["evidence"]["criterion_vector_sha256"]
        == records[1]["evidence"]["criterion_vector_sha256"]
    )
    serialized = json.dumps(records)
    assert "view=all" not in serialized
    assert "reference_url" not in serialized
    process_runtime_page_publisher().abort(
        session_id=session, episode_id=episode, task_id=task.task_id
    )
    assert cleanup == ["closed"]


def test_broker_rejects_page_mutation_during_read_only_html_evaluation(
    tmp_path,
) -> None:
    index = 904
    episode = f"mutation:E2:webarena.{index}:repeat-0:seed-42"
    task = _task(index)
    page = _FixturePage(
        url="https://shop.test/start",
        dom={"main|outerText": "done"},
        mutate_on_evaluate=True,
    )
    cleanup: list[str] = []
    _publish(page=page, episode_id=episode, task=task, cleanup_calls=cleanup)
    config = {
        "eval_types": ["program_html"],
        "reference_answers": None,
        "reference_url": None,
        "program_html": [
            {
                "url": "last",
                "locator": "document.querySelector(\"main\").outerText",
                "required_contents": {"exact_match": "done"},
            }
        ],
    }
    sink = _sink(
        tmp_path,
        episode_id=episode,
        task_id=task.task_id,
        system_id="E2",
    )
    evaluator = create_sealed_webarena_page_state_evaluator(
        _task_row(index, config), SealedVerifierWriter(sink)
    )
    with pytest.raises(RuntimeError, match="mutated"):
        _bound(
            evaluator,
            task=task,
            episode_id=episode,
            receipt_kind="after_reset",
            sequence=1,
        )
    assert cleanup == ["closed"]


@pytest.mark.parametrize("system_id", [SystemID.E2, SystemID.E3])
def test_final_evidence_reconciles_verified_executor_recovery_and_p4_conservatively(
    tmp_path,
    system_id: SystemID,
) -> None:
    index = 905 if system_id is SystemID.E2 else 906
    episode = f"recovery:{system_id.value}:webarena.{index}:repeat-0:seed-42"
    task = _task(index)
    page = _FixturePage(url="https://shop.test/start")
    cleanup: list[str] = []
    session = _publish(
        page=page, episode_id=episode, task=task, cleanup_calls=cleanup
    )
    sink = _sink(
        tmp_path,
        episode_id=episode,
        task_id=task.task_id,
        system_id=system_id.value,
    )
    writer = SealedVerifierWriter(sink)
    evaluator = create_sealed_webarena_page_state_evaluator(
        _task_row(index, _url_config("https://shop.test/done")),
        writer,
    )
    _bound(
        evaluator,
        task=task,
        episode_id=episode,
        receipt_kind="after_reset",
        sequence=1,
    )

    logs = EpisodeEventLogs(
        tmp_path / "runtime",
        episode_id=episode,
        include_memory=system_id is SystemID.E3,
        system_id=system_id.value,
        task_id=task.task_id,
        repeat_id=0,
        matched_seed=42,
        timestamp_factory=lambda: "2000-01-01T00:00:00+00:00",
    )
    failed_action = f"{episode}:normal-action:1"
    incident = f"{episode}:incident:1"
    attempt = f"{incident}:attempt:1"
    recovery_action = f"{attempt}:action:1"
    logs.append(
        "actions",
        "normal_action",
        _action_payload(failed_action, status="error", executor_step=1),
    )
    _bound(
        evaluator,
        task=task,
        episode_id=episode,
        receipt_kind="after_normal_action",
        sequence=2,
        action_id=failed_action,
    )

    shadow = {
        "decision_id": f"{incident}:shadow",
        "incident_id": incident,
        "strategy": "RETRY",
        "trigger_sources": ["executor"],
        "diagnosis": "element_not_found",
        "planned_action": None,
    }
    if system_id is SystemID.E3:
        query_id = f"{incident}:memory-query:1"
        logs.append("memory_queries", "no_memory_shadow_before_retrieval", shadow)
        logs.append(
            "memory_queries",
            "post_failure_query",
            {
                "query": {
                    "query_id": query_id,
                    "incident_id": incident,
                    "failed_action_id": failed_action,
                },
                "query_result": {
                    "query_id": query_id,
                    "candidate_ids": ["memory-train-1"],
                    "admitted": True,
                    "admitted_candidate_id": "memory-train-1",
                    "changed_strategy": True,
                    "changed_target_or_parameters": False,
                },
            },
        )
    logs.append(
        "recoveries",
        "recovery_plan",
        {
            "shadow_decision": shadow,
            "final_decision": shadow,
            "plan": {
                "attempt_id": attempt,
                "incident_id": incident,
                "strategy": "RETRY",
                "incident_attempt_index": 1,
                "episode_attempt_index": 1,
                "actions": [{"action_id": recovery_action}],
                "resolution_status": "READY",
                "rejection_reason": "",
            },
        },
    )
    logs.append(
        "actions",
        "recovery_action",
        _action_payload(
            recovery_action,
            status="executed",
            executor_step=2,
            attempt_id=attempt,
        ),
    )
    page.url = "https://shop.test/done"
    terminal = _bound(
        evaluator,
        task=task,
        episode_id=episode,
        receipt_kind="after_recovery_action",
        sequence=3,
        action_id=recovery_action,
    )
    assert terminal.terminate is True
    logs.append(
        "recoveries",
        "recovery_attempt",
        {
            "attempt": {
                "attempt_id": attempt,
                "incident_id": incident,
                "strategy": "RETRY",
                "incident_attempt_index": 1,
                "episode_attempt_index": 1,
                "action_ids": [recovery_action],
                "completed": True,
                "predicted_assessment_id": "assessment-1",
            }
        },
    )
    summary = EpisodeSummary(
        episode_id=episode,
        protocol_id="table2-pc01-pilot-v1",
        system_id=system_id,
        task_id=task.task_id,
        repeat_id=0,
        model_seed=42,
        valid_for_primary=True,
        terminal_reason=TerminalReason.OPAQUE_VERIFIER_TERMINAL,
        executor_steps=2,
        normal_actions=1,
        recovery_actions=1,
        recovery_attempts=1,
        failure_incidents=1,
        memory_queries=1 if system_id is SystemID.E3 else 0,
        memory_interventions=1 if system_id is SystemID.E3 else 0,
        elapsed_seconds=1.0,
        verifier_event_id=terminal.event_id,
        verifier_token_sha256=terminal.token_sha256,
    )
    process_runtime_page_publisher().close(
        session_id=session, episode_id=episode, task_id=task.task_id
    )
    evaluator.finalize_episode_evidence(summary, writer, tmp_path / "runtime")
    assert cleanup == ["closed"]
    evidence = verify_sealed_stream(sink.path)[-1]["evidence"]
    assert evidence["task_success"] is True
    assert evidence["verified_failure_event_count"] == 1
    assert evidence["failure_incidents"][0]["failure_kind"] == "EXECUTION_ERROR"
    assert evidence["failure_incidents"][0]["resolved"] is True
    assert evidence["recovery_verifications"] == [
        {
            "execution_error_resolved": True,
            "failure_incident_id": incident,
            "reached_task_success": True,
            "recovery_attempt_id": attempt,
            "strict_criterion_progress": True,
            "successful": True,
            "verified_failure_present": True,
        }
    ]
    if system_id is SystemID.E3:
        relevance = evidence["memory_relevance"][f"{incident}:memory-query:1"]
        assert relevance["relevant_ids"] == []
        assert relevance["useful_intervention"] is True
        assert relevance["harmful_intervention"] is False
    else:
        assert evidence["memory_relevance"] == {}
    recovery_events = [
        row
        for row in read_jsonl(tmp_path / "runtime" / "recoveries.jsonl")
        if row["event_type"] == "recovery_attempt"
    ]
    post_queries = [
        row["payload"]["query_result"]
        for row in (
            read_jsonl(tmp_path / "runtime" / "memory_queries.jsonl")
            if system_id is SystemID.E3
            else []
        )
        if row["event_type"] == "post_failure_query"
    ]
    _validate_final_evidence(
        evidence,
        system_id=system_id.value,
        summary={
            "runtime_terminal_reason": summary.terminal_reason.value,
            "infrastructure_invalid": False,
        },
        recovery_attempt_events=recovery_events,
        post_queries=post_queries,
        context="sealed-page-evaluator-unit-fixture",
    )


def test_policy_only_trigger_without_executor_loop_or_regression_is_false_positive(
    tmp_path,
) -> None:
    index = 907
    episode = f"false-positive:E2:webarena.{index}:repeat-0:seed-42"
    task = _task(index)
    page = _FixturePage(url="https://shop.test/start")
    cleanup: list[str] = []
    session = _publish(
        page=page, episode_id=episode, task=task, cleanup_calls=cleanup
    )
    sink = _sink(
        tmp_path,
        episode_id=episode,
        task_id=task.task_id,
        system_id="E2",
    )
    writer = SealedVerifierWriter(sink)
    evaluator = create_sealed_webarena_page_state_evaluator(
        _task_row(index, _url_config("https://shop.test/done")),
        writer,
    )
    _bound(
        evaluator,
        task=task,
        episode_id=episode,
        receipt_kind="after_reset",
        sequence=1,
    )
    logs = EpisodeEventLogs(
        tmp_path / "runtime",
        episode_id=episode,
        include_memory=False,
        timestamp_factory=lambda: "2000-01-01T00:00:00+00:00",
    )
    normal = f"{episode}:normal-action:1"
    incident = f"{episode}:incident:1"
    attempt = f"{incident}:attempt:1"
    recovery = f"{attempt}:action:1"
    logs.append(
        "actions", "normal_action", _action_payload(normal, status="executed", executor_step=1)
    )
    _bound(
        evaluator,
        task=task,
        episode_id=episode,
        receipt_kind="after_normal_action",
        sequence=2,
        action_id=normal,
    )
    shadow = {
        "incident_id": incident,
        "trigger_sources": ["policy"],
    }
    logs.append(
        "recoveries",
        "recovery_plan",
        {
            "shadow_decision": shadow,
            "plan": {
                "attempt_id": attempt,
                "incident_id": incident,
                "strategy": "RETRY",
                "incident_attempt_index": 1,
                "episode_attempt_index": 1,
                "actions": [{"action_id": recovery}],
                "resolution_status": "READY",
            },
        },
    )
    logs.append(
        "actions",
        "recovery_action",
        _action_payload(
            recovery, status="executed", executor_step=2, attempt_id=attempt
        ),
    )
    _bound(
        evaluator,
        task=task,
        episode_id=episode,
        receipt_kind="after_recovery_action",
        sequence=3,
        action_id=recovery,
    )
    logs.append(
        "recoveries",
        "recovery_attempt",
        {
            "attempt_id": attempt,
            "incident_id": incident,
            "strategy": "RETRY",
            "incident_attempt_index": 1,
            "episode_attempt_index": 1,
            "action_ids": [recovery],
            "completed": True,
        },
    )
    summary = EpisodeSummary(
        episode_id=episode,
        protocol_id="table2-pc01-pilot-v1",
        system_id=SystemID.E2,
        task_id=task.task_id,
        repeat_id=0,
        model_seed=42,
        valid_for_primary=True,
        terminal_reason=TerminalReason.ACTION_BUDGET_EXHAUSTED,
        executor_steps=2,
        normal_actions=1,
        recovery_actions=1,
        recovery_attempts=1,
        failure_incidents=1,
        memory_queries=0,
        memory_interventions=0,
        elapsed_seconds=1.0,
    )
    process_runtime_page_publisher().close(
        session_id=session, episode_id=episode, task_id=task.task_id
    )
    evaluator.finalize_episode_evidence(summary, writer, tmp_path / "runtime")
    evidence = verify_sealed_stream(sink.path)[-1]["evidence"]
    assert evidence["verified_failure_event_count"] == 0
    assert evidence["failure_incidents"][0]["verified_agent_failure"] is False
    assert (
        evidence["failure_incidents"][0]["failure_kind"]
        == "POLICY_TRIGGER_FALSE_POSITIVE"
    )
    assert evidence["recovery_verifications"][0]["successful"] is False
    assert cleanup == ["closed"]


def test_each_attempt_keeps_its_sealed_success_when_runtime_continues(
    tmp_path,
) -> None:
    index = 909
    episode = f"two-attempts:E2:webarena.{index}:repeat-0:seed-42"
    task = _task(index)
    page = _FixturePage(
        url="https://shop.test/start",
        dom={"main|outerText": "waiting"},
    )
    cleanup: list[str] = []
    session = _publish(
        page=page, episode_id=episode, task=task, cleanup_calls=cleanup
    )
    sink = _sink(
        tmp_path,
        episode_id=episode,
        task_id=task.task_id,
        system_id="E2",
    )
    writer = SealedVerifierWriter(sink)
    config = {
        "eval_types": ["url_match", "program_html"],
        "reference_answers": None,
        "reference_url": "https://shop.test/done",
        "url_note": "GOLD in PRED",
        "program_html": [
            {
                "url": "last",
                "locator": "document.querySelector(\"main\").outerText",
                "required_contents": {"exact_match": "ready"},
            }
        ],
    }
    evaluator = create_sealed_webarena_page_state_evaluator(
        _task_row(index, config), writer
    )
    _bound(
        evaluator,
        task=task,
        episode_id=episode,
        receipt_kind="after_reset",
        sequence=1,
    )
    logs = EpisodeEventLogs(
        tmp_path / "runtime",
        episode_id=episode,
        include_memory=False,
        timestamp_factory=lambda: "2000-01-01T00:00:00+00:00",
    )
    normal = f"{episode}:normal-action:1"
    incident = f"{episode}:incident:1"
    logs.append(
        "actions",
        "normal_action",
        _action_payload(normal, status="error", executor_step=1),
    )
    _bound(
        evaluator,
        task=task,
        episode_id=episode,
        receipt_kind="after_normal_action",
        sequence=2,
        action_id=normal,
    )

    shadow = {"incident_id": incident, "trigger_sources": ["executor"]}
    attempt_ids: list[str] = []
    recovery_ids: list[str] = []
    for attempt_index in (1, 2):
        attempt_id = f"{incident}:attempt:{attempt_index}"
        recovery_id = f"{attempt_id}:action:1"
        attempt_ids.append(attempt_id)
        recovery_ids.append(recovery_id)
        logs.append(
            "recoveries",
            "recovery_plan",
            {
                "shadow_decision": shadow,
                "plan": {
                    "attempt_id": attempt_id,
                    "incident_id": incident,
                    "strategy": "RETRY",
                    "incident_attempt_index": attempt_index,
                    "episode_attempt_index": attempt_index,
                    "actions": [{"action_id": recovery_id}],
                    "resolution_status": "READY",
                },
            },
        )
        logs.append(
            "actions",
            "recovery_action",
            _action_payload(
                recovery_id,
                status="executed",
                executor_step=attempt_index + 1,
                attempt_id=attempt_id,
            ),
        )
        if attempt_index == 1:
            page.url = "https://shop.test/done"
        else:
            page.dom["main|outerText"] = "ready"
        signal = _bound(
            evaluator,
            task=task,
            episode_id=episode,
            receipt_kind="after_recovery_action",
            sequence=attempt_index + 2,
            action_id=recovery_id,
        )
        assert signal.terminate is (attempt_index == 2)
        logs.append(
            "recoveries",
            "recovery_attempt",
            {
                "attempt_id": attempt_id,
                "incident_id": incident,
                "strategy": "RETRY",
                "incident_attempt_index": attempt_index,
                "episode_attempt_index": attempt_index,
                "action_ids": [recovery_id],
                "completed": True,
            },
        )

    summary = EpisodeSummary(
        episode_id=episode,
        protocol_id="table2-pc01-pilot-v1",
        system_id=SystemID.E2,
        task_id=task.task_id,
        repeat_id=0,
        model_seed=42,
        valid_for_primary=True,
        terminal_reason=TerminalReason.OPAQUE_VERIFIER_TERMINAL,
        executor_steps=3,
        normal_actions=1,
        recovery_actions=2,
        recovery_attempts=2,
        failure_incidents=1,
        memory_queries=0,
        memory_interventions=0,
        elapsed_seconds=1.0,
        verifier_event_id=signal.event_id,
        verifier_token_sha256=signal.token_sha256,
    )
    process_runtime_page_publisher().close(
        session_id=session, episode_id=episode, task_id=task.task_id
    )
    evaluator.finalize_episode_evidence(summary, writer, tmp_path / "runtime")
    evidence = verify_sealed_stream(sink.path)[-1]["evidence"]
    assert [
        item["successful"] for item in evidence["recovery_verifications"]
    ] == [True, True]
    assert evidence["failure_incidents"][0]["resolved_attempt_index"] == 1
    assert evidence["failure_incidents"][0]["resolved"] is True
    assert cleanup == ["closed"]


def test_forced_runtime_failure_cannot_become_success_from_late_page_state(
    tmp_path,
) -> None:
    index = 908
    episode = f"forced:E2:webarena.{index}:repeat-0:seed-42"
    task = _task(index)
    page = _FixturePage(url="https://shop.test/start")
    cleanup: list[str] = []
    session = _publish(
        page=page, episode_id=episode, task=task, cleanup_calls=cleanup
    )
    sink = _sink(
        tmp_path,
        episode_id=episode,
        task_id=task.task_id,
        system_id="E2",
    )
    writer = SealedVerifierWriter(sink)
    evaluator = create_sealed_webarena_page_state_evaluator(
        _task_row(index, _url_config("https://shop.test/done")),
        writer,
    )
    _bound(
        evaluator,
        task=task,
        episode_id=episode,
        receipt_kind="after_reset",
        sequence=1,
    )
    logs = EpisodeEventLogs(
        tmp_path / "runtime",
        episode_id=episode,
        include_memory=False,
    )
    page.url = "https://shop.test/done"
    summary = EpisodeSummary(
        episode_id=episode,
        protocol_id="table2-pc01-pilot-v1",
        system_id=SystemID.E2,
        task_id=task.task_id,
        repeat_id=0,
        model_seed=42,
        valid_for_primary=True,
        terminal_reason=TerminalReason.TIMEOUT,
        executor_steps=0,
        normal_actions=0,
        recovery_actions=0,
        recovery_attempts=0,
        failure_incidents=0,
        memory_queries=0,
        memory_interventions=0,
        elapsed_seconds=600.0,
    )
    process_runtime_page_publisher().close(
        session_id=session, episode_id=episode, task_id=task.task_id
    )
    evaluator.finalize_episode_evidence(summary, writer, tmp_path / "runtime")
    evidence = verify_sealed_stream(sink.path)[-1]["evidence"]
    assert evidence["raw_page_criteria_satisfied"] is True
    assert evidence["task_success"] is False
    assert evidence["forced_failure_runtime_semantics_applied"] is True
    assert evidence["terminal_reason"] == "FORCED_RUNTIME_FAILURE:TIMEOUT"
    assert cleanup == ["closed"]
