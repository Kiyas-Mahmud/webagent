from __future__ import annotations

import hashlib
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest

from web_agent.runtime.checkpoint_inference import (
    LoadedSelectedBackboneBackend,
    LoadedSelectedCheckpointBackend,
    ValidationSelectedBackbone,
    ValidationSelectedCheckpoint,
)
from web_agent.runtime.observation import ProcessorParityContract
from web_agent.runtime.policy import ActionParseError
from web_agent.runtime.action_parameters import DeterministicParameterProvider
from web_agent.runtime.contracts import (
    ActionType,
    PolicyObservation,
    RuntimeTaskView,
)
from web_agent.runtime.qwen2vl_pc01 import (
    E0_ACTION_PROMPT,
    E0_PARSER_ID,
    E0_PARSER_VERSION,
    PARAMETER_PROVIDER_PROMPT,
    REGISTERED_GENERATION_KWARGS,
    LoadedPC01Backends,
    PC01_LIVE_DEPLOYMENT_CAPABILITIES,
    PC01EvaluationRuntimeFactory,
    PC01SeedOperationalServices,
    PC01RuntimeArtifacts,
    PC01RuntimeError,
    make_pc01_backend_factories,
    parse_e0_action_output,
    require_live_factory,
)
from web_agent.runtime import qwen2vl_pc01


def _processor() -> ProcessorParityContract:
    return ProcessorParityContract(
        processor_class="test.FrozenProcessor",
        processor_revision="895c3a49bc3fa70a340399125c650a463535e71c",
        processor_config_sha256="a" * 64,
        pre_action_field_mapping={"image": "images[0]"},
        post_action_field_mapping={
            "before": "images[0]",
            "post": "images[1]",
        },
    )


def _selections(tmp_path: Path):
    selected = ValidationSelectedCheckpoint(
        manifest_id="pc01",
        model_seed=42,
        checkpoint_path=tmp_path / "selected.ckpt",
        selected_checkpoint_sha256="1" * 64,
        resolved_config_sha256="2" * 64,
        processor_contract_sha256=_processor().record_sha256,
        validation_rows_read=7861,
        checkpoint_selection="validation_only",
        selection_scope="validation_only",
        test_rows_read=0,
        locked_test_rows_read=0,
    )
    e0 = ValidationSelectedBackbone(
        manifest_id="pc01-e0",
        backbone_id="Qwen/Qwen2-VL-2B-Instruct",
        backbone_revision="895c3a49bc3fa70a340399125c650a463535e71c",
        backbone_path=tmp_path / "base",
        backbone_sha256="3" * 64,
        resolved_config_sha256="4" * 64,
        processor_contract_sha256=_processor().record_sha256,
        base_prompt_sha256=hashlib.sha256(E0_ACTION_PROMPT.encode()).hexdigest(),
        parser_id=E0_PARSER_ID,
        parser_version=E0_PARSER_VERSION,
        validation_rows_read=7861,
        selection_scope="validation_only",
        test_rows_read=0,
        locked_test_rows_read=0,
    )
    return selected, e0


def _artifacts(tmp_path: Path) -> PC01RuntimeArtifacts:
    return PC01RuntimeArtifacts(
        resolved_config_path=tmp_path / "resolved.json",
        processor_contract_path=tmp_path / "processor.json",
        processor_source=tmp_path / "base",
        e0_resolved_config_path=tmp_path / "e0.json",
        e0_processor_contract_path=tmp_path / "e0-processor.json",
        e0_backbone_path=tmp_path / "base",
    )


def _loaded_bundle(selected, e0) -> LoadedPC01Backends:
    processor = _processor()

    def unavailable(*args, **kwargs):
        raise AssertionError("test callback should not run")

    return LoadedPC01Backends(
        selected=LoadedSelectedCheckpointBackend(
            checkpoint_sha256=selected.selected_checkpoint_sha256,
            resolved_config_sha256=selected.resolved_config_sha256,
            processor_contract=processor,
            action_predictor=unavailable,
            transition_predictor=unavailable,
            recovery_predictor=unavailable,
            memory_embedding=unavailable,
            frozen=True,
            training=False,
        ),
        e0=LoadedSelectedBackboneBackend(
            backbone_id=e0.backbone_id,
            backbone_revision=e0.backbone_revision,
            backbone_sha256=e0.backbone_sha256,
            resolved_config_sha256=e0.resolved_config_sha256,
            processor_contract=processor,
            base_prompt_sha256=e0.base_prompt_sha256,
            parser_id=e0.parser_id,
            parser_version=e0.parser_version,
            action_predictor=unavailable,
            frozen=True,
            training=False,
            adaptation_loaded=False,
            task_heads_loaded=False,
        ),
        parameter_fallback_resolver=unavailable,
    )


def test_import_has_no_torch_transformers_or_peft_side_effect() -> None:
    repository = Path(__file__).resolve().parents[2]
    code = (
        "import sys; "
        "import web_agent.runtime.qwen2vl_pc01; "
        "assert 'torch' not in sys.modules; "
        "assert 'transformers' not in sys.modules; "
        "assert 'peft' not in sys.modules"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository / "src")
    subprocess.run([sys.executable, "-c", code], check=True, env=environment)


def test_e0_parser_is_strict_and_never_repairs_output() -> None:
    assert parse_e0_action_output(
        '{"action_type":"CLICK","target":"submit",'
        '"bbox":[0.1,0.2,0.3,0.4],"value":null}'
    ) == {
        "action_type": "CLICK",
        "target": "submit",
        "bbox": [0.1, 0.2, 0.3, 0.4],
        "value": None,
    }
    invalid = (
        "```json\n{\"action_type\":\"CLICK\",\"target\":\"submit\","
        "\"bbox\":[0.1,0.2,0.3,0.4],\"value\":null}\n```",
        '{"action_type":"CLICK","target":"submit","bbox":null,'
        '"value":null,"confidence":1}',
        '{"action_type":"UNKNOWN","target":null,"bbox":null,"value":null}',
        '{"action_type":"TYPE","target":{},"bbox":[0.1,0.2,0.3,0.4],'
        '"value":"x"}',
        '{"action_type":"CLICK","action_type":"TYPE","target":null,'
        '"bbox":[0.1,0.2,0.3,0.4],"value":null}',
        '{"action_type":"CLICK","target":"submit","bbox":null,"value":null}',
        '{"action_type":"CLICK","target":"submit",'
        '"bbox":[0.9,0.2,0.3,0.4],"value":null}',
        '{"action_type":"CLICK","target":"submit",'
        '"bbox":[true,0.2,0.3,0.4],"value":null}',
        '{"action_type":"SCROLL","target":null,'
        '"bbox":[0.1,0.2,0.3,0.4],"value":"down"}',
    )
    for output in invalid:
        with pytest.raises(ActionParseError):
            parse_e0_action_output(output)


def test_e0_grounded_output_reaches_parameter_provider(monkeypatch) -> None:
    runtime = qwen2vl_pc01._UnadaptedBaseRuntime(
        model=object(),
        processor=object(),
        torch=object(),
    )
    monkeypatch.setattr(
        runtime,
        "_generate",
        lambda **_: (
            '{"action_type":"TYPE","target":"search box",'
            '"bbox":[0.1,0.2,0.3,0.4],"value":"failure-aware agent"}'
        ),
    )
    observation = PolicyObservation(
        task_id="task-1",
        goal="search",
        observation_id="observation-1",
        screenshot_sha256="a" * 64,
        screenshot_path=None,
        width=1280,
        height=720,
        url="https://example.test",
        title="Example",
    )
    decision = runtime.predict_action(
        RuntimeTaskView(task_id="task-1", goal="search"),
        observation,
        random.Random(1),
    )
    assert decision.action_type is ActionType.TYPE
    assert decision.bbox == (0.1, 0.2, 0.3, 0.4)
    assert decision.parameter_hints["text"] == "failure-aware agent"
    parameters = DeterministicParameterProvider().resolve(
        RuntimeTaskView(task_id="task-1", goal="search"),
        observation,
        decision,
        rng=random.Random(1),
    )
    assert parameters.action_type is ActionType.TYPE
    assert parameters.values == {
        "target_x": 0.25,
        "target_y": 0.4,
        "target_bbox": [0.1, 0.2, 0.3, 0.4],
        "text": "failure-aware agent",
    }


def test_frozen_prompt_constants_equal_registered_files() -> None:
    repository = Path(__file__).resolve().parents[2]
    assert (repository / "configs/eval/table2/prompts/e0_action_v1.txt").read_text(
        encoding="utf-8"
    ) == E0_ACTION_PROMPT
    assert (
        repository / "configs/eval/table2/prompts/parameter_provider_v1.txt"
    ).read_text(encoding="utf-8") == PARAMETER_PROVIDER_PROMPT
    assert REGISTERED_GENERATION_KWARGS == {
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 128,
    }


def test_matched_backend_factories_are_lazy_cached_and_selection_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected, e0 = _selections(tmp_path)
    loaded = _loaded_bundle(selected, e0)
    calls: list[tuple] = []

    def fake_loader(**kwargs):
        calls.append((kwargs["selected"], kwargs["e0"], kwargs["artifacts"]))
        return loaded

    monkeypatch.setattr(
        "web_agent.runtime.qwen2vl_pc01.load_pc01_runtime_backends",
        fake_loader,
    )
    factories = make_pc01_backend_factories(
        selected=selected,
        e0=e0,
        artifacts=_artifacts(tmp_path),
    )
    assert calls == []
    assert factories.selected_checkpoint_backend(selected) is loaded.selected
    assert factories.selected_backbone_backend(e0) is loaded.e0
    assert len(calls) == 1
    other = ValidationSelectedCheckpoint(
        manifest_id="other",
        model_seed=42,
        checkpoint_path=tmp_path / "other.ckpt",
        selected_checkpoint_sha256="1" * 64,
        resolved_config_sha256="2" * 64,
        processor_contract_sha256=_processor().record_sha256,
        validation_rows_read=7861,
    )
    with pytest.raises(PC01RuntimeError, match="another selection"):
        factories.selected_checkpoint_backend(other)


def test_live_binding_factory_is_explicit_and_fails_closed_without_services() -> None:
    with pytest.raises(PC01RuntimeError, match="concrete BrowserGym/WebArena"):
        PC01EvaluationRuntimeFactory(create_webarena_runtime=None, seed_services={})
    with pytest.raises(PC01RuntimeError, match="freezing all live deployment capabilities"):
        require_live_factory()


def test_live_binding_requirements_are_explicit_and_cannot_accept_untyped_services() -> None:
    assert PC01_LIVE_DEPLOYMENT_CAPABILITIES == (
        "deterministic WebArena service/account/database/start-state reset with "
        "hashes-only receipt",
        "exclusive browser-controller input audit proving zero manual rescue",
        "frozen oracle-blind BrowserGym observation/action/page-settle/screenshot mapping",
        "frozen action-safety and infrastructure-fault classification",
        "concrete REPLAN/ALTERNATIVE_TARGET recovery action planner",
        "complete per-episode model/resource measurement",
        "source-reviewed libwebarena evaluator port with pinned upstream/delta/config "
        "evidence, measured gpt-4-1106-preview availability, and write-only sealed output",
    )
    with pytest.raises(PC01RuntimeError, match="services must cover seed 42 exactly"):
        PC01EvaluationRuntimeFactory(
            create_webarena_runtime=lambda task: task,
            seed_services={43: object()},
        )
    with pytest.raises(PC01RuntimeError, match="must use PC01SeedOperationalServices"):
        PC01EvaluationRuntimeFactory(
            create_webarena_runtime=lambda task: task,
            seed_services={42: object()},
        )


def test_operational_service_record_rejects_untyped_placeholder_capabilities() -> None:
    with pytest.raises(PC01RuntimeError, match="typed episode-state resetter"):
        PC01SeedOperationalServices(
            episode_state_resetter=object(),
            recovery_action_planner=object(),
            begin_measurement=lambda *args: None,
            finish_measurement=lambda *args: {},
        )


def test_invalid_bbox_is_rejected_without_clipping_or_repair() -> None:
    assert qwen2vl_pc01._bounded_bbox((0.1, 0.2, 0.3, 0.4)) == (
        0.1,
        0.2,
        0.3,
        0.4,
    )
    for invalid in (
        (-0.01, 0.2, 0.3, 0.4),
        (0.8, 0.2, 0.3, 0.4),
        (0.1, 0.8, 0.3, 0.3),
        (0.1, 0.2, float("nan"), 0.4),
    ):
        with pytest.raises(PC01RuntimeError, match="bbox"):
            qwen2vl_pc01._bounded_bbox(invalid)
