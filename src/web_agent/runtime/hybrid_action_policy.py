"""H0–H3 normal acting: frozen generation with optional frozen policy advice.

This adapter owns its model dispatch accounting; do not wrap it in another
CallablePolicyAdapter. The episode runner still owns execution, feedback and
recovery budgets. No model, decoding, parameter-provider or executor is changed.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import random
from threading import Lock

from web_agent.benchmarks.miniwob_controls import prompt_controls
from web_agent.runtime.contracts import (
    ConcreteAction, PolicyObservation, PreActionDecision, canonical_sha256,
    detached_record_copy, runtime_task_view,
)
from web_agent.runtime.named_target_policy import resolve_named_target
from web_agent.runtime.observation import assert_oracle_blind_mapping
from web_agent.runtime.policy import (
    ActionParseError, CallablePolicyAdapter, PolicyAdapter, PolicyError, PolicyKind,
)
from web_agent.runtime.qwen2vl_pc01 import _UnadaptedBaseRuntime, _strict_json_object

VERSION = "hybrid-normal-action-v1"
PROMPT_PATH = Path(__file__).resolve().parents[3] / "configs/eval/table2/miniwob_hybrid_action_prompt_v1.txt"


def trained_advice(decision: PreActionDecision, *, checkpoint_sha256: str) -> dict:
    """Detached, lossless projection of the registered pre-action outputs."""
    decision = detached_record_copy(decision)
    probabilities = dict(decision.action_probabilities)
    if probabilities[decision.action_type.value] != max(probabilities.values()):
        raise PolicyError("hybrid advice requires the unmasked trained argmax")
    return {
        "schema": "table2.trained-advice.v1",
        "observation_id": decision.observation_id,
        "input_observation_ids": list(decision.input_observation_ids),
        "checkpoint_sha256": checkpoint_sha256,
        "source_decision_sha256": decision.record_sha256,
        "policy_id": decision.policy_id, "policy_version": decision.policy_version,
        "action_probabilities": probabilities, "argmax": decision.action_type.value,
        "bbox": list(decision.bbox) if decision.bbox is not None else None,
        "grounding_confidence": decision.grounding_confidence,
        "confidence_before": decision.confidence_before,
        "advisory_only": True,
    }


@dataclass(frozen=True)
class HybridActionContext:
    """Runner-owned completed records; never inferred from a task solution.

    History fingerprints bind exact issued parameters to the already-observed
    transitions. The optional rejection is supplied by the runner at the current
    observation (including parser failures which have no ConcreteAction).
    """

    episode_id: str
    observation_id: str
    completed_actions: tuple[ConcreteAction, ...] = ()
    last_rejection: dict | None = None

    def project(self, observation: PolicyObservation) -> dict:
        if not self.episode_id or self.observation_id != observation.observation_id:
            raise PolicyError("hybrid context observation binding mismatch")
        if len(self.completed_actions) != len(observation.causal_history):
            raise PolicyError("hybrid context requires the complete causal action history")
        actions = []
        for action, history in zip(self.completed_actions, observation.causal_history):
            if (type(action) is not ConcreteAction or action.action_id != history.action_id
                    or action.fingerprint != history.action_target_fingerprint
                    or action.action_type != history.action_type):
                raise PolicyError("hybrid history action/fingerprint mismatch")
            actions.append({
                "action_id": action.action_id, "action_type": action.action_type.value,
                "parameters": deepcopy(dict(action.parameters)),
                "bbox": list(action.bbox) if action.bbox is not None else None,
                "execution_status": history.execution_status.value,
                "state_changed": history.state_changed,
                "post_observation_id": history.post_observation_id,
                "execution_error_sha256": history.execution_error_sha256,
            })
        rejection = deepcopy(self.last_rejection)
        if rejection is not None:
            if (set(rejection) != {"observation_id", "stage", "code", "detail"}
                    or any(type(v) is not str for v in rejection.values())
                    or rejection["observation_id"] != observation.observation_id
                    or rejection["stage"] not in {
                        "parsing", "target_resolution", "parameter_resolution",
                        "strategy_validation", "browser_execution"}
                    or not rejection["code"].strip()):
                raise PolicyError("hybrid rejection has invalid fields or observation binding")
        result = {
            "schema": "table2.hybrid-action-context.v1", "episode_id": self.episode_id,
            "task_id": observation.task_id, "observation_id": observation.observation_id,
            "screenshot_sha256": observation.screenshot_sha256,
            "url": observation.url, "title": observation.title,
            "current_controls": deepcopy(prompt_controls(observation)),
            "completed_actions": actions, "last_rejection": rejection,
        }
        assert_oracle_blind_mapping(result)
        return result


class _HybridGenerator(_UnadaptedBaseRuntime):
    """Reuse the existing generation decoder and PreActionDecision conversion."""

    def __init__(self, *, base, prompt, context, receipt, policy_id, interface_version=1):
        self.base = base
        self.prompt = prompt
        self.context = context
        self.receipt = receipt
        self.policy_id = policy_id
        self.interface_version = interface_version
        self.version = VERSION if interface_version == 1 else f'hybrid-normal-action-v{interface_version}'

    def _generate(self, *, prompt, task, observation, suffix):
        suffix = "\nhybrid_action_context: " + json.dumps(self.context, sort_keys=True)
        self.receipt.update(input_suffix=suffix,
                            input_suffix_sha256=hashlib.sha256(suffix.encode()).hexdigest())
        if self.interface_version == 3 and callable(getattr(self.base, '_generate_with_metadata', None)):
            raw, metadata = self.base._generate_with_metadata(
                prompt=self.prompt, task=task, observation=observation, suffix=suffix)
            self.receipt['generation_metadata'] = deepcopy(metadata)
        else:
            raw = self.base._generate(prompt=self.prompt, task=task, observation=observation, suffix=suffix)
            if self.interface_version == 3:
                self.receipt['generation_metadata'] = {
                    'available': False, 'reason': 'backend_does_not_expose_generation_metadata'}
        self.receipt["raw_response"] = raw
        self.receipt["stage"] = "parsing"
        if self.interface_version in {2, 3}:
            from web_agent.runtime.hybrid_interface import parse_action
            parsed, resolution = parse_action(raw, observation, interface_version=self.interface_version)
            self.receipt.update(resolved_proposal=parsed, target_resolution=resolution)
            return json.dumps(parsed)
        # The hybrid contract deliberately requires all four fields and one
        # bare object. Existing E-profile permissive parsing remains unchanged.
        value = _strict_json_object(raw, context=VERSION)
        if set(value) != {"action_type", "target", "bbox", "value"}:
            raise ActionParseError("hybrid output requires exactly four action fields")
        self.receipt["stage"] = "target_resolution"
        parsed, resolution = resolve_named_target(raw, observation, allow_role_target=True)
        # CLICK may carry an explicit label in value in the historical parser.
        # Keep that issued value in the new decision/evidence as well.
        parsed["value"] = value["value"]
        self.receipt.update(resolved_proposal=parsed, target_resolution=resolution)
        return json.dumps(parsed)

    def predict_action(self, task, observation, rng):
        decision = super().predict_action(task, observation, rng)
        if self.interface_version == 3:
            control_id = self.receipt.get('target_resolution', {}).get('resolved_control_id')
            if control_id is not None:
                decision = replace(decision, parameter_hints={
                    **dict(decision.parameter_hints), 'resolved_control_id': control_id})
        return replace(decision, policy_id=self.policy_id, policy_version=self.version,
                       decision_id=f"{self.policy_id}-{canonical_sha256(self.receipt)[:24]}")


class HybridActionPolicy(PolicyAdapter):
    """One shared normal actor, with separately accounted advice in H1–H3.

    Construct once per episode/system with a fresh evidence directory. Pass the
    already guarded trained CallablePolicyAdapter, never a raw trained callback.
    H2/H3 assessment methods delegate unchanged to that trained adapter; recovery
    orchestration and memory remain responsibilities of the episode runner.
    """

    policy_version = VERSION

    def __init__(self, *, system: str, episode_id: str, base,
                 evidence_dir: str | Path, trained_policy: CallablePolicyAdapter | None = None,
                 interface_version: int = 1):
        if interface_version not in {1, 2, 3}:
            raise ValueError('unsupported hybrid interface version')
        self.interface_version = interface_version
        self.policy_version = VERSION if interface_version == 1 else f'hybrid-normal-action-v{interface_version}'
        if system not in {"H0", "H1", "H2", "H3"}:
            raise ValueError("hybrid system must be H0–H3")
        if not episode_id:
            raise ValueError("hybrid policy requires an episode binding")
        if system == "H0" and trained_policy is not None:
            raise ValueError("H0 must not receive a trained policy")
        if system != "H0":
            if (not isinstance(trained_policy, CallablePolicyAdapter)
                    or trained_policy.kind is not PolicyKind.TRAINED):
                raise ValueError("H1–H3 require a guarded trained policy")
            backend = getattr(trained_policy.action_predictor, "__self__", None)
            if getattr(backend, "restrict_to_executable_actions", False):
                raise ValueError("hybrid advice must not use executable-action masking")
        self.system = system
        self.episode_id = episode_id
        self.policy_id = f"miniwob-{system.lower()}-normal"
        self.kind = PolicyKind.BASE if system == "H0" else PolicyKind.TRAINED
        self.checkpoint_sha256 = trained_policy.checkpoint_sha256 if trained_policy else None
        self.trained_policy = trained_policy
        self.base = base
        self.prompt = (PROMPT_PATH if interface_version == 1 else
                       PROMPT_PATH.with_name(f'miniwob_hybrid_action_prompt_v{interface_version}.txt')).read_text(encoding="utf-8")
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=False)
        self._lock = Lock()
        self._calls = 0

    def predict_action(self, task, observation, *, rng: random.Random,
                       context: HybridActionContext | None = None) -> PreActionDecision:
        task = runtime_task_view(task)
        context = context or HybridActionContext(self.episode_id, observation.observation_id)
        if context.episode_id != self.episode_id:
            raise PolicyError("hybrid context belongs to another episode")
        projection = context.project(observation)
        with self._lock:
            self._calls += 1
            receipt = {
                "schema": self.policy_version, "system": self.system, "episode_id": self.episode_id,
                "observation_id": observation.observation_id,
                "observation_sha256": observation.record_sha256,
                "observation": observation.to_dict(),
                "task_sha256": task.record_sha256,
                "task": task.to_dict(),
                "prompt": self.prompt,
                "prompt_sha256": hashlib.sha256(self.prompt.encode()).hexdigest(),
                "status": "STARTED", "stage": "trained_advice" if self.trained_policy else "generation",
            }
            try:
                if self.trained_policy is not None:
                    backend = getattr(self.trained_policy.action_predictor, "__self__", None)
                    if getattr(backend, "restrict_to_executable_actions", False):
                        raise PolicyError("hybrid advice must not use executable-action masking")
                    advice_rng = random.Random()
                    advice_rng.setstate(rng.getstate())
                    decision = self.trained_policy.predict_action(task, observation, rng=advice_rng)
                    advice = trained_advice(decision, checkpoint_sha256=self.checkpoint_sha256)
                    receipt["trained_decision"] = decision.to_dict()
                    receipt["trained_advice"] = deepcopy(advice)
                    projection["trained_advice"] = advice
                receipt["stage"] = "generation"
                if self.interface_version in {2, 3}:
                    from web_agent.runtime.hybrid_interface import semantic_context
                    receipt['audit_context'] = deepcopy(projection)
                    projection = semantic_context(projection, episode_id=self.episode_id,
                                                  interface_version=self.interface_version)
                backend = _HybridGenerator(base=self.base, prompt=self.prompt, context=projection,
                                           receipt=receipt, policy_id=self.policy_id,
                                           interface_version=self.interface_version)
                generator = CallablePolicyAdapter(
                    policy_id=self.policy_id, policy_version=self.policy_version, kind=PolicyKind.BASE,
                    action_predictor=backend.predict_action)
                result = generator.predict_action(task, observation, rng=rng)
                receipt.update(status="RESOLVED", decision=result.to_dict())
                return result
            except Exception as exc:
                if isinstance(exc, ActionParseError):
                    exc.failure_stage = getattr(exc, "failure_stage", receipt["stage"])
                receipt.update(status="REJECTED" if isinstance(exc, ActionParseError) else "ERROR",
                               error_type=type(exc).__name__, error=str(exc),
                               failure_stage=getattr(exc, "failure_stage", receipt["stage"]),
                               diagnostic_code=getattr(exc, "diagnostic_code", None))
                if self.interface_version == 3 and isinstance(exc, ActionParseError):
                    receipt['diagnostic'] = deepcopy(getattr(exc, 'diagnostic', None))
                    if hasattr(exc, 'proposed_action'):
                        receipt['proposed_action'] = deepcopy(exc.proposed_action)
                raise
            finally:
                # Never overwrite a previous attempt or confuse proposal with execution.
                with (self.evidence_dir / f"proposal-{self._calls:04d}.json").open("x") as stream:
                    json.dump(receipt, stream, indent=2, allow_nan=False)
                    stream.write("\n")

    def assess_transition(self, task, transition, *, rng):
        if self.system not in {"H2", "H3"}:
            raise PolicyError("H0/H1 have no learned recovery path")
        return self.trained_policy.assess_transition(task, transition, rng=rng)

    def assess_recovery(self, task, transition, *, rng):
        if self.system not in {"H2", "H3"}:
            raise PolicyError("H0/H1 have no learned recovery path")
        return self.trained_policy.assess_recovery(task, transition, rng=rng)
