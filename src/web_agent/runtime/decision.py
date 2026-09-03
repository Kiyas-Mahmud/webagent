"""Oracle-blind recovery trigger, no-memory shadow, and loop guard."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

from web_agent.runtime.contracts import (
    ConcreteAction,
    ExecutionResult,
    ExecutionStatus,
    Observation,
    RecoveryDecision,
    RecoveryStrategy,
    TransitionAssessment,
    VersionedRecord,
    canonical_sha256,
)
from web_agent.runtime.protocol import (
    LoopRule,
    RecoveryTriggerConfig,
    SystemSwitches,
)


@dataclass(frozen=True, slots=True)
class RecoveryTrigger(VersionedRecord):
    triggered: bool
    sources: tuple[str, ...]
    diagnosis: str

    def __post_init__(self) -> None:
        allowed = {"policy", "executor", "loop_guard"}
        if set(self.sources) - allowed:
            raise ValueError("recovery trigger contains oracle/unknown source")
        if self.triggered != bool(self.sources):
            raise ValueError("trigger flag and sources disagree")


class LoopGuard:
    """Deterministic loop detector over agent-observable states/actions only."""

    def __init__(self, rule: LoopRule) -> None:
        self.rule = rule
        self._states: deque[str] = deque(maxlen=rule.rolling_window)
        self._pairs: deque[str] = deque(maxlen=rule.rolling_window)

    def reset(self) -> None:
        self._states.clear()
        self._pairs.clear()

    def record(self, observation: Observation, action: ConcreteAction) -> bool:
        state_values = {
            "screenshot_sha256": observation.screenshot_sha256,
            "url": observation.url,
            "title": observation.title,
            "page_state": observation.page_state,
        }
        action_values = {
            "action_type": action.action_type.value,
            "parameters": action.parameters,
            "bbox": action.bbox,
        }
        state_fingerprint = canonical_sha256(
            {field: state_values[field] for field in self.rule.state_fingerprint_fields}
        )
        action_fingerprint = canonical_sha256(
            {
                field: action_values[field]
                for field in self.rule.action_target_fingerprint_fields
            }
        )
        pair = canonical_sha256(
            {"state": state_fingerprint, "action": action_fingerprint}
        )
        self._states.append(state_fingerprint)
        self._pairs.append(pair)
        if Counter(self._pairs).most_common(1)[0][1] >= self.rule.equivalent_repetition_count:
            return True
        if Counter(self._states).most_common(1)[0][1] >= self.rule.equivalent_repetition_count:
            return True
        if self.rule.detect_abab_cycle and len(self._pairs) >= 4:
            a, b, c, d = list(self._pairs)[-4:]
            if a == c and b == d and a != b:
                return True
        return False


class DecisionCombiner:
    """Primary E0-E3 decision logic; verifier types are intentionally absent."""

    def __init__(
        self,
        switches: SystemSwitches,
        trigger_config: RecoveryTriggerConfig,
    ) -> None:
        if type(trigger_config) is not RecoveryTriggerConfig:
            raise TypeError("DecisionCombiner requires the registered typed trigger config")
        self.switches = switches
        self.trigger_config = trigger_config

    def recovery_trigger(
        self,
        *,
        assessment: TransitionAssessment | None,
        execution: ExecutionResult,
        loop_detected: bool,
    ) -> RecoveryTrigger:
        if not self.switches.recovery_controller:
            return RecoveryTrigger(False, (), "recovery controller disabled")
        sources: list[str] = []
        diagnosis_parts: list[str] = []
        if assessment is not None:
            policy_signals = (
                assessment.predicted_failure is True,
                assessment.failure_probability
                >= self.trigger_config.failure_probability_threshold,
                assessment.needs_recovery is True,
                assessment.needs_recovery_probability
                >= self.trigger_config.needs_recovery_probability_threshold,
            )
            if self.trigger_config.policy_trigger(policy_signals):
                sources.append("policy")
                diagnosis_parts.append(assessment.failure_type)
        if execution.status is not ExecutionStatus.EXECUTED:
            sources.append("executor")
            diagnosis_parts.append(execution.error_kind or execution.status.value)
        if loop_detected:
            sources.append("loop_guard")
            diagnosis_parts.append("LOOP_DETECTED")
        return RecoveryTrigger(
            triggered=bool(sources),
            sources=tuple(dict.fromkeys(sources)),
            diagnosis=";".join(dict.fromkeys(diagnosis_parts)) or "NONE",
        )

    def no_memory_shadow(
        self,
        *,
        trigger: RecoveryTrigger,
        assessment: TransitionAssessment | None,
        incident_id: str,
        planned_action: ConcreteAction | None = None,
    ) -> RecoveryDecision:
        """Compute the E2-equivalent decision before any E3 retrieval."""
        if not trigger.triggered:
            raise ValueError("cannot form recovery decision without a trigger")
        if assessment is not None and assessment.recovery_strategy is not RecoveryStrategy.NONE:
            strategy = assessment.recovery_strategy
        elif "loop_guard" in trigger.sources:
            strategy = RecoveryStrategy.BACKTRACK
        else:
            strategy = RecoveryStrategy.RETRY
        shadow_payload = {
            "sources": trigger.sources,
            "diagnosis": trigger.diagnosis,
            "strategy": strategy.value,
        }
        decision_id = (
            f"{incident_id}:shadow:{canonical_sha256(shadow_payload)[:12]}"
        )
        return RecoveryDecision(
            decision_id=decision_id,
            incident_id=incident_id,
            strategy=strategy,
            trigger_sources=trigger.sources,
            diagnosis=trigger.diagnosis,
            planned_action=planned_action,
        )
