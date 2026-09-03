"""Synchronous, evidence-first E0-E3 episode state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from collections.abc import Callable
from typing import Any, Mapping

from web_agent.runtime.action_parameters import (
    ActionParameterProvider,
    HybridParameterProvider,
    HybridParameterResolutionError,
    ParameterResolutionError,
    concretize,
)
from web_agent.runtime.contracts import (
    ActionParameters,
    CausalHistoryEntry,
    ConcreteAction,
    EPISODE_FOREIGN_KEY_RECEIPT_VERSION,
    EpisodeContractBundle,
    EpisodeSummary,
    ExecutionResult,
    ExecutionStatus,
    MemoryQuery,
    MemoryQueryResult,
    Observation,
    OpaqueTerminalSignal,
    PolicyObservation,
    PreActionDecision,
    PreActionParseRejection,
    RecoveryAssessment,
    RecoveryAttempt,
    RecoveryDecision,
    RecoveryTransitionInput,
    RuntimeTaskView,
    SystemID,
    TaskSpecification,
    TerminalReason,
    TransitionAssessment,
    TransitionInput,
    VerifierReceiptBinding,
    canonical_sha256,
    completed_causal_history_entry,
    detached_record_copy,
    runtime_task_view,
    validate_episode_foreign_keys,
)
from web_agent.runtime.decision import DecisionCombiner, LoopGuard, RecoveryTrigger
from web_agent.runtime.duplicate_audit import (
    DuplicateAuditError,
    FrozenDuplicateAuditManifest,
)
from web_agent.runtime.event_log import EpisodeEventLogs
from web_agent.runtime.executor import (
    EpisodeTimeout,
    Executor,
    ExecutorBudgetExceeded,
)
from web_agent.runtime.memory_adapter import (
    MemoryAdapter,
    MemoryBoundaryError,
    PostFailureEmbeddingRequest,
)
from web_agent.runtime.model_calls import (
    ModelCallBudgetExceeded,
    ModelCallLedger,
    activate_model_call_ledger,
    deactivate_model_call_ledger,
)
from web_agent.runtime.observation import (
    ObservationBuilder,
    ProcessorParityContract,
    validate_processor_parity,
)
from web_agent.runtime.policy import ActionParseError, PolicyError, SystemPolicy
from web_agent.runtime.protocol import (
    RuntimeProtocol,
    RuntimeStage,
    StageRNGFactory,
    StageSeedPlan,
    StageSeedUse,
)
from web_agent.runtime.recovery.controller import (
    RecoveryBudgetExceeded,
    RecoveryController,
)
from web_agent.runtime.state_reset import (
    PreBrowserSetupEvidence,
    WebArenaResetStateReceipt,
)


@dataclass(slots=True)
class _Counters:
    normal_actions: int = 0
    recovery_actions: int = 0
    failure_incidents: int = 0
    memory_queries: int = 0
    memory_interventions: int = 0


@dataclass(slots=True)
class _EpisodeContracts:
    """Unredacted, episode-local records retained only for causal validation."""

    observations: list[Observation] = field(default_factory=list)
    decisions: list[PreActionDecision] = field(default_factory=list)
    actions: list[ConcreteAction] = field(default_factory=list)
    executions: list[ExecutionResult] = field(default_factory=list)
    transition_inputs: list[TransitionInput] = field(default_factory=list)
    transition_assessments: list[TransitionAssessment] = field(default_factory=list)
    recovery_decisions: list[RecoveryDecision] = field(default_factory=list)
    recovery_attempts: list[RecoveryAttempt] = field(default_factory=list)
    recovery_transitions: list[RecoveryTransitionInput] = field(default_factory=list)
    recovery_assessments: list[RecoveryAssessment] = field(default_factory=list)
    memory_queries: list[MemoryQuery] = field(default_factory=list)
    memory_results: list[MemoryQueryResult] = field(default_factory=list)
    parse_rejections: list[PreActionParseRejection] = field(default_factory=list)

    def register_recovery_decision(self, decision: RecoveryDecision) -> None:
        """Register a decision identity once; repeated attempts may reuse it."""

        for existing in self.recovery_decisions:
            if existing.decision_id != decision.decision_id:
                continue
            if existing.record_sha256 != decision.record_sha256:
                raise ValueError(
                    "recovery decision ID was reused with different canonical content"
                )
            return
        self.recovery_decisions.append(decision)

    def bundle(
        self,
        *,
        episode_id: str,
        summary: EpisodeSummary,
    ) -> EpisodeContractBundle:
        return EpisodeContractBundle(
            episode_id=episode_id,
            observations=tuple(self.observations),
            decisions=tuple(self.decisions),
            actions=tuple(self.actions),
            executions=tuple(self.executions),
            transition_inputs=tuple(self.transition_inputs),
            transition_assessments=tuple(self.transition_assessments),
            recovery_decisions=tuple(self.recovery_decisions),
            recovery_attempts=tuple(self.recovery_attempts),
            recovery_transitions=tuple(self.recovery_transitions),
            recovery_assessments=tuple(self.recovery_assessments),
            memory_queries=tuple(self.memory_queries),
            memory_results=tuple(self.memory_results),
            parse_rejections=tuple(self.parse_rejections),
            summary=summary,
        )


class EpisodeRunner:
    """Run one system/task/repeat/seed package without reading sealed truth."""

    def __init__(
        self,
        *,
        protocol: RuntimeProtocol,
        system_policy: SystemPolicy,
        provider: ActionParameterProvider,
        executor: Executor,
        recovery_controller: RecoveryController | None = None,
        memory_adapter: MemoryAdapter | None = None,
        observation_builder: ObservationBuilder | None = None,
        training_processor_contract: ProcessorParityContract | None = None,
        duplicate_audit_registry: FrozenDuplicateAuditManifest | None = None,
        event_logs: EpisodeEventLogs | None = None,
        before_environment_reset: Callable[[], PreBrowserSetupEvidence] | None = None,
        defer_contract_validation_receipt: bool = False,
    ) -> None:
        self.protocol = protocol
        self.system_policy = system_policy
        self.switches = system_policy.switches
        self.provider = provider
        self.executor = executor
        self.recovery_controller = recovery_controller
        self.memory_adapter = memory_adapter
        self.observation_builder = observation_builder or ObservationBuilder()
        self.training_processor_contract = training_processor_contract
        self.duplicate_audit_registry = duplicate_audit_registry
        self.event_logs = event_logs
        self.before_environment_reset = before_environment_reset
        if type(defer_contract_validation_receipt) is not bool:
            raise TypeError("receipt deferral flag must be an exact boolean")
        self.defer_contract_validation_receipt = defer_contract_validation_receipt
        self._contract_validation_receipt: dict[str, Any] | None = None
        self.decision = DecisionCombiner(
            self.switches,
            protocol.recovery_trigger,
        )
        self.loop_guard = LoopGuard(protocol.loop_rule)
        if executor.budgets != protocol.budgets:
            raise ValueError("executor and protocol budgets differ")
        if protocol.evaluation_mode:
            if not isinstance(provider, HybridParameterProvider):
                raise ValueError(
                    "evaluation runtime requires the registered hybrid parameter provider"
                )
            if provider.provider_id != protocol.provider_id:
                raise ValueError("runtime provider identity differs from frozen protocol")
            if provider.prompt_sha256 != protocol.provider_prompt_sha256:
                raise ValueError("runtime provider prompt differs from frozen protocol")
            if (
                provider.frozen_base_fallback.prompt_sha256
                != protocol.provider_prompt_sha256
            ):
                raise ValueError("hybrid fallback prompt differs from frozen protocol")
            if dict(provider.decoding_parameters) != dict(
                provider.frozen_base_fallback.decoding_parameters
            ):
                raise ValueError("hybrid fallback decoding differs from common provider")
            if not provider.frozen:
                raise ValueError("evaluation parameter provider must be frozen")
        if self.switches.recovery_controller != (recovery_controller is not None):
            raise ValueError("recovery controller presence disagrees with system switches")
        if self.switches.memory_query != (memory_adapter is not None):
            raise ValueError("memory adapter presence disagrees with system switches")
        if memory_adapter is not None and (
            memory_adapter.evaluation_mode != protocol.evaluation_mode
        ):
            raise ValueError(
                "memory adapter evaluation attestation mode differs from runtime protocol"
            )
        if self.switches.memory_query != (duplicate_audit_registry is not None):
            raise ValueError(
                "E3 requires a frozen duplicate-audit registry; E0-E2 must not receive one"
            )
        if recovery_controller and recovery_controller.budgets != protocol.budgets:
            raise ValueError("recovery controller and protocol budgets differ")
        if training_processor_contract is not None:
            runtime_processor = self.observation_builder.processor_contract
            if runtime_processor is None:
                raise ValueError(
                    "training processor contract supplied without runtime contract"
                )
            # Constructor-time validation guarantees mismatch failure before
            # reset, policy inference, browser execution, or verifier access.
            validate_processor_parity(training_processor_contract, runtime_processor)

    @property
    def contract_validation_receipt(self) -> Mapping[str, Any] | None:
        """Return the latest digest-only receipt, never the unredacted bundle."""

        if self._contract_validation_receipt is None:
            return None
        return dict(self._contract_validation_receipt)

    def run(
        self,
        task: TaskSpecification,
        *,
        repeat_id: int,
        model_seed: int,
        stage_seeds: Mapping[str, int] | None = None,
    ) -> EpisodeSummary:
        if repeat_id < 0:
            raise ValueError("repeat_id cannot be negative")
        self._contract_validation_receipt = None
        callback_task = runtime_task_view(task)
        episode_id = (
            f"{self.protocol.campaign_id}:{self.switches.system_id.value}:"
            f"{task.task_id}:repeat-{repeat_id}:seed-{model_seed}"
        )
        if self.event_logs is not None and self.event_logs.episode_id != episode_id:
            raise ValueError(
                "event-log episode_id differs from the deterministic runner episode_id"
            )
        if self.memory_adapter is not None:
            # The per-seed immutable store must be paired with the same model
            # seed before reset or any browser/evaluator interaction occurs.
            self.memory_adapter.assert_model_seed(model_seed)
            assert self.duplicate_audit_registry is not None
            task_partition = str(task.metadata.get("task_partition") or "normal")
            try:
                duplicate_cluster_ids = self.duplicate_audit_registry.clusters_for(
                    task.task_id,
                    task_partition=task_partition,
                    allow_synthetic_diagnostic=(task_partition == "recovery_diagnostic"),
                )
            except DuplicateAuditError as exc:
                raise MemoryBoundaryError(
                    f"E3 duplicate-audit boundary rejected task before reset: {exc}"
                ) from exc
            self._log(
                "environment_events",
                "duplicate_audit_binding",
                {
                    "manifest_id": self.duplicate_audit_registry.manifest_id,
                    "manifest_sha256": self.duplicate_audit_registry.manifest_sha256,
                    "entry_binding_sha256": (
                        self.duplicate_audit_registry.entry_binding_sha256(task.task_id)
                    ),
                    "task_partition": task_partition,
                    "duplicate_cluster_ids": list(duplicate_cluster_ids),
                },
            )
            assert self.memory_adapter.reader is not None
            self._log(
                "environment_events",
                "memory_store_binding",
                {
                    "reader_id": self.memory_adapter.reader.reader_id,
                    "reader_evidence_scope": (
                        self.memory_adapter.reader.evidence_scope
                    ),
                    "store_manifest_sha256": (
                        self.memory_adapter.reader.store_manifest_sha256
                    ),
                    "expected_store_manifest_sha256": (
                        self.memory_adapter.expected_store_manifest_sha256
                    ),
                    "model_seed": self.memory_adapter.reader.model_seed,
                    "frozen": self.memory_adapter.reader.frozen,
                    "write_enabled": self.memory_adapter.reader.write_enabled,
                },
            )
        else:
            duplicate_cluster_ids = ()
        rng_factory = self.protocol.rng_factory()
        if stage_seeds is None:
            if self.protocol.require_frozen_stage_seed_plan:
                raise ValueError(
                    "frozen protocol execution requires schedule stage_seeds"
                )
            seed_plan = rng_factory.seed_plan(
                task_id=task.task_id,
                repeat_id=repeat_id,
                matched_seed=model_seed,
            )
        else:
            seed_plan = StageSeedPlan(
                protocol_id=self.protocol.protocol_id,
                campaign_id=self.protocol.campaign_id,
                campaign_seed=self.protocol.campaign_seed,
                task_id=task.task_id,
                repeat_id=repeat_id,
                matched_seed=model_seed,
                decision_index=0,
                stage_seeds=dict(stage_seeds),
            )
            rng_factory.validate_seed_plan(seed_plan)
        self._log("environment_events", "rng_seed_plan", seed_plan)
        if self.training_processor_contract is not None:
            runtime_processor = self.observation_builder.processor_contract
            assert runtime_processor is not None
            self._log(
                "environment_events",
                "processor_parity",
                {
                    "training": self.training_processor_contract.to_dict(),
                    "runtime": runtime_processor.to_dict(),
                    "matched": True,
                },
            )
        counters = _Counters()
        contracts = _EpisodeContracts()
        causal_history: list[CausalHistoryEntry] = []
        model_call_ledger = ModelCallLedger(
            episode_id=episode_id,
            maximum=self.protocol.budgets.max_model_calls_per_episode,
            sink=lambda payload: self._log(
                "environment_events",
                "model_call_dispatch",
                payload,
            ),
        )
        model_call_token = activate_model_call_ledger(model_call_ledger)
        terminal_reason = TerminalReason.CLOSED
        valid_for_primary = True
        environment_failure = False
        reset_already_success = False
        final_signal: OpaqueTerminalSignal | None = None
        current: Observation | None = None
        self.loop_guard.reset()
        if self.recovery_controller:
            self.recovery_controller.reset()
        try:
            reset_seed = self._stage_seed(
                rng_factory,
                task_id=task.task_id,
                repeat_id=repeat_id,
                matched_seed=model_seed,
                stage=RuntimeStage.RESET,
                decision_index=0,
            )
            if (
                self.protocol.evaluation_mode
                and task.benchmark_id.lower() == "webarena"
                and self.before_environment_reset is None
            ):
                raise RuntimeError(
                    "evaluation WebArena reset lacks pre-browser setup evidence"
                )
            setup_evidence: PreBrowserSetupEvidence | None = None
            if self.before_environment_reset is not None:
                setup_evidence = self.before_environment_reset()
                if type(setup_evidence) is not PreBrowserSetupEvidence:
                    raise TypeError(
                        "pre-browser setup callback returned the wrong evidence contract"
                    )
                if (
                    setup_evidence.episode_id != episode_id
                    or setup_evidence.system_id is not self.switches.system_id
                ):
                    raise ValueError(
                        "pre-browser setup evidence cites another episode/system"
                    )
            current = self.executor.reset(
                task,
                episode_id=episode_id,
                seed=reset_seed,
            )
            contracts.observations.append(current)
            reset_state_receipt = self.executor.reset_state_receipt()
            if self.protocol.evaluation_mode and task.benchmark_id.lower() == "webarena":
                if type(reset_state_receipt) is not WebArenaResetStateReceipt:
                    raise RuntimeError(
                        "evaluation WebArena reset lacks its typed hashes-only receipt"
                    )
            if setup_evidence is not None:
                # The evidence is written only after ``Executor.reset`` has
                # started the 600-second clock.  This avoids placing event-log
                # I/O in an unmeasured gap between the two registered clocks.
                self._log(
                    "environment_events",
                    "pre_browser_setup_boundary",
                    {
                        "evidence": setup_evidence.to_dict(),
                        "evidence_sha256": setup_evidence.record_sha256,
                    },
                )
            if reset_state_receipt is not None:
                if type(reset_state_receipt) is not WebArenaResetStateReceipt:
                    raise TypeError("adapter returned an unregistered reset-state receipt")
                expected_reset_identity = (
                    episode_id,
                    task.task_id,
                    task.benchmark_version,
                    task.start_state_id,
                    reset_seed,
                )
                actual_reset_identity = (
                    reset_state_receipt.episode_id,
                    reset_state_receipt.task_id,
                    reset_state_receipt.benchmark_version,
                    reset_state_receipt.start_state_id,
                    reset_state_receipt.reset_stage_seed,
                )
                if actual_reset_identity != expected_reset_identity:
                    raise ValueError(
                        "WebArena reset-state receipt differs from runtime reset identity"
                    )
                self._log(
                    "environment_events",
                    "webarena_reset_state_receipt",
                    {
                        "receipt": reset_state_receipt.to_dict(),
                        "receipt_sha256": reset_state_receipt.record_sha256,
                    },
                )
            self._log_observation("reset", current)
            reset_binding = self._receipt_binding(
                "after_reset",
                observation=current,
            )
            reset_signal = self.executor.terminal_signal(reset_binding)
            self._log_terminal_receipt(reset_binding, reset_signal)
            # ``Observation.environment_error`` is an oracle-blind, observable
            # page/browser outcome.  It is not authorization for an
            # infrastructure rerun.  Only a typed InfrastructureInvalidError
            # raised by the benchmark boundary can invalidate the paired block.
            # The agent may therefore act on an error page visible after reset.
            if reset_signal.terminate:
                # The registered reset contract defines a terminal initial
                # state as already complete. Runtime still receives no
                # success/progress label, only the stop bit.
                final_signal = reset_signal
                reset_already_success = True
                valid_for_primary = False
                terminal_reason = TerminalReason.RESET_ALREADY_SUCCESS

            decision_index = 0
            while terminal_reason is TerminalReason.CLOSED:
                self.executor.require_time_remaining()
                if current is None:  # pragma: no cover - construction invariant
                    raise RuntimeError("episode has no current observation")
                decision_index += 1
                prior_history = tuple(causal_history)
                pre_policy = self.observation_builder.pre_action(
                    task,
                    current,
                    causal_history=prior_history,
                )
                pre_action_rng = self._stage_random(
                    rng_factory,
                    task_id=task.task_id,
                    repeat_id=repeat_id,
                    matched_seed=model_seed,
                    stage=RuntimeStage.PRE_ACTION,
                    decision_index=decision_index,
                )
                try:
                    decision = self.executor.run_blocking(
                        "pre-action policy inference",
                        lambda: self.system_policy.predict_action(
                            callback_task,
                            pre_policy,
                            rng=pre_action_rng,
                        ),
                    )
                except ActionParseError as exc:
                    # This catch is intentionally scoped to pre-action
                    # inference.  Post-action/recovery PolicyError instances
                    # must never be mistaken for charged parser requests.
                    if self.switches.system_id is not SystemID.E0:
                        raise PolicyError(
                            "pre-action parse rejection is registered only for E0"
                        ) from exc
                    request_id = (
                        f"{episode_id}:pre-action-parse-request:{decision_index}"
                    )
                    rejection = PreActionParseRejection(
                        request_id=request_id,
                        observation_id=pre_policy.observation_id,
                        policy_id=self.system_policy.adapter.policy_id,
                        policy_version=self.system_policy.adapter.policy_version,
                        decision_index=decision_index,
                        error_kind=exc.error_kind,
                        error_sha256=exc.error_sha256,
                    )
                    execution = self.executor.reject_pre_action_parse_request(
                        request_id=request_id,
                    )
                    contracts.parse_rejections.append(rejection)
                    contracts.executions.append(execution)
                    counters.normal_actions += 1
                    self._log(
                        "actions",
                        "pre_action_parse_rejection",
                        {
                            "rejection": rejection.to_dict(),
                            "decision": None,
                            "parameters": None,
                            "action": None,
                            "action_sha256": None,
                            "execution": execution.to_dict(),
                        },
                    )
                    terminal_reason = TerminalReason.POLICY_ERROR
                    break
                action_id = f"{episode_id}:normal-action:{decision_index}"
                parameter_error: ParameterResolutionError | None = None
                resolution_attempts: tuple[Mapping[str, Any], ...] = ()
                parameter_resolution: Mapping[str, Any] | None = None
                try:
                    parameter_rng = self._stage_random(
                        rng_factory,
                        task_id=task.task_id,
                        repeat_id=repeat_id,
                        matched_seed=model_seed,
                        stage=RuntimeStage.ACTION_PARAMETERS,
                        decision_index=decision_index,
                    )
                    parameters = self.executor.run_blocking(
                        "action-parameter provider inference",
                        lambda: self._resolve_parameters_isolated(
                            callback_task,
                            pre_policy,
                            decision,
                            rng=parameter_rng,
                        ),
                    )
                    if parameters.resolution_trace is not None:
                        parameter_resolution = parameters.resolution_trace.to_dict()
                        resolution_attempts = tuple(
                            item.to_dict()
                            for item in parameters.resolution_trace.attempts
                        )
                    action = concretize(
                        action_id=action_id,
                        decision=decision,
                        parameters=parameters,
                    )
                    try:
                        execution = self.executor.execute(action)
                    except BaseException as exc:
                        preserved = getattr(exc, "execution_result", None)
                        is_infrastructure = (
                            getattr(exc, "infrastructure_invalid", False) is True
                        )
                        if type(preserved) is ExecutionResult:
                            execution = preserved
                        elif is_infrastructure:
                            execution = ExecutionResult(
                                action_id=action.action_id,
                                status=ExecutionStatus.ERROR,
                                executor_step=self.executor.steps_used,
                                state_changed=False,
                                environment_error=True,
                                error_kind="infrastructure_interruption",
                                message="classified infrastructure interruption",
                            )
                        else:
                            raise
                        # The adapter fault occurred after Executor consumed this
                        # browser request.  Preserve the exact charged request in
                        # the append-only action stream before the whole-block
                        # infrastructure invalidation propagates.
                        counters.normal_actions += 1
                        self._log(
                            "actions",
                            "normal_action",
                            {
                                "decision": decision.to_dict(),
                                "parameters": parameters.to_dict(),
                                "parameter_error": None,
                                "parameter_resolution": parameter_resolution,
                                "resolution_attempts": list(resolution_attempts),
                                "action": action.to_dict(),
                                "action_sha256": action.record_sha256,
                                "execution": execution.to_dict(),
                                "interrupted": True,
                            },
                        )
                        raise
                except HybridParameterResolutionError as exc:
                    parameter_error = exc
                    parameter_resolution = exc.trace.to_dict()
                    resolution_attempts = tuple(item.to_dict() for item in exc.attempts)
                    action = ConcreteAction(
                        action_id=action_id,
                        source_decision_id=decision.decision_id,
                        action_type=decision.action_type,
                        parameters={},
                        bbox=decision.bbox,
                    )
                    execution = self.executor.reject_unresolved_request(
                        action_id=action_id,
                        reason=str(exc),
                    )
                    parameters = None
                except ParameterResolutionError as exc:
                    parameter_error = exc
                    action = ConcreteAction(
                        action_id=action_id,
                        source_decision_id=decision.decision_id,
                        action_type=decision.action_type,
                        parameters={},
                        bbox=decision.bbox,
                    )
                    execution = self.executor.reject_unresolved_request(
                        action_id=action_id,
                        reason=str(exc),
                    )
                    parameters = None
                contracts.decisions.append(decision)
                contracts.actions.append(action)
                contracts.executions.append(execution)
                counters.normal_actions += 1
                self._log(
                    "actions",
                    "normal_action",
                    {
                        "decision": decision.to_dict(),
                        "parameters": parameters.to_dict() if parameters else None,
                        "parameter_error": str(parameter_error) if parameter_error else None,
                        "parameter_resolution": parameter_resolution,
                        "resolution_attempts": list(resolution_attempts),
                        "action": action.to_dict(),
                        "action_sha256": action.record_sha256,
                        "execution": execution.to_dict(),
                    },
                )
                post = self.executor.observe_after(action)
                contracts.observations.append(post)
                self._log_observation("post_action_observation", post)
                causal_history.append(
                    completed_causal_history_entry(
                        history_index=len(causal_history) + 1,
                        action=action,
                        execution=execution,
                        post_observation=post,
                    )
                )
                loop_detected = self.loop_guard.record(post, action)

                assessment = None
                transition: TransitionInput | None = None
                if self.switches.post_action_diagnosis:
                    transition = self.observation_builder.transition(
                        task,
                        current,
                        action,
                        execution,
                        post,
                        causal_history=prior_history,
                    )
                    transition_rng = self._stage_random(
                        rng_factory,
                        task_id=task.task_id,
                        repeat_id=repeat_id,
                        matched_seed=model_seed,
                        stage=RuntimeStage.POST_ACTION_ASSESSMENT,
                        decision_index=decision_index,
                    )
                    assessment = self.executor.run_blocking(
                        "post-action policy assessment",
                        lambda: self.system_policy.assess_transition(
                            callback_task,
                            transition,
                            rng=transition_rng,
                        ),
                    )
                    contracts.transition_inputs.append(transition)
                    contracts.transition_assessments.append(assessment)
                    self._log(
                        "transitions",
                        "post_action_assessment",
                        {
                            "input": transition.to_dict(),
                            "assessment": assessment.to_dict(),
                        },
                    )

                # This trigger is frozen before the sealed terminal call.  The
                # latter therefore cannot influence trigger, strategy, memory,
                # provider output, or action parameters.
                trigger = self.decision.recovery_trigger(
                    assessment=assessment,
                    execution=execution,
                    loop_detected=loop_detected,
                )
                receipt_binding = self._receipt_binding(
                    "after_normal_action",
                    observation=post,
                    action=action,
                )
                terminal_signal = self.executor.terminal_signal(receipt_binding)
                self._log_terminal_receipt(receipt_binding, terminal_signal)
                if terminal_signal.terminate:
                    final_signal = terminal_signal
                    terminal_reason = TerminalReason.OPAQUE_VERIFIER_TERMINAL
                    current = post
                    break
                if loop_detected and not self.switches.recovery_controller:
                    terminal_reason = TerminalReason.LOOP
                    current = post
                    break
                if not trigger.triggered:
                    current = post
                    continue

                counters.failure_incidents += 1
                incident_id = f"{episode_id}:incident:{counters.failure_incidents}"
                terminal_reason, current, signal = self._run_recovery_incident(
                    task=task,
                    callback_task=callback_task,
                    repeat_id=repeat_id,
                    model_seed=model_seed,
                    decision_index=decision_index,
                    incident_id=incident_id,
                    trigger=trigger,
                    assessment=assessment,
                    failed_action=action,
                    post_failure=post,
                    post_failure_transition=transition,
                    counters=counters,
                    contracts=contracts,
                    duplicate_cluster_ids=duplicate_cluster_ids,
                    causal_history=causal_history,
                )
                if signal is not None:
                    final_signal = signal
                if terminal_reason is TerminalReason.ENVIRONMENT_FAILURE:
                    environment_failure = True
                    valid_for_primary = False

        except ExecutorBudgetExceeded:
            terminal_reason = TerminalReason.ACTION_BUDGET_EXHAUSTED
        except EpisodeTimeout:
            terminal_reason = TerminalReason.TIMEOUT
        except RecoveryBudgetExceeded:
            terminal_reason = TerminalReason.RECOVERY_BUDGET_EXHAUSTED
        except ModelCallBudgetExceeded:
            terminal_reason = TerminalReason.POLICY_ERROR
        except MemoryBoundaryError:
            # Memory provenance/configuration violations invalidate primary E3;
            # they are not agent failures that may be scored conveniently.
            valid_for_primary = False
            terminal_reason = TerminalReason.POLICY_ERROR
            raise
        except PolicyError:
            terminal_reason = TerminalReason.POLICY_ERROR
        except BaseException as exc:
            if getattr(exc, "infrastructure_invalid", False) is True:
                partial = EpisodeSummary(
                    episode_id=episode_id,
                    protocol_id=self.protocol.protocol_id,
                    system_id=self.switches.system_id,
                    task_id=task.task_id,
                    repeat_id=repeat_id,
                    model_seed=model_seed,
                    valid_for_primary=False,
                    terminal_reason=TerminalReason.ENVIRONMENT_FAILURE,
                    executor_steps=self.executor.steps_used,
                    normal_actions=counters.normal_actions,
                    recovery_actions=counters.recovery_actions,
                    recovery_attempts=(
                        self.recovery_controller.episode_attempts
                        if self.recovery_controller
                        else 0
                    ),
                    failure_incidents=counters.failure_incidents,
                    memory_queries=counters.memory_queries,
                    memory_interventions=counters.memory_interventions,
                    elapsed_seconds=self.executor.elapsed_seconds,
                    model_call_count=model_call_ledger.count,
                    environment_failure=True,
                    event_log_sha256=(
                        canonical_sha256(self.event_logs.file_hashes())
                        if self.event_logs is not None
                        else None
                    ),
                )
                setattr(exc, "partial_episode_summary", partial.to_dict())
            raise
        finally:
            try:
                self.executor.close()
            finally:
                deactivate_model_call_ledger(model_call_token)

        # Deliberately commit the runner-owned causal streams *before* the
        # validation receipt and outer orchestration's episode-final receipt.
        # This removes a receipt/summary hash cycle. Package validation
        # reconstructs this exact prefix from the hash-chained JSONL files.
        log_sha256 = (
            canonical_sha256(self.event_logs.file_hashes())
            if self.event_logs is not None
            else None
        )
        summary = EpisodeSummary(
            episode_id=episode_id,
            protocol_id=self.protocol.protocol_id,
            system_id=self.switches.system_id,
            task_id=task.task_id,
            repeat_id=repeat_id,
            model_seed=model_seed,
            valid_for_primary=valid_for_primary,
            terminal_reason=terminal_reason,
            executor_steps=self.executor.steps_used,
            normal_actions=counters.normal_actions,
            recovery_actions=counters.recovery_actions,
            recovery_attempts=(
                self.recovery_controller.episode_attempts
                if self.recovery_controller
                else 0
            ),
            failure_incidents=counters.failure_incidents,
            memory_queries=counters.memory_queries,
            memory_interventions=counters.memory_interventions,
            elapsed_seconds=self.executor.elapsed_seconds,
            model_call_count=model_call_ledger.count,
            reset_already_success=reset_already_success,
            environment_failure=environment_failure,
            verifier_event_id=final_signal.event_id if final_signal else None,
            verifier_token_sha256=(
                final_signal.token_sha256 if final_signal else None
            ),
            event_log_sha256=log_sha256,
        )
        bundle = contracts.bundle(episode_id=episode_id, summary=summary)
        validation = validate_episode_foreign_keys(bundle)
        receipt = {
            "receipt_version": EPISODE_FOREIGN_KEY_RECEIPT_VERSION,
            "episode_id": episode_id,
            "bundle_record_type": "EpisodeContractBundle",
            "bundle_sha256": bundle.record_sha256,
            "summary_sha256": summary.record_sha256,
            "causal_log_snapshot_sha256": summary.event_log_sha256,
            "validation": validation.to_dict(),
            "validation_sha256": validation.record_sha256,
        }
        self._contract_validation_receipt = receipt
        if not self.defer_contract_validation_receipt:
            self._log(
                "environment_events",
                "episode_contract_validation",
                receipt,
            )
        return summary

    def _run_recovery_incident(
        self,
        *,
        task: TaskSpecification,
        callback_task: RuntimeTaskView,
        repeat_id: int,
        model_seed: int,
        decision_index: int,
        incident_id: str,
        trigger: RecoveryTrigger,
        assessment: Any,
        failed_action: ConcreteAction,
        post_failure: Observation,
        post_failure_transition: TransitionInput | None,
        counters: _Counters,
        contracts: _EpisodeContracts,
        duplicate_cluster_ids: tuple[str, ...],
        causal_history: list[CausalHistoryEntry],
    ) -> tuple[TerminalReason, Observation, OpaqueTerminalSignal | None]:
        assert self.recovery_controller is not None
        rng_factory = self.protocol.rng_factory()
        while True:
            # Check both recovery caps before computing/logging a shadow or
            # performing an E3 query for an attempt that cannot be invoked.
            self.recovery_controller.require_attempt_available(incident_id)
            attempt_index = self.recovery_controller.incident_attempts(incident_id) + 1
            shadow = self.decision.no_memory_shadow(
                trigger=trigger,
                assessment=assessment,
                incident_id=incident_id,
            )
            final_decision = shadow
            if self.switches.system_id is SystemID.E3:
                assert self.memory_adapter is not None
                if not duplicate_cluster_ids or any(
                    not item for item in duplicate_cluster_ids
                ):
                    raise MemoryBoundaryError(
                        "E3 query requires non-empty audited duplicate-cluster evidence"
                    )
                query = MemoryQuery(
                    query_id=f"{incident_id}:memory-query:{attempt_index}",
                    task_id=task.task_id,
                    episode_id=post_failure.episode_id,
                    incident_id=incident_id,
                    post_failure_observation_id=post_failure.observation_id,
                    failed_action_id=failed_action.action_id,
                    diagnosis=trigger.diagnosis,
                    duplicate_cluster_ids=duplicate_cluster_ids,
                    post_failure_observation_sha256=(
                        post_failure.record_sha256
                    ),
                    post_action_input_sha256=(
                        post_failure_transition.record_sha256
                        if post_failure_transition is not None
                        else None
                    ),
                    processor_contract_sha256=(
                        self.training_processor_contract.record_sha256
                        if self.training_processor_contract is not None
                        else (
                            self.observation_builder.processor_contract.record_sha256
                            if self.observation_builder.processor_contract is not None
                            else None
                        )
                    ),
                    checkpoint_sha256=getattr(
                        self.system_policy.adapter,
                        "checkpoint_sha256",
                        None,
                    ),
                )
                embedding_request = None
                if self.memory_adapter.evaluation_mode:
                    if post_failure_transition is None:
                        raise MemoryBoundaryError(
                            "E3 embedding requires the exact post-action TransitionInput"
                        )
                    if (
                        query.post_failure_observation_sha256 is None
                        or query.post_action_input_sha256 is None
                        or query.processor_contract_sha256 is None
                        or query.checkpoint_sha256 is None
                    ):
                        raise MemoryBoundaryError(
                            "E3 embedding request lacks a frozen causal/model binding"
                        )
                    embedding_request = PostFailureEmbeddingRequest(
                        query_id=query.query_id,
                        post_failure_observation_id=query.post_failure_observation_id,
                        failed_action_id=query.failed_action_id,
                        post_action_input=post_failure_transition,
                        post_failure_observation_sha256=(
                            query.post_failure_observation_sha256
                        ),
                        post_action_input_sha256=query.post_action_input_sha256,
                        processor_contract_sha256=query.processor_contract_sha256,
                        checkpoint_sha256=query.checkpoint_sha256,
                    )
                self._log(
                    "memory_queries",
                    "no_memory_shadow_before_retrieval",
                    shadow,
                )
                memory_rng = self._stage_random(
                    rng_factory,
                    task_id=task.task_id,
                    repeat_id=repeat_id,
                    matched_seed=model_seed,
                    stage=RuntimeStage.MEMORY_QUERY,
                    decision_index=decision_index,
                    incident_index=counters.failure_incidents,
                    attempt_index=attempt_index,
                )
                memory_decision = self.executor.run_blocking(
                    "post-failure memory embedding and retrieval",
                    lambda: self.memory_adapter.apply_post_failure(
                        query=query,
                        shadow_decision=shadow,
                        rng=memory_rng,
                        embedding_request=embedding_request,
                    ),
                )
                contracts.memory_queries.append(query)
                contracts.memory_results.append(memory_decision.query_result)
                counters.memory_queries += 1
                if memory_decision.query_result.changed_strategy or (
                    memory_decision.query_result.changed_target_or_parameters
                ):
                    counters.memory_interventions += 1
                final_decision = memory_decision.final_decision
                self._log(
                    "memory_queries",
                    "post_failure_query",
                    {
                        **memory_decision.to_dict(),
                        "query": query.to_dict(),
                        "embedding_request": (
                            embedding_request.to_dict()
                            if embedding_request is not None
                            else None
                        ),
                    },
                )
            recovery_resolution_rng = self._stage_random(
                rng_factory,
                task_id=task.task_id,
                repeat_id=repeat_id,
                matched_seed=model_seed,
                stage=RuntimeStage.RECOVERY_RESOLUTION,
                decision_index=decision_index,
                incident_index=counters.failure_incidents,
                attempt_index=attempt_index,
            )
            recovery_observation = self.observation_builder.pre_action(
                task,
                post_failure,
                causal_history=tuple(causal_history),
            )
            self.executor.require_time_remaining(
                operation="recovery action planning"
            )
            try:
                plan = self.recovery_controller.begin_attempt(
                    final_decision,
                    failed_action,
                    task=callback_task,
                    post_failure_observation=recovery_observation,
                    rng=recovery_resolution_rng,
                )
                contracts.register_recovery_decision(final_decision)
            except Exception as exc:
                try:
                    self.executor.require_time_remaining(
                        operation="recovery action planning"
                    )
                except EpisodeTimeout as timeout:
                    raise timeout from exc
                raise
            try:
                self.executor.require_time_remaining(
                    operation="recovery action planning"
                )
            except EpisodeTimeout as timeout:
                # begin_attempt is the registered invocation point. Preserve
                # one-to-one accounting even though its late result is barred
                # from controlling the browser.
                self._log(
                    "recoveries",
                    "recovery_plan",
                    {
                        "shadow_decision": shadow.to_dict(),
                        "final_decision": final_decision.to_dict(),
                        "plan": plan.to_dict(),
                        "deadline_overrun": True,
                    },
                )
                self._log_interrupted_recovery_attempt(
                    plan,
                    [],
                    timeout,
                    contracts=contracts,
                )
                raise
            self._log(
                "recoveries",
                "recovery_plan",
                {
                    "shadow_decision": shadow.to_dict(),
                    "final_decision": final_decision.to_dict(),
                    "plan": plan.to_dict(),
                },
            )
            if plan.resolution_status == "ABORT":
                attempt = self.recovery_controller.finish_attempt(plan)
                contracts.recovery_attempts.append(attempt)
                self._log("recoveries", "recovery_attempt", attempt)
                return TerminalReason.ABORT, post_failure, None
            if plan.resolution_status == "REJECTED":
                attempt = self.recovery_controller.finish_attempt(plan)
                contracts.recovery_attempts.append(attempt)
                self._log("recoveries", "recovery_attempt", attempt)
                if (
                    self.recovery_controller.incident_attempts(incident_id)
                    >= self.protocol.budgets.max_recovery_attempts_per_incident
                ):
                    return (
                        TerminalReason.RECOVERY_BUDGET_EXHAUSTED,
                        post_failure,
                        None,
                    )
                continue

            recovery_pre = post_failure
            recovery_pre_history = tuple(causal_history)
            recovery_results = []
            executed_recovery_actions: list[ConcreteAction] = []
            recovery_post = post_failure
            recovery_terminal_signal: OpaqueTerminalSignal | None = None
            for recovery_action in plan.actions:
                try:
                    result = self.executor.execute(recovery_action)
                except Exception as exc:
                    preserved = getattr(exc, "execution_result", None)
                    is_infrastructure = (
                        getattr(exc, "infrastructure_invalid", False) is True
                    )
                    if type(preserved) is ExecutionResult or is_infrastructure:
                        result = (
                            preserved
                            if type(preserved) is ExecutionResult
                            else ExecutionResult(
                                action_id=recovery_action.action_id,
                                status=ExecutionStatus.ERROR,
                                executor_step=self.executor.steps_used,
                                state_changed=False,
                                environment_error=True,
                                error_kind="infrastructure_interruption",
                                message="classified infrastructure interruption",
                            )
                        )
                        recovery_results.append(result)
                        executed_recovery_actions.append(recovery_action)
                        contracts.actions.append(recovery_action)
                        contracts.executions.append(result)
                        counters.recovery_actions += 1
                        self._log(
                            "actions",
                            "recovery_action",
                            {
                                "attempt_id": plan.attempt_id,
                                "action": recovery_action.to_dict(),
                                "action_sha256": recovery_action.record_sha256,
                                "execution": result.to_dict(),
                                "interrupted": True,
                            },
                        )
                    self._log_interrupted_recovery_attempt(
                        plan,
                        executed_recovery_actions,
                        exc,
                        contracts=contracts,
                    )
                    raise
                recovery_results.append(result)
                executed_recovery_actions.append(recovery_action)
                contracts.actions.append(recovery_action)
                contracts.executions.append(result)
                counters.recovery_actions += 1
                self._log(
                    "actions",
                    "recovery_action",
                    {
                        "attempt_id": plan.attempt_id,
                        "action": recovery_action.to_dict(),
                        "action_sha256": recovery_action.record_sha256,
                        "execution": result.to_dict(),
                    },
                )
                try:
                    recovery_post = self.executor.observe_after(
                        recovery_action,
                        recovery=True,
                    )
                    contracts.observations.append(recovery_post)
                    self._log_observation(
                        "post_recovery_observation",
                        recovery_post,
                    )
                    causal_history.append(
                        completed_causal_history_entry(
                            history_index=len(causal_history) + 1,
                            action=recovery_action,
                            execution=result,
                            post_observation=recovery_post,
                        )
                    )
                    receipt_binding = self._receipt_binding(
                        "after_recovery_action",
                        observation=recovery_post,
                        action=recovery_action,
                    )
                    recovery_terminal_signal = self.executor.terminal_signal(
                        receipt_binding
                    )
                    self._log_terminal_receipt(
                        receipt_binding,
                        recovery_terminal_signal,
                    )
                except Exception as exc:
                    self._log_interrupted_recovery_attempt(
                        plan,
                        executed_recovery_actions,
                        exc,
                        contracts=contracts,
                    )
                    raise
                if recovery_terminal_signal.terminate:
                    # The whole plan is fixed before its first browser request;
                    # the opaque stop bit may terminate execution but cannot
                    # alter recovery selection or already-bound parameters.
                    break

            if not recovery_results or any(
                result.status is not ExecutionStatus.EXECUTED
                for result in recovery_results
            ):
                attempt = self.recovery_controller.finish_attempt(plan)
                contracts.recovery_attempts.append(attempt)
                self._log("recoveries", "recovery_attempt", attempt)
                if (
                    recovery_terminal_signal is not None
                    and recovery_terminal_signal.terminate
                ):
                    return (
                        TerminalReason.OPAQUE_VERIFIER_TERMINAL,
                        recovery_post,
                        recovery_terminal_signal,
                    )
                if (
                    self.recovery_controller.incident_attempts(incident_id)
                    >= self.protocol.budgets.max_recovery_attempts_per_incident
                ):
                    return (
                        TerminalReason.RECOVERY_BUDGET_EXHAUSTED,
                        recovery_post,
                        None,
                    )
                failed_action = plan.actions[-1] if plan.actions else failed_action
                post_failure = recovery_post
                continue

            try:
                recovery_transition = RecoveryTransitionInput(
                    task_id=task.task_id,
                    incident_id=incident_id,
                    attempt_id=plan.attempt_id,
                    pre_recovery_observation=self.observation_builder.pre_action(
                        task,
                        recovery_pre,
                        causal_history=recovery_pre_history,
                    ),
                    recovery_actions=tuple(executed_recovery_actions),
                    post_recovery_observation=self.observation_builder.post_action(
                        task,
                        recovery_post,
                        executed_recovery_actions[-1],
                        causal_history=tuple(causal_history),
                    ),
                )
                recovery_assessment_rng = self._stage_random(
                    rng_factory,
                    task_id=task.task_id,
                    repeat_id=repeat_id,
                    matched_seed=model_seed,
                    stage=RuntimeStage.RECOVERY_ASSESSMENT,
                    decision_index=decision_index,
                    incident_index=counters.failure_incidents,
                    attempt_index=attempt_index,
                )
                recovery_assessment = self.executor.run_blocking(
                    "recovery-transition policy assessment",
                    lambda: self.system_policy.assess_recovery(
                        callback_task,
                        recovery_transition,
                        rng=recovery_assessment_rng,
                    ),
                )
            except Exception as exc:
                self._log_interrupted_recovery_attempt(
                    plan,
                    executed_recovery_actions,
                    exc,
                    contracts=contracts,
                )
                raise
            contracts.recovery_transitions.append(recovery_transition)
            contracts.recovery_assessments.append(recovery_assessment)
            attempt = self.recovery_controller.finish_attempt(
                plan,
                predicted_assessment_id=recovery_assessment.assessment_id,
            )
            contracts.recovery_attempts.append(attempt)
            self._log(
                "recoveries",
                "recovery_attempt",
                {
                    "attempt": attempt.to_dict(),
                    "transition": recovery_transition.to_dict(),
                    "predicted_assessment": recovery_assessment.to_dict(),
                },
            )
            # Opaque evaluation occurs only after trigger, shadow, memory, plan,
            # and concrete actions are fixed/executed.  Each action receipt was
            # already written immediately after its causal observation.
            if (
                recovery_terminal_signal is not None
                and recovery_terminal_signal.terminate
            ):
                return (
                    TerminalReason.OPAQUE_VERIFIER_TERMINAL,
                    recovery_post,
                    recovery_terminal_signal,
                )
            if (
                recovery_assessment.predicted_failure_resolved
                or recovery_assessment.predicted_progress
            ):
                return TerminalReason.CLOSED, recovery_post, None
            if (
                self.recovery_controller.incident_attempts(incident_id)
                >= self.protocol.budgets.max_recovery_attempts_per_incident
            ):
                return (
                    TerminalReason.RECOVERY_BUDGET_EXHAUSTED,
                    recovery_post,
                    None,
                )
            failed_action = plan.actions[-1]
            post_failure = recovery_post

    def _resolve_parameters_isolated(
        self,
        task: RuntimeTaskView,
        observation: PolicyObservation,
        decision: PreActionDecision,
        *,
        rng: random.Random,
    ) -> ActionParameters:
        """Protect runtime-owned policy output from arbitrary provider code."""

        if type(observation) is not PolicyObservation:
            raise ParameterResolutionError(
                "parameter provider requires a versioned policy observation"
            )
        protected = (task, observation, decision)
        protected_hashes = tuple(item.record_sha256 for item in protected)
        callback_inputs = tuple(detached_record_copy(item) for item in protected)
        callback_hashes = tuple(item.record_sha256 for item in callback_inputs)
        callback_error: BaseException | None = None
        result: object | None = None
        try:
            result = self.provider.resolve(
                callback_inputs[0],  # type: ignore[arg-type]
                callback_inputs[1],  # type: ignore[arg-type]
                callback_inputs[2],  # type: ignore[arg-type]
                rng=rng,
            )
        except BaseException as exc:
            callback_error = exc
        mutated: list[str] = []
        for label, records, hashes in (
            ("runtime", protected, protected_hashes),
            ("callback", callback_inputs, callback_hashes),
        ):
            for index, (item, expected) in enumerate(zip(records, hashes)):
                try:
                    actual = item.record_sha256
                except Exception:
                    actual = "<unhashable>"
                if actual != expected:
                    mutated.append(f"{label}[{index}]")
        if mutated:
            mutation = ParameterResolutionError(
                "parameter provider mutated protected callback input: "
                + ", ".join(mutated)
            )
            if callback_error is not None:
                raise mutation from callback_error
            raise mutation
        if callback_error is not None:
            raise callback_error
        if type(result) is not ActionParameters:
            raise ParameterResolutionError(
                "parameter provider returned an invalid action-parameter contract"
            )
        if result.action_type is not decision.action_type:
            raise ParameterResolutionError(
                "parameter provider changed the selected action type"
            )
        return detached_record_copy(result)

    def _log(
        self,
        stream: str,
        event_type: str,
        payload: Any,
    ) -> None:
        if self.event_logs is not None:
            self.event_logs.append(stream, event_type, payload)

    def _log_observation(
        self,
        event_type: str,
        observation: Observation,
    ) -> None:
        """Log a redaction-safe commitment to the complete causal record."""

        self._log(
            "environment_events",
            event_type,
            {
                **observation.to_dict(),
                # This digest is computed before event-log redaction.  A
                # sensitive page-state value may therefore be redacted from
                # the artifact without reducing the sealed causal commitment
                # to a screenshot-only hash.
                "observation_record_sha256": observation.record_sha256,
            },
        )

    def _log_interrupted_recovery_attempt(
        self,
        plan: Any,
        executed_actions: list[ConcreteAction],
        error: Exception,
        *,
        contracts: _EpisodeContracts,
    ) -> None:
        """Close the attempt ledger before propagating a runtime interruption."""

        assert self.recovery_controller is not None
        executed_action_ids = tuple(action.action_id for action in executed_actions)
        attempt = self.recovery_controller.finish_attempt(
            plan,
            executed_action_ids=executed_action_ids,
            completed=False,
        )
        contracts.recovery_attempts.append(attempt)
        self._log(
            "recoveries",
            "recovery_attempt",
            {
                "attempt": attempt.to_dict(),
                "interrupted": True,
                "interruption_kind": type(error).__name__,
                "executed_action_ids": list(executed_action_ids),
            },
        )

    @staticmethod
    def _receipt_binding(
        receipt_kind: str,
        *,
        observation: Observation,
        action: ConcreteAction | None = None,
    ) -> VerifierReceiptBinding:
        return VerifierReceiptBinding(
            receipt_kind=receipt_kind,
            observation_id=observation.observation_id,
            observation_sha256=observation.record_sha256,
            action_id=action.action_id if action is not None else None,
            action_sha256=action.record_sha256 if action is not None else None,
        )

    def _log_terminal_receipt(
        self,
        binding: VerifierReceiptBinding,
        signal: OpaqueTerminalSignal,
    ) -> None:
        self._log(
            "terminal_signals",
            binding.receipt_kind,
            {
                **signal.to_dict(),
                "receipt_binding": binding.to_dict(),
            },
        )

    def _stage_seed(
        self,
        factory: StageRNGFactory,
        **key_fields: object,
    ) -> int:
        key = factory.key(**key_fields)  # type: ignore[arg-type]
        seed = factory.seed_for_key(key)
        self._log(
            "environment_events",
            "stage_rng_use",
            StageSeedUse(
                protocol_id=factory.protocol_id,
                campaign_seed=factory.campaign_seed,
                key=key,
                seed=seed,
            ),
        )
        return seed

    def _stage_random(
        self,
        factory: StageRNGFactory,
        **key_fields: object,
    ) -> random.Random:
        return random.Random(self._stage_seed(factory, **key_fields))
