"""Versioned, JSON-serializable contracts for the Table 2 runtime.

The classes in this module are intentionally data-only.  In particular,
``PolicyObservation`` and ``TransitionInput`` contain no verifier object or
oracle result.  That type boundary is the first defence against causal
leakage; runtime components also validate arbitrary metadata recursively.
"""

from __future__ import annotations

from collections.abc import Mapping as ABCMapping, Sequence as ABCSequence
from dataclasses import fields, is_dataclass, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import inspect
import json
import math
from pathlib import Path, PurePosixPath
import struct
from types import UnionType
from typing import (
    Any,
    ClassVar,
    ForwardRef,
    Mapping,
    Sequence,
    Union,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)
from urllib.parse import urlsplit


PROCESS_BROKER_IMPORT_SOURCE_SHA256 = hashlib.sha256(
    Path(__file__).resolve().read_bytes()
).hexdigest()

SCHEMA_VERSION = "table2.runtime.v1"
EPISODE_FOREIGN_KEY_RECEIPT_VERSION = (
    "table2.episode-foreign-key-validation.v1"
)
MEMORY_QUERY_VECTOR_DIMENSION = 768
MAX_EXECUTION_MESSAGE_CHARS = 4096
JsonValue = (
    type(None)
    | bool
    | int
    | float
    | str
    | list["JsonValue"]
    | dict[str, "JsonValue"]
)


def _json_value(value: Any) -> JsonValue:
    """Convert supported runtime values to deterministic JSON-compatible data."""
    if isinstance(value, VersionedRecord):
        return value.to_dict()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {
            item.name: _json_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return the canonical encoding used for hashes and append-only logs."""
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def float32_vector_sha256(values: Sequence[float]) -> str:
    """Hash the exact little-endian float32 bytes used by P4 retrieval."""

    return hashlib.sha256(
        struct.pack(f"<{len(values)}f", *(float(value) for value in values))
    ).hexdigest()


_RECORD_REGISTRY: dict[str, type["VersionedRecord"]] = {}


class VersionedRecord:
    """Mixin for immutable runtime records.

    ``from_dict`` is intentionally conservative: it validates the schema and
    record name, then delegates nested reconstruction to a subclass when one
    is needed.  All contracts expose a complete ``to_dict``/``to_json`` form.
    """

    schema_version: ClassVar[str] = SCHEMA_VERSION

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _RECORD_REGISTRY[cls.__name__] = cls

    def to_dict(self) -> dict[str, JsonValue]:
        if not is_dataclass(self):
            raise TypeError("VersionedRecord subclasses must be dataclasses")
        payload = {
            item.name: _json_value(getattr(self, item.name))
            for item in fields(self)
        }
        return {
            "schema_version": self.schema_version,
            "record_type": type(self).__name__,
            **payload,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=indent,
            separators=None if indent is not None else (",", ":"),
        )

    @property
    def record_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def __getattr__(self, name: str) -> Any:
        # Backward-compatible instance convenience without a class-level
        # ``sha256`` descriptor, so records may legitimately declare a
        # serialized field named ``sha256``.
        if name == "sha256":
            return self.record_sha256
        raise AttributeError(name)

    @classmethod
    def _plain_payload(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        schema = payload.get("schema_version")
        if schema != cls.schema_version:
            raise ValueError(
                f"{cls.__name__} schema mismatch: {schema!r} != {cls.schema_version!r}"
            )
        record_type = payload.get("record_type")
        if record_type != cls.__name__:
            raise ValueError(
                f"record type mismatch: {record_type!r} != {cls.__name__!r}"
            )
        return {
            key: value
            for key, value in payload.items()
            if key not in {"schema_version", "record_type"}
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VersionedRecord":
        """Strictly reconstruct one registered record from ``to_dict`` output."""
        if not isinstance(payload, Mapping):
            raise TypeError(f"{cls.__name__}.from_dict requires a mapping")
        target: type[VersionedRecord]
        if cls is VersionedRecord:
            record_name = payload.get("record_type")
            try:
                target = _RECORD_REGISTRY[str(record_name)]
            except KeyError as exc:
                raise ValueError(f"unregistered record type: {record_name!r}") from exc
        else:
            target = cls
        plain = target._plain_payload(payload)
        expected_fields = {item.name: item for item in fields(target) if item.init}
        unknown = set(plain) - set(expected_fields)
        missing = set(expected_fields) - set(plain)
        if unknown or missing:
            raise ValueError(
                f"{target.__name__} fields mismatch: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        hints = get_type_hints(target)
        decoded = {
            name: _decode_typed(hints.get(name, Any), plain[name], path=name)
            for name in expected_fields
        }
        return target(**decoded)


def registered_record_types() -> tuple[str, ...]:
    return tuple(sorted(_RECORD_REGISTRY))


_RecordT = TypeVar("_RecordT", bound=VersionedRecord)


def detached_record_copy(value: _RecordT) -> _RecordT:
    """Strictly reconstruct a record into a detached object graph.

    ``dataclass(frozen=True)`` protects attribute assignment but does not freeze
    nested dictionaries or lists.  Runtime callbacks therefore receive a
    schema-validated reconstruction rather than an object owned by the episode
    ledger.  The hash equality check also makes decoder drift fail closed.
    """

    if not isinstance(value, VersionedRecord):
        raise TypeError("detached_record_copy requires a VersionedRecord")
    expected_sha256 = value.record_sha256
    restored = type(value).from_dict(value.to_dict())
    if type(restored) is not type(value):  # pragma: no cover - registry invariant
        raise TypeError("strict record reconstruction changed the record type")
    if restored.record_sha256 != expected_sha256:
        raise ValueError("strict record reconstruction changed canonical content")
    return cast(_RecordT, restored)


def _decode_typed(annotation: Any, value: Any, *, path: str) -> Any:
    if isinstance(annotation, ForwardRef):
        if annotation.__forward_arg__ != "JsonValue":
            raise TypeError(f"unsupported forward reference at {path}: {annotation!r}")
        return _decode_json_value(value, path=path)
    if annotation is Any:
        return value
    if annotation is type(None):
        if value is not None:
            raise TypeError(f"{path} must be null")
        return None
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in {Union, UnionType}:
        errors: list[str] = []
        for option in args:
            try:
                return _decode_typed(option, value, path=path)
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
        raise TypeError(f"{path} does not match its union type: {'; '.join(errors)}")
    if origin in {tuple}:
        if not isinstance(value, list):
            raise TypeError(f"{path} must be a JSON array for tuple reconstruction")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(
                _decode_typed(args[0], item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            )
        if len(args) != len(value):
            raise ValueError(f"{path} tuple length mismatch")
        return tuple(
            _decode_typed(expected, item, path=f"{path}[{index}]")
            for index, (expected, item) in enumerate(zip(args, value))
        )
    if origin in {list, ABCSequence, Sequence}:
        if not isinstance(value, list):
            raise TypeError(f"{path} must be a JSON array")
        subtype = args[0] if args else Any
        return [
            _decode_typed(subtype, item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if origin in {dict, ABCMapping, Mapping}:
        if not isinstance(value, Mapping):
            raise TypeError(f"{path} must be a JSON object")
        key_type, value_type = args if len(args) == 2 else (Any, Any)
        return {
            _decode_typed(key_type, key, path=f"{path}.<key>"):
            _decode_typed(value_type, item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if inspect.isclass(annotation) and issubclass(annotation, VersionedRecord):
        if not isinstance(value, Mapping):
            raise TypeError(f"{path} must contain a versioned record object")
        return annotation.from_dict(value)
    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        try:
            return annotation(value)
        except ValueError as exc:
            raise ValueError(f"{path} has unknown {annotation.__name__}: {value!r}") from exc
    if annotation is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{path} must be bool")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{path} must be int")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{path} must be float")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise TypeError(f"{path} must be str")
        return value
    if annotation is Path:
        if not isinstance(value, str):
            raise TypeError(f"{path} must be a path string")
        return Path(value)
    if isinstance(value, annotation):
        return value
    raise TypeError(f"unsupported typed field {path}: {annotation!r}")


def _decode_json_value(value: Any, *, path: str) -> JsonValue:
    """Strictly detach recursive ``JsonValue`` forward references."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite JSON number")
        return value
    if isinstance(value, list):
        return [
            _decode_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{path} JSON object keys must be strings")
        return {
            key: _decode_json_value(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    raise TypeError(f"{path} is not a JSON value")


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_lowercase_sha256(name: str, value: str) -> None:
    _require_text(name, value)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256 digest")


def _require_probability(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _require_nonnegative_finite(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _require_utc_timestamp(name: str, value: str) -> datetime:
    _require_text(name, value)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{name} must include an explicit UTC offset")
    return parsed


def _require_probability_distribution(
    name: str,
    values: Mapping[str, float],
    *,
    selected_label: str,
    expected_labels: Sequence[str] | None = None,
    allowed_labels: Sequence[str] | None = None,
) -> None:
    """Validate one explicit, finite categorical probability distribution.

    Action-head evidence has a fixed registered vocabulary and therefore must
    contain every class.  Post-action diagnostic fixtures may use a narrower
    failure vocabulary, but every reported map must still be non-empty,
    normalized, and contain the selected label.  Sparse one-hot recovery maps
    remain valid for deterministic fixtures while unknown strategies fail.
    """

    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{name} must be a non-empty probability mapping")
    labels: set[str] = set()
    total = 0.0
    for label, probability in values.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"{name} labels must be non-empty strings")
        if label in labels:
            raise ValueError(f"{name} contains a duplicate label: {label!r}")
        labels.add(label)
        _require_probability(f"{name}[{label}]", probability)
        total += float(probability)

    if expected_labels is not None:
        expected = set(expected_labels)
        if labels != expected:
            raise ValueError(
                f"{name} class set differs: "
                f"missing={sorted(expected - labels)}, "
                f"unknown={sorted(labels - expected)}"
            )
    if allowed_labels is not None:
        unknown = labels - set(allowed_labels)
        if unknown:
            raise ValueError(f"{name} has unknown labels: {sorted(unknown)}")
    if selected_label not in labels:
        raise ValueError(f"{name} omits selected label {selected_label!r}")
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{name} probabilities must sum to 1, got {total}")


def _validate_bbox(bbox: tuple[float, float, float, float] | None) -> None:
    if bbox is None:
        return
    if len(bbox) != 4:
        raise ValueError("bbox must contain normalized [x, y, width, height]")
    x, y, width, height = (float(value) for value in bbox)
    if min(x, y, width, height) < 0.0:
        raise ValueError(f"bbox contains a negative value: {bbox}")
    if x > 1.0 or y > 1.0 or x + width > 1.0 or y + height > 1.0:
        raise ValueError(f"bbox falls outside the normalized coordinate plane: {bbox}")


class SystemID(str, Enum):
    E0 = "E0"
    E1 = "E1"
    E2 = "E2"
    E3 = "E3"


class ObservationStage(str, Enum):
    RESET = "reset"
    PRE_ACTION = "pre_action"
    POST_ACTION = "post_action"
    POST_RECOVERY = "post_recovery"


class ActionType(str, Enum):
    CLICK = "CLICK"
    TYPE = "TYPE"
    SELECT = "SELECT"
    SCROLL = "SCROLL"
    NAVIGATE = "NAVIGATE"
    PRESS_KEY = "PRESS_KEY"


class RecoveryStrategy(str, Enum):
    NONE = "NONE"
    RETRY = "RETRY"
    REPLAN = "REPLAN"
    BACKTRACK = "BACKTRACK"
    ALTERNATIVE_TARGET = "ALTERNATIVE_TARGET"
    ABORT = "ABORT"


class ExecutionStatus(str, Enum):
    EXECUTED = "executed"
    REJECTED = "rejected"
    ERROR = "error"


class TerminalReason(str, Enum):
    ABORT = "abort"
    ACTION_BUDGET_EXHAUSTED = "action_budget_exhausted"
    RECOVERY_BUDGET_EXHAUSTED = "recovery_budget_exhausted"
    TIMEOUT = "timeout"
    LOOP = "loop"
    ENVIRONMENT_FAILURE = "environment_failure"
    RESET_ALREADY_SUCCESS = "reset_already_success"
    OPAQUE_VERIFIER_TERMINAL = "opaque_verifier_terminal"
    POLICY_ERROR = "policy_error"
    PROVIDER_ERROR = "provider_error"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class RuntimeGeolocation(VersionedRecord):
    """Immutable, observable browser geolocation used at environment reset."""

    latitude: int | float
    longitude: int | float
    accuracy: int | float | None = None

    def __post_init__(self) -> None:
        for name, lower, upper in (
            ("latitude", -90.0, 90.0),
            ("longitude", -180.0, 180.0),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not lower <= float(value) <= upper
            ):
                raise ValueError(f"runtime geolocation {name} is invalid")
        if self.accuracy is not None and (
            isinstance(self.accuracy, bool)
            or not isinstance(self.accuracy, (int, float))
            or not math.isfinite(float(self.accuracy))
            or float(self.accuracy) < 0.0
        ):
            raise ValueError("runtime geolocation accuracy must be finite and nonnegative")

    def to_webarena_mapping(self) -> dict[str, int | float]:
        value: dict[str, int | float] = {
            "latitude": self.latitude,
            "longitude": self.longitude,
        }
        if self.accuracy is not None:
            value["accuracy"] = self.accuracy
        return value


@dataclass(frozen=True, slots=True)
class RuntimeStartState(VersionedRecord):
    """Exact, oracle-blind WebArena reset inputs exposed to the environment.

    ``storage_state`` is deliberately only the frozen relative reference used
    by WebArena.  Cookie/header/local-storage contents are credentials and must
    stay inside the deployment's account-reset capability; they are committed
    separately by the hashes-only reset receipt.
    """

    sites: tuple[str, ...]
    start_url: str
    require_login: bool
    storage_state: str | None
    geolocation: RuntimeGeolocation | None
    require_reset: bool

    def __post_init__(self) -> None:
        if (
            type(self.sites) is not tuple
            or not self.sites
            or any(not isinstance(site, str) or not site.strip() for site in self.sites)
        ):
            raise ValueError("runtime start state requires a nonempty sites tuple")
        if len(set(self.sites)) != len(self.sites):
            raise ValueError("runtime start-state sites must be unique")
        _require_text("runtime start_state.start_url", self.start_url)
        for candidate in self.start_url.split(" |AND| "):
            parsed = urlsplit(candidate)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.username is not None
                or parsed.password is not None
            ):
                raise ValueError(
                    "runtime start URL must contain credential-free HTTP(S) locations"
                )
        for name in ("require_login", "require_reset"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"runtime start-state {name} must be an exact boolean")
        if self.storage_state is not None:
            reference = self.storage_state
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError("storage_state must be a nonempty relative reference")
            if any(character in reference for character in ("\\", "\x00", "\r", "\n")):
                raise ValueError("storage_state contains unsafe path characters")
            parsed_reference = PurePosixPath(reference)
            if (
                parsed_reference.is_absolute()
                or ".." in parsed_reference.parts
                or parsed_reference.suffix.casefold() != ".json"
                or urlsplit(reference).scheme
            ):
                raise ValueError(
                    "storage_state must be a relative JSON reference, not credential data"
                )
        if (
            self.geolocation is not None
            and type(self.geolocation) is not RuntimeGeolocation
        ):
            raise TypeError("runtime start-state geolocation has the wrong contract")

    @classmethod
    def from_webarena_mapping(cls, value: Mapping[str, Any]) -> "RuntimeStartState":
        """Strictly project only the six registered public reset fields."""

        if not isinstance(value, Mapping):
            raise TypeError("WebArena runtime start state must be a mapping")
        if any(not isinstance(key, str) for key in value):
            raise TypeError("WebArena runtime start-state keys must be strings")
        expected = {
            "sites",
            "start_url",
            "require_login",
            "storage_state",
            "geolocation",
            "require_reset",
        }
        if set(value) != expected:
            raise ValueError(
                "WebArena runtime start-state fields mismatch: "
                f"missing={sorted(expected - set(value))}, "
                f"unknown={sorted(set(value) - expected)}"
            )
        sites = value["sites"]
        if not isinstance(sites, list):
            raise TypeError("WebArena runtime start-state sites must be an array")
        raw_geolocation = value["geolocation"]
        geolocation: RuntimeGeolocation | None
        if raw_geolocation is None:
            geolocation = None
        else:
            if not isinstance(raw_geolocation, Mapping):
                raise TypeError("WebArena runtime geolocation must be an object or null")
            if any(not isinstance(key, str) for key in raw_geolocation):
                raise TypeError("WebArena runtime geolocation keys must be strings")
            required_geolocation = {"latitude", "longitude"}
            allowed_geolocation = {*required_geolocation, "accuracy"}
            if not required_geolocation <= set(raw_geolocation) or not set(
                raw_geolocation
            ) <= allowed_geolocation:
                raise ValueError("WebArena runtime geolocation fields are invalid")
            if "accuracy" in raw_geolocation and raw_geolocation["accuracy"] is None:
                raise ValueError("WebArena runtime geolocation accuracy cannot be null")
            geolocation = RuntimeGeolocation(
                latitude=raw_geolocation["latitude"],
                longitude=raw_geolocation["longitude"],
                accuracy=raw_geolocation.get("accuracy"),
            )
        return cls(
            sites=tuple(sites),
            start_url=value["start_url"],
            require_login=value["require_login"],
            storage_state=value["storage_state"],
            geolocation=geolocation,
            require_reset=value["require_reset"],
        )

    def to_webarena_mapping(self) -> dict[str, JsonValue]:
        return {
            "sites": list(self.sites),
            "start_url": self.start_url,
            "require_login": self.require_login,
            "storage_state": self.storage_state,
            "geolocation": (
                None
                if self.geolocation is None
                else self.geolocation.to_webarena_mapping()
            ),
            "require_reset": self.require_reset,
        }

    @property
    def start_state_sha256(self) -> str:
        """Canonical identity shared by the task and reset-state receipt."""

        return canonical_sha256(self.to_webarena_mapping())


@dataclass(frozen=True, slots=True)
class TaskSpecification(VersionedRecord):
    task_id: str
    goal: str
    benchmark_id: str
    benchmark_version: str
    start_state_id: str
    site: str = ""
    start_url: str | None = None
    development_partition: bool = True
    destructive_actions_allowed: bool = False
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    runtime_start_state: RuntimeStartState | None = None

    def __post_init__(self) -> None:
        for name in ("task_id", "goal", "benchmark_id", "benchmark_version", "start_state_id"):
            _require_text(name, getattr(self, name))
        if self.runtime_start_state is not None:
            if type(self.runtime_start_state) is not RuntimeStartState:
                raise TypeError("task runtime_start_state has the wrong contract")
            if self.start_state_id != self.runtime_start_state.start_state_sha256:
                raise ValueError(
                    "task start_state_id differs from the typed runtime start state"
                )
            if self.site != self.runtime_start_state.sites[0]:
                raise ValueError("task site differs from the typed runtime start state")
            if self.start_url != self.runtime_start_state.start_url:
                raise ValueError(
                    "task start_url differs from the typed runtime start state"
                )


@dataclass(frozen=True, slots=True)
class RuntimeTaskView(VersionedRecord):
    """Minimal task identity exposed to model and decision callbacks.

    ``TaskSpecification`` intentionally remains the environment/reset contract:
    benchmark adapters need its start-state and operational fields.  Model,
    parameter-provider, and recovery-planner callbacks need only the task ID and
    user-visible goal.  Keeping this as a separate slotted record makes task
    metadata (including nested evaluator/oracle material) structurally
    unavailable at those callback boundaries rather than relying on callers to
    remember not to inspect it.
    """

    task_id: str
    goal: str

    def __post_init__(self) -> None:
        _require_text("task_id", self.task_id)
        _require_text("goal", self.goal)


def runtime_task_view(
    task: TaskSpecification | RuntimeTaskView,
) -> RuntimeTaskView:
    """Return an exact metadata-free task projection for runtime callbacks."""

    if type(task) is RuntimeTaskView:
        return task
    if type(task) is not TaskSpecification:
        raise TypeError(
            "runtime callback task must be TaskSpecification or RuntimeTaskView"
        )
    return RuntimeTaskView(task_id=task.task_id, goal=task.goal)


@dataclass(frozen=True, slots=True)
class Observation(VersionedRecord):
    observation_id: str
    episode_id: str
    stage: ObservationStage
    screenshot_sha256: str
    screenshot_path: str | None = None
    width: int = 1280
    height: int = 720
    url: str = "about:blank"
    title: str = ""
    page_state: Mapping[str, JsonValue] = field(default_factory=dict)
    page_settled: bool = True
    environment_error: bool = False
    prior_action_id: str | None = None

    def __post_init__(self) -> None:
        _require_text("observation_id", self.observation_id)
        _require_text("episode_id", self.episode_id)
        _require_text("screenshot_sha256", self.screenshot_sha256)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("observation dimensions must be positive")
        if self.stage in {ObservationStage.POST_ACTION, ObservationStage.POST_RECOVERY}:
            _require_text("prior_action_id", self.prior_action_id or "")


@dataclass(frozen=True, slots=True)
class CausalHistoryEntry(VersionedRecord):
    """One completed, oracle-blind browser request visible to later policy calls.

    The entry intentionally contains no action parameters, typed text, browser
    message, page-state value, task metadata, or verifier receipt.  Hashes bind
    those runtime-owned records without copying their potentially private raw
    content into a model input.
    """

    schema_version: ClassVar[str] = "table2.causal-history.v1"

    history_index: int
    action_id: str
    action_type: ActionType
    action_target_fingerprint: str
    execution_status: ExecutionStatus
    executor_step: int
    state_changed: bool
    environment_error: bool
    execution_error_sha256: str | None
    post_observation_id: str
    post_observation_sha256: str
    post_screenshot_sha256: str
    recovery_attempt_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.history_index) is not int or self.history_index <= 0:
            raise ValueError("causal-history index must be a positive integer")
        if type(self.executor_step) is not int or self.executor_step <= 0:
            raise ValueError("causal-history executor step must be positive")
        for name in ("action_id", "post_observation_id"):
            _require_text(name, getattr(self, name))
        if type(self.action_type) is not ActionType:
            raise ValueError("causal-history action type is not registered")
        if type(self.execution_status) is not ExecutionStatus:
            raise ValueError("causal-history execution status is not registered")
        for name in (
            "action_target_fingerprint",
            "post_observation_sha256",
            "post_screenshot_sha256",
        ):
            _require_lowercase_sha256(name, getattr(self, name))
        if type(self.state_changed) is not bool:
            raise ValueError("causal-history state_changed must be an exact boolean")
        if type(self.environment_error) is not bool:
            raise ValueError(
                "causal-history environment_error must be an exact boolean"
            )
        if self.execution_error_sha256 is not None:
            _require_lowercase_sha256(
                "execution_error_sha256",
                self.execution_error_sha256,
            )
        if self.recovery_attempt_id is not None:
            _require_text("recovery_attempt_id", self.recovery_attempt_id)


def completed_causal_history_entry(
    *,
    history_index: int,
    action: ConcreteAction,
    execution: ExecutionResult,
    post_observation: Observation,
) -> CausalHistoryEntry:
    """Build the minimal history projection after a post-state is available."""

    if execution.action_id != action.action_id:
        raise ValueError("causal-history execution belongs to another action")
    if post_observation.prior_action_id != action.action_id:
        raise ValueError("causal-history post-observation cites another action")
    if post_observation.stage not in {
        ObservationStage.POST_ACTION,
        ObservationStage.POST_RECOVERY,
    }:
        raise ValueError("causal-history requires a completed post-action state")
    return CausalHistoryEntry(
        history_index=history_index,
        action_id=action.action_id,
        action_type=action.action_type,
        action_target_fingerprint=action.fingerprint,
        execution_status=execution.status,
        executor_step=execution.executor_step,
        state_changed=execution.state_changed,
        environment_error=execution.environment_error,
        execution_error_sha256=(
            canonical_sha256(
                {
                    "error_kind": execution.error_kind,
                    "message": execution.message,
                }
            )
            if (
                execution.status is not ExecutionStatus.EXECUTED
                or execution.environment_error
                or execution.error_kind is not None
            )
            else None
        ),
        post_observation_id=post_observation.observation_id,
        post_observation_sha256=post_observation.record_sha256,
        post_screenshot_sha256=post_observation.screenshot_sha256,
        recovery_attempt_id=action.recovery_attempt_id,
    )


@dataclass(frozen=True, slots=True)
class PolicyObservation(VersionedRecord):
    """Oracle-blind observation visible to a policy or parameter provider."""

    task_id: str
    goal: str
    observation_id: str
    screenshot_sha256: str
    screenshot_path: str | None
    width: int
    height: int
    url: str
    title: str
    current_page_state: Mapping[str, JsonValue] = field(default_factory=dict)
    causal_history: tuple[CausalHistoryEntry, ...] = ()

    def __post_init__(self) -> None:
        for name in ("task_id", "goal", "observation_id"):
            _require_text(name, getattr(self, name))
        _require_lowercase_sha256("screenshot_sha256", self.screenshot_sha256)
        if type(self.width) is not int or type(self.height) is not int:
            raise ValueError("policy observation dimensions must be exact integers")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("policy observation dimensions must be positive")
        if not isinstance(self.current_page_state, Mapping):
            raise ValueError("policy current_page_state must be a mapping")
        if type(self.causal_history) is not tuple:
            raise ValueError("policy causal_history must be an immutable tuple")
        if any(type(item) is not CausalHistoryEntry for item in self.causal_history):
            raise ValueError(
                "policy causal_history must contain only CausalHistoryEntry records"
            )
        expected_indices = tuple(range(1, len(self.causal_history) + 1))
        if tuple(item.history_index for item in self.causal_history) != expected_indices:
            raise ValueError("policy causal-history indices are not contiguous")
        executor_steps = tuple(item.executor_step for item in self.causal_history)
        if executor_steps != tuple(sorted(set(executor_steps))):
            raise ValueError("policy causal-history executor steps are not increasing")
        action_ids = tuple(item.action_id for item in self.causal_history)
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("policy causal_history contains a duplicate action")
        if self.causal_history and (
            self.causal_history[-1].post_observation_id != self.observation_id
        ):
            raise ValueError(
                "policy causal_history does not end at the current observation"
            )


@dataclass(frozen=True, slots=True)
class PreActionDecision(VersionedRecord):
    decision_id: str
    observation_id: str
    action_type: ActionType
    action_probabilities: Mapping[str, float]
    bbox: tuple[float, float, float, float] | None
    grounding_confidence: float
    confidence_before: float
    input_observation_ids: tuple[str, ...]
    policy_id: str
    policy_version: str
    parameter_hints: Mapping[str, JsonValue] = field(default_factory=dict)
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        _require_text("decision_id", self.decision_id)
        _require_text("observation_id", self.observation_id)
        _require_text("policy_id", self.policy_id)
        _require_text("policy_version", self.policy_version)
        _validate_bbox(self.bbox)
        _require_probability("grounding_confidence", self.grounding_confidence)
        _require_probability("confidence_before", self.confidence_before)
        if tuple(self.input_observation_ids) != (self.observation_id,):
            raise ValueError(
                "pre-action decisions must cite exactly the current observation"
            )
        if type(self.action_type) is not ActionType:
            raise ValueError("action_type must be a registered ActionType")
        _require_probability_distribution(
            "action_probabilities",
            self.action_probabilities,
            selected_label=self.action_type.value,
            expected_labels=tuple(item.value for item in ActionType),
        )


@dataclass(frozen=True, slots=True)
class ParameterResolutionAttempt(VersionedRecord):
    """Measured outcome of one registered provider stage.

    The latency is evidence only.  It is measured around the already-frozen
    stage call and therefore cannot affect provider inputs, RNG state, fallback
    eligibility, or the selected action.
    """

    source: str
    status: str
    latency_ms: float
    reason: str = ""

    def __post_init__(self) -> None:
        _require_text("parameter-resolution source", self.source)
        if self.status not in {"RESOLVED", "REJECTED"}:
            raise ValueError("parameter-resolution status is not registered")
        _require_nonnegative_finite(
            "parameter-resolution attempt latency_ms", self.latency_ms
        )
        if not isinstance(self.reason, str):
            raise TypeError("parameter-resolution reason must be a string")
        if self.status == "RESOLVED" and self.reason:
            raise ValueError("resolved parameter attempt cannot carry a rejection reason")
        if self.status == "REJECTED" and not self.reason.strip():
            raise ValueError("rejected parameter attempt requires a reason")


@dataclass(frozen=True, slots=True)
class ParameterResolutionTrace(VersionedRecord):
    """Versioned timing/provenance receipt for one complete provider request."""

    provider_id: str
    provider_version: str
    attempts: tuple[ParameterResolutionAttempt, ...]
    resolved: bool
    resolution_source: str | None
    latency_ms: float
    trace_version: str = "table2.parameter-resolution-trace.v1"

    def __post_init__(self) -> None:
        _require_text("parameter-resolution provider_id", self.provider_id)
        _require_text("parameter-resolution provider_version", self.provider_version)
        if self.trace_version != "table2.parameter-resolution-trace.v1":
            raise ValueError("parameter-resolution trace version is not registered")
        if type(self.attempts) is not tuple or not self.attempts:
            raise ValueError("parameter-resolution trace requires measured attempts")
        if any(type(item) is not ParameterResolutionAttempt for item in self.attempts):
            raise TypeError("parameter-resolution trace contains an invalid attempt")
        sources = tuple(item.source for item in self.attempts)
        if len(sources) != len(set(sources)):
            raise ValueError("parameter-resolution trace repeats a provider stage")
        total = _require_nonnegative_finite(
            "parameter-resolution total latency_ms", self.latency_ms
        )
        measured = sum(float(item.latency_ms) for item in self.attempts)
        if total + 1e-9 < measured:
            raise ValueError(
                "parameter-resolution total latency is below measured stage latency"
            )
        if type(self.resolved) is not bool:
            raise TypeError("parameter-resolution resolved must be an exact boolean")
        resolved_attempts = tuple(
            item for item in self.attempts if item.status == "RESOLVED"
        )
        if self.resolved:
            if (
                len(resolved_attempts) != 1
                or resolved_attempts[0] is not self.attempts[-1]
                or self.resolution_source != resolved_attempts[0].source
            ):
                raise ValueError(
                    "resolved parameter trace contradicts its terminal attempt"
                )
        elif resolved_attempts or self.resolution_source is not None:
            raise ValueError(
                "rejected parameter trace cannot name a resolved provider stage"
            )


@dataclass(frozen=True, slots=True)
class ActionParameters(VersionedRecord):
    provider_id: str
    provider_version: str
    action_type: ActionType
    values: Mapping[str, JsonValue]
    prompt_sha256: str
    resolution_source: str = "unspecified"
    attempted_sources: tuple[str, ...] = ()
    decoding_parameters: Mapping[str, JsonValue] = field(default_factory=dict)
    latency_ms: float = 0.0
    resolution_trace: ParameterResolutionTrace | None = None

    def __post_init__(self) -> None:
        _require_text("provider_id", self.provider_id)
        _require_text("provider_version", self.provider_version)
        _require_lowercase_sha256("prompt_sha256", self.prompt_sha256)
        _require_text("resolution_source", self.resolution_source)
        _require_nonnegative_finite("action-parameter latency_ms", self.latency_ms)
        if type(self.attempted_sources) is not tuple or any(
            type(source) is not str or not source.strip()
            for source in self.attempted_sources
        ):
            raise ValueError("attempted_sources must contain non-empty stage names")
        if self.resolution_trace is not None:
            trace = self.resolution_trace
            if type(trace) is not ParameterResolutionTrace:
                raise TypeError("resolution_trace has the wrong typed contract")
            if (
                trace.provider_id != self.provider_id
                or trace.provider_version != self.provider_version
                or trace.resolved is not True
                or trace.resolution_source != self.resolution_source
                or tuple(item.source for item in trace.attempts)
                != self.attempted_sources
                or float(trace.latency_ms) != float(self.latency_ms)
            ):
                raise ValueError(
                    "action parameters contradict their resolution trace"
                )


@dataclass(frozen=True, slots=True)
class ConcreteAction(VersionedRecord):
    action_id: str
    source_decision_id: str
    action_type: ActionType
    parameters: Mapping[str, JsonValue]
    bbox: tuple[float, float, float, float] | None = None
    recovery_attempt_id: str | None = None
    destructive: bool = False

    def __post_init__(self) -> None:
        _require_text("action_id", self.action_id)
        _require_text("source_decision_id", self.source_decision_id)
        _validate_bbox(self.bbox)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(
            {
                "action_type": self.action_type.value,
                "parameters": self.parameters,
                "bbox": self.bbox,
            }
        )


@dataclass(frozen=True, slots=True)
class ControllerCommandCommitment(VersionedRecord):
    """Privacy-safe commitment to the exact native controller command."""

    action_id: str
    adapter_id: str
    adapter_version: str
    command_type: str
    command_sha256: str
    canonical_byte_length: int
    encoding: str = "canonical-json-utf8-v1"

    def __post_init__(self) -> None:
        for name in ("action_id", "adapter_id", "adapter_version", "command_type"):
            _require_text(name, getattr(self, name))
        _require_lowercase_sha256("command_sha256", self.command_sha256)
        if type(self.canonical_byte_length) is not int or self.canonical_byte_length < 1:
            raise ValueError("controller command canonical byte length must be positive")
        if self.encoding != "canonical-json-utf8-v1":
            raise ValueError("controller command encoding is not registered")


@dataclass(frozen=True, slots=True)
class ExecutionSafetyReceipt(VersionedRecord):
    """Typed copy plus hash binding of a registered adapter safety decision."""

    safety_policy_id: str
    safety_policy_version: str
    action_id: str
    allowed: bool
    reason_code: str
    source_record_type: str
    source_record_sha256: str
    receipt_version: str = "table2.execution-safety-receipt.v1"

    def __post_init__(self) -> None:
        for name in (
            "safety_policy_id",
            "safety_policy_version",
            "action_id",
            "reason_code",
            "source_record_type",
        ):
            _require_text(name, getattr(self, name))
        if type(self.allowed) is not bool:
            raise TypeError("execution safety allowed must be an exact boolean")
        expected_prefix = "ALLOW_" if self.allowed else "DENY_"
        if not self.reason_code.startswith(expected_prefix):
            raise ValueError("execution safety reason contradicts its decision")
        _require_lowercase_sha256(
            "execution safety source_record_sha256", self.source_record_sha256
        )
        if self.receipt_version != "table2.execution-safety-receipt.v1":
            raise ValueError("execution safety receipt version is not registered")


@dataclass(frozen=True, slots=True)
class ExecutionEvidence(VersionedRecord):
    """Versioned timing, safety and native-command evidence for one request."""

    action_id: str
    started_at_utc: str
    ended_at_utc: str
    status: ExecutionStatus
    controller_command: ControllerCommandCommitment | None = None
    safety_receipt: ExecutionSafetyReceipt | None = None
    evidence_version: str = "table2.execution-evidence.v1"

    def __post_init__(self) -> None:
        _require_text("execution evidence action_id", self.action_id)
        if type(self.status) is not ExecutionStatus:
            raise TypeError("execution evidence status must use ExecutionStatus")
        started = _require_utc_timestamp(
            "execution evidence started_at_utc", self.started_at_utc
        )
        ended = _require_utc_timestamp(
            "execution evidence ended_at_utc", self.ended_at_utc
        )
        if ended < started:
            raise ValueError("execution evidence ends before it starts")
        if self.evidence_version != "table2.execution-evidence.v1":
            raise ValueError("execution evidence version is not registered")
        if self.controller_command is not None:
            if type(self.controller_command) is not ControllerCommandCommitment:
                raise TypeError("execution evidence controller command has wrong type")
            if self.controller_command.action_id != self.action_id:
                raise ValueError("controller command belongs to another action")
        if self.safety_receipt is not None:
            if type(self.safety_receipt) is not ExecutionSafetyReceipt:
                raise TypeError("execution evidence safety receipt has wrong type")
            if self.safety_receipt.action_id != self.action_id:
                raise ValueError("execution safety receipt belongs to another action")
            if self.safety_receipt.allowed is False and (
                self.status is not ExecutionStatus.REJECTED
                or self.controller_command is not None
            ):
                raise ValueError(
                    "denied execution cannot contain a controller command or non-rejection"
                )


@dataclass(frozen=True, slots=True)
class ExecutionResult(VersionedRecord):
    action_id: str
    status: ExecutionStatus
    executor_step: int
    state_changed: bool
    environment_error: bool = False
    error_kind: str | None = None
    message: str = ""
    internal_retry_count: int = 0
    latency_ms: float = 0.0
    evidence: ExecutionEvidence | None = None

    def __post_init__(self) -> None:
        _require_text("action_id", self.action_id)
        if type(self.message) is not str or len(self.message) > MAX_EXECUTION_MESSAGE_CHARS:
            raise ValueError(
                "execution result message must be text within the registered bound"
            )
        if self.executor_step <= 0:
            raise ValueError("executor_step is one-based and must be positive")
        if self.internal_retry_count < 0:
            raise ValueError("internal_retry_count cannot be negative")
        _require_nonnegative_finite("execution latency_ms", self.latency_ms)
        if self.evidence is not None:
            if type(self.evidence) is not ExecutionEvidence:
                raise TypeError("execution result evidence has the wrong type")
            if (
                self.evidence.action_id != self.action_id
                or self.evidence.status is not self.status
            ):
                raise ValueError("execution result contradicts its evidence")


@dataclass(frozen=True, slots=True)
class PreActionParseRejection(VersionedRecord):
    """One charged, non-executable request rejected at the E0 parser boundary.

    The record deliberately contains neither a decision nor a concrete action.
    ``error_sha256`` commits to the parser failure without copying possibly
    sensitive model output into the runtime artifact.
    """

    request_id: str
    observation_id: str
    policy_id: str
    policy_version: str
    decision_index: int
    error_kind: str
    error_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "observation_id",
            "policy_id",
            "policy_version",
        ):
            _require_text(name, getattr(self, name))
        if type(self.decision_index) is not int or self.decision_index <= 0:
            raise ValueError("decision_index must be an exact positive integer")
        if self.error_kind != "invalid_pre_action_parser_output":
            raise ValueError("unregistered pre-action parse rejection kind")
        _require_lowercase_sha256("error_sha256", self.error_sha256)


@dataclass(frozen=True, slots=True)
class TransitionInput(VersionedRecord):
    task_id: str
    pre_observation: PolicyObservation
    executed_action: ConcreteAction
    execution_result: ExecutionResult
    post_observation: PolicyObservation

    def __post_init__(self) -> None:
        if self.pre_observation.task_id != self.task_id:
            raise ValueError("pre-observation belongs to another task")
        if self.post_observation.task_id != self.task_id:
            raise ValueError("post-observation belongs to another task")
        if self.execution_result.action_id != self.executed_action.action_id:
            raise ValueError("execution result belongs to another action")
        if self.pre_observation.observation_id == self.post_observation.observation_id:
            raise ValueError("transition requires distinct pre/post observations")


@dataclass(frozen=True, slots=True)
class TransitionAssessment(VersionedRecord):
    assessment_id: str
    pre_observation_id: str
    post_observation_id: str
    executed_action_id: str
    predicted_failure: bool
    failure_probability: float
    failure_type: str
    failure_type_probabilities: Mapping[str, float]
    needs_recovery: bool
    needs_recovery_probability: float
    recovery_strategy: RecoveryStrategy
    recovery_probabilities: Mapping[str, float]
    memory_update_prediction: bool = False
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "assessment_id",
            "pre_observation_id",
            "post_observation_id",
            "executed_action_id",
            "failure_type",
        ):
            _require_text(name, getattr(self, name))
        _require_probability("failure_probability", self.failure_probability)
        _require_probability(
            "needs_recovery_probability", self.needs_recovery_probability
        )
        _require_probability_distribution(
            "failure_type_probabilities",
            self.failure_type_probabilities,
            selected_label=self.failure_type,
        )
        if type(self.recovery_strategy) is not RecoveryStrategy:
            raise ValueError("recovery_strategy must be a registered RecoveryStrategy")
        _require_probability_distribution(
            "recovery_probabilities",
            self.recovery_probabilities,
            selected_label=self.recovery_strategy.value,
            allowed_labels=tuple(item.value for item in RecoveryStrategy),
        )


@dataclass(frozen=True, slots=True)
class OpaqueTerminalSignal(VersionedRecord):
    """Only information the sealed evaluator may return to runtime control.

    The token is an opaque digest joined to success/progress truth by the eval
    package after execution.  Runtime cannot infer why termination occurred.
    """

    event_id: str
    token_sha256: str
    terminate: bool

    def __post_init__(self) -> None:
        _require_text("event_id", self.event_id)
        _require_lowercase_sha256("token_sha256", self.token_sha256)
        if type(self.terminate) is not bool:
            raise ValueError("opaque terminal terminate must be an exact boolean")


@dataclass(frozen=True, slots=True)
class VerifierReceiptBinding(VersionedRecord):
    """Causal runtime artifact to which one sealed verifier receipt is bound.

    The binding contains no verifier truth.  It lets offline validation prove
    that the sealed evaluator ran once after reset and once after every
    normal/recovery browser request, in the same order as the runtime logs.
    ``observation_sha256`` commits the complete causal :class:`Observation`
    record (not merely its screenshot); ``action_sha256`` commits the complete
    pre-execution action record without exposing its parameters in the sealed
    stream.
    """

    receipt_kind: str
    observation_id: str
    observation_sha256: str
    action_id: str | None = None
    action_sha256: str | None = None

    def __post_init__(self) -> None:
        allowed = {"after_reset", "after_normal_action", "after_recovery_action"}
        if self.receipt_kind not in allowed:
            raise ValueError(f"unknown verifier receipt kind: {self.receipt_kind!r}")
        _require_text("observation_id", self.observation_id)
        _require_lowercase_sha256("observation_sha256", self.observation_sha256)
        if self.receipt_kind == "after_reset":
            if self.action_id is not None or self.action_sha256 is not None:
                raise ValueError("reset verifier receipt cannot cite an action")
        else:
            _require_text("action_id", self.action_id or "")
            _require_lowercase_sha256("action_sha256", self.action_sha256 or "")


@dataclass(frozen=True, slots=True)
class RecoveryDecision(VersionedRecord):
    decision_id: str
    incident_id: str
    strategy: RecoveryStrategy
    trigger_sources: tuple[str, ...]
    diagnosis: str
    planned_action: ConcreteAction | None = None

    def __post_init__(self) -> None:
        _require_text("decision_id", self.decision_id)
        _require_text("incident_id", self.incident_id)
        _require_text("diagnosis", self.diagnosis)
        allowed = {"policy", "executor", "loop_guard"}
        unknown = set(self.trigger_sources) - allowed
        if unknown:
            raise ValueError(f"oracle/unknown recovery trigger source(s): {sorted(unknown)}")
        if not self.trigger_sources:
            raise ValueError("recovery decision must identify a trigger source")


@dataclass(frozen=True, slots=True)
class RecoveryTransitionInput(VersionedRecord):
    task_id: str
    incident_id: str
    attempt_id: str
    pre_recovery_observation: PolicyObservation
    recovery_actions: tuple[ConcreteAction, ...]
    post_recovery_observation: PolicyObservation

    def __post_init__(self) -> None:
        for name in ("task_id", "incident_id", "attempt_id"):
            _require_text(name, getattr(self, name))
        if not self.recovery_actions:
            raise ValueError("executed recovery assessment requires recovery actions")
        if self.pre_recovery_observation.task_id != self.task_id:
            raise ValueError("pre-recovery observation belongs to another task")
        if self.post_recovery_observation.task_id != self.task_id:
            raise ValueError("post-recovery observation belongs to another task")
        if (
            self.pre_recovery_observation.observation_id
            == self.post_recovery_observation.observation_id
        ):
            raise ValueError("recovery transition needs distinct pre/post observations")


@dataclass(frozen=True, slots=True)
class RecoveryAssessment(VersionedRecord):
    """Model prediction over an executed recovery transition, never oracle truth."""

    assessment_id: str
    incident_id: str
    attempt_id: str
    pre_recovery_observation_id: str
    post_recovery_observation_id: str
    recovery_action_ids: tuple[str, ...]
    predicted_failure_resolved: bool
    predicted_resolution_probability: float
    predicted_progress: bool
    predicted_progress_probability: float
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "assessment_id",
            "incident_id",
            "attempt_id",
            "pre_recovery_observation_id",
            "post_recovery_observation_id",
        ):
            _require_text(name, getattr(self, name))
        if not self.recovery_action_ids:
            raise ValueError("recovery assessment must cite executed actions")
        _require_probability(
            "predicted_resolution_probability",
            self.predicted_resolution_probability,
        )
        _require_probability(
            "predicted_progress_probability",
            self.predicted_progress_probability,
        )


@dataclass(frozen=True, slots=True)
class RecoveryAttempt(VersionedRecord):
    attempt_id: str
    incident_id: str
    strategy: RecoveryStrategy
    incident_attempt_index: int
    episode_attempt_index: int
    action_ids: tuple[str, ...]
    completed: bool
    predicted_assessment_id: str | None = None

    def __post_init__(self) -> None:
        _require_text("attempt_id", self.attempt_id)
        _require_text("incident_id", self.incident_id)
        if self.incident_attempt_index <= 0 or self.episode_attempt_index <= 0:
            raise ValueError("recovery attempt indices are one-based")
        if self.strategy is RecoveryStrategy.ABORT and self.action_ids:
            raise ValueError("ABORT is one recovery attempt with zero browser actions")


@dataclass(frozen=True, slots=True)
class MemoryCandidate(VersionedRecord):
    memory_id: str
    source_split: str
    source_task_id: str
    source_episode_id: str
    duplicate_cluster_id: str
    strategy: RecoveryStrategy
    similarity: float
    memory_update_flag: bool
    verified_recovery_success: bool
    final_task_success: bool
    advice: str = ""

    def __post_init__(self) -> None:
        for name in (
            "memory_id",
            "source_split",
            "source_task_id",
            "source_episode_id",
            "duplicate_cluster_id",
        ):
            _require_text(name, getattr(self, name))
        if not math.isfinite(float(self.similarity)) or not -1.0 <= float(self.similarity) <= 1.0:
            raise ValueError(
                f"cosine memory similarity must be finite in [-1, 1]: {self.similarity}"
            )

    def assert_primary_eligible(self) -> None:
        if self.source_split.lower() != "train":
            raise ValueError(f"memory {self.memory_id} is not train-only")
        if not (
            self.memory_update_flag
            and self.verified_recovery_success
            and self.final_task_success
        ):
            raise ValueError(f"memory {self.memory_id} failed primary eligibility")


@dataclass(frozen=True, slots=True)
class MemoryQuery(VersionedRecord):
    query_id: str
    task_id: str
    episode_id: str
    incident_id: str
    post_failure_observation_id: str
    failed_action_id: str
    diagnosis: str
    duplicate_cluster_ids: tuple[str, ...] = ()
    post_failure_observation_sha256: str | None = None
    post_action_input_sha256: str | None = None
    processor_contract_sha256: str | None = None
    checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "query_id",
            "task_id",
            "episode_id",
            "incident_id",
            "post_failure_observation_id",
            "failed_action_id",
            "diagnosis",
        ):
            _require_text(name, getattr(self, name))
        if not self.duplicate_cluster_ids:
            raise ValueError("memory query requires audited duplicate-cluster IDs")
        if any(not str(item).strip() for item in self.duplicate_cluster_ids):
            raise ValueError("memory query contains an empty duplicate-cluster ID")
        if len(self.duplicate_cluster_ids) != len(set(self.duplicate_cluster_ids)):
            raise ValueError("memory query duplicate-cluster IDs must be unique")
        for name in (
            "post_failure_observation_sha256",
            "post_action_input_sha256",
            "processor_contract_sha256",
            "checkpoint_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_lowercase_sha256(name, value)


@dataclass(frozen=True, slots=True)
class MemoryQueryResult(VersionedRecord):
    query_id: str
    shadow_decision_sha256: str
    candidate_ids: tuple[str, ...]
    scores: tuple[float, ...]
    exclusion_reasons: Mapping[str, str]
    admitted_candidate_id: str | None
    admitted: bool
    changed_strategy: bool
    changed_target_or_parameters: bool
    final_strategy: RecoveryStrategy
    latency_ms: float = 0.0
    query_embedding_sha256: str | None = None
    reader_considered_count: int | None = None
    reader_eligible_count: int | None = None
    reader_id: str | None = None
    store_manifest_sha256: str | None = None
    embedding_binding_sha256: str | None = None
    embedding_request_sha256: str | None = None
    processed_batch_sha256: str | None = None
    normalized_query_embedding: tuple[float, ...] | None = None
    normalized_query_embedding_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_text("query_id", self.query_id)
        _require_lowercase_sha256(
            "shadow_decision_sha256", self.shadow_decision_sha256
        )
        if len(self.candidate_ids) != len(self.scores):
            raise ValueError("candidate IDs and scores must have equal length")
        for score in self.scores:
            if not math.isfinite(float(score)) or not -1.0 <= float(score) <= 1.0:
                raise ValueError(f"cosine memory score must be finite in [-1, 1]: {score}")
        if self.admitted != (self.admitted_candidate_id is not None):
            raise ValueError("memory admission flag and candidate ID disagree")
        if self.admitted_candidate_id is not None and (
            self.admitted_candidate_id not in self.candidate_ids
        ):
            raise ValueError("admitted memory is absent from returned candidates")
        if self.query_embedding_sha256 is not None:
            _require_lowercase_sha256(
                "query_embedding_sha256", self.query_embedding_sha256
            )
        if self.reader_id is not None:
            _require_text("reader_id", self.reader_id)
        if self.store_manifest_sha256 is not None:
            _require_lowercase_sha256(
                "store_manifest_sha256", self.store_manifest_sha256
            )
        if self.embedding_binding_sha256 is not None:
            _require_lowercase_sha256(
                "embedding_binding_sha256", self.embedding_binding_sha256
            )
            if self.reader_id is None:
                raise ValueError("memory store identity requires a reader identity")
        for name in ("embedding_request_sha256", "processed_batch_sha256"):
            value = getattr(self, name)
            if value is not None:
                _require_lowercase_sha256(name, value)
        if (self.embedding_request_sha256 is None) != (
            self.processed_batch_sha256 is None
        ):
            raise ValueError("embedding request/batch hashes must be supplied together")
        if (self.normalized_query_embedding is None) != (
            self.normalized_query_embedding_sha256 is None
        ):
            raise ValueError(
                "normalized query embedding/vector hash must be supplied together"
            )
        if self.normalized_query_embedding is not None:
            values = tuple(float(value) for value in self.normalized_query_embedding)
            object.__setattr__(self, "normalized_query_embedding", values)
            if len(values) != MEMORY_QUERY_VECTOR_DIMENSION:
                raise ValueError(
                    "normalized query embedding must contain exactly "
                    f"{MEMORY_QUERY_VECTOR_DIMENSION} values"
                )
            if not all(math.isfinite(value) for value in values):
                raise ValueError("normalized query embedding contains non-finite values")
            if not math.isclose(
                math.hypot(*values),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-5,
            ):
                raise ValueError("normalized query embedding is not L2-normalized")
            assert self.normalized_query_embedding_sha256 is not None
            _require_lowercase_sha256(
                "normalized_query_embedding_sha256",
                self.normalized_query_embedding_sha256,
            )
            if (
                float32_vector_sha256(values)
                != self.normalized_query_embedding_sha256
            ):
                raise ValueError(
                    "normalized query embedding hash differs from float32 values"
                )
        if (self.reader_considered_count is None) != (
            self.reader_eligible_count is None
        ):
            raise ValueError("memory reader counts must be supplied together")
        if self.reader_considered_count is not None:
            assert self.reader_eligible_count is not None
            if (
                type(self.reader_considered_count) is not int
                or type(self.reader_eligible_count) is not int
            ):
                raise ValueError("memory reader counts must be exact integers")
            if self.reader_considered_count < 0 or self.reader_eligible_count < 0:
                raise ValueError("memory reader counts cannot be negative")
            if self.reader_eligible_count > self.reader_considered_count:
                raise ValueError("eligible memory count exceeds considered count")
            if len(self.candidate_ids) > self.reader_eligible_count:
                raise ValueError(
                    "memory reader returned more candidates than eligible items"
                )


@dataclass(frozen=True, slots=True)
class EpisodeSummary(VersionedRecord):
    episode_id: str
    protocol_id: str
    system_id: SystemID
    task_id: str
    repeat_id: int
    model_seed: int
    valid_for_primary: bool
    terminal_reason: TerminalReason
    executor_steps: int
    normal_actions: int
    recovery_actions: int
    recovery_attempts: int
    failure_incidents: int
    memory_queries: int
    memory_interventions: int
    elapsed_seconds: float
    model_call_count: int = 0
    reset_already_success: bool = False
    environment_failure: bool = False
    verifier_event_id: str | None = None
    verifier_token_sha256: str | None = None
    event_log_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in ("episode_id", "protocol_id", "task_id"):
            _require_text(name, getattr(self, name))
        for name in (
            "repeat_id",
            "executor_steps",
            "normal_actions",
            "recovery_actions",
            "recovery_attempts",
            "failure_incidents",
            "memory_queries",
            "memory_interventions",
            "model_call_count",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.reset_already_success:
            if self.valid_for_primary or self.terminal_reason is not TerminalReason.RESET_ALREADY_SUCCESS:
                raise ValueError("already-successful resets must be primary-invalid")
        if self.terminal_reason is TerminalReason.OPAQUE_VERIFIER_TERMINAL:
            _require_text("verifier_event_id", self.verifier_event_id or "")
            _require_text("verifier_token_sha256", self.verifier_token_sha256 or "")
        if self.executor_steps != self.normal_actions + self.recovery_actions:
            raise ValueError("executor-step accounting does not reconcile")


@dataclass(frozen=True, slots=True)
class EpisodeContractBundle(VersionedRecord):
    """In-memory contract package used for strict foreign-key validation."""

    episode_id: str
    observations: tuple[Observation, ...]
    decisions: tuple[PreActionDecision, ...]
    actions: tuple[ConcreteAction, ...]
    executions: tuple[ExecutionResult, ...]
    transition_inputs: tuple[TransitionInput, ...]
    transition_assessments: tuple[TransitionAssessment, ...]
    recovery_decisions: tuple[RecoveryDecision, ...]
    recovery_attempts: tuple[RecoveryAttempt, ...]
    recovery_transitions: tuple[RecoveryTransitionInput, ...]
    recovery_assessments: tuple[RecoveryAssessment, ...]
    memory_queries: tuple[MemoryQuery, ...]
    memory_results: tuple[MemoryQueryResult, ...]
    parse_rejections: tuple[PreActionParseRejection, ...]
    summary: EpisodeSummary


@dataclass(frozen=True, slots=True)
class EpisodeForeignKeyValidation(VersionedRecord):
    episode_id: str
    status: str
    observations: int
    actions: int
    executions: int
    transition_assessments: int
    recovery_attempts: int
    recovery_assessments: int
    memory_queries: int


def _unique_ids(name: str, values: Sequence[str]) -> set[str]:
    resolved = list(values)
    if any(not value for value in resolved):
        raise ValueError(f"{name} contains an empty ID")
    if len(resolved) != len(set(resolved)):
        raise ValueError(f"{name} contains duplicate IDs")
    return set(resolved)


def _policy_observation_projection_sha256(value: PolicyObservation) -> str:
    """Hash fields copied verbatim from an environment ``Observation``."""

    return canonical_sha256(
        {
            "observation_id": value.observation_id,
            "screenshot_sha256": value.screenshot_sha256,
            "screenshot_path": value.screenshot_path,
            "width": value.width,
            "height": value.height,
            "url": value.url,
            "title": value.title,
            "page_state": value.current_page_state,
        }
    )


def _observation_projection_sha256(value: Observation) -> str:
    """Hash the canonical environment fields visible in ``PolicyObservation``."""

    return canonical_sha256(
        {
            "observation_id": value.observation_id,
            "screenshot_sha256": value.screenshot_sha256,
            "screenshot_path": value.screenshot_path,
            "width": value.width,
            "height": value.height,
            "url": value.url,
            "title": value.title,
            "page_state": value.page_state,
        }
    )


def validate_episode_foreign_keys(
    bundle: EpisodeContractBundle,
) -> EpisodeForeignKeyValidation:
    """Replay IDs, embedded records, ordering, and counts for one episode.

    Embedded contracts are compared by canonical content, not merely by their
    foreign-key IDs.  This prevents a transition from carrying an altered copy
    of an otherwise registered action/observation while retaining the same ID.
    """
    if type(bundle) is not EpisodeContractBundle:
        raise TypeError("foreign-key validation requires EpisodeContractBundle")
    if bundle.summary.episode_id != bundle.episode_id:
        raise ValueError("runtime summary belongs to another episode")
    if any(item.episode_id != bundle.episode_id for item in bundle.observations):
        raise ValueError("observation belongs to another episode")
    if bundle.summary.task_id == "":
        raise ValueError("runtime summary has an empty task ID")
    observation_order = tuple(item.observation_id for item in bundle.observations)
    observation_ids = _unique_ids(
        "observations",
        observation_order,
    )
    observations_by_id = {
        item.observation_id: item for item in bundle.observations
    }
    observation_positions = {
        observation_id: index for index, observation_id in enumerate(observation_order)
    }
    decision_order = tuple(item.decision_id for item in bundle.decisions)
    decision_ids = _unique_ids(
        "pre-action decisions",
        decision_order,
    )
    parse_rejection_order = tuple(
        item.request_id for item in bundle.parse_rejections
    )
    parse_rejection_ids = _unique_ids(
        "pre-action parse rejections",
        parse_rejection_order,
    )
    parse_rejection_indices = tuple(
        item.decision_index for item in bundle.parse_rejections
    )
    if parse_rejection_indices != tuple(sorted(set(parse_rejection_indices))):
        raise ValueError("pre-action parse rejections are not in canonical decision order")
    recovery_decision_order = tuple(
        item.decision_id for item in bundle.recovery_decisions
    )
    recovery_decision_ids = _unique_ids(
        "recovery decisions",
        recovery_decision_order,
    )
    if decision_ids & recovery_decision_ids:
        raise ValueError("normal and recovery decisions share an ID")
    action_order = tuple(item.action_id for item in bundle.actions)
    action_ids = _unique_ids("actions", action_order)
    if action_ids & parse_rejection_ids:
        raise ValueError("actions and pre-action parse rejections share a request ID")
    actions_by_id = {item.action_id: item for item in bundle.actions}
    attempt_order = tuple(item.attempt_id for item in bundle.recovery_attempts)
    attempt_ids = _unique_ids(
        "recovery attempts",
        attempt_order,
    )
    attempts_by_id = {
        item.attempt_id: item for item in bundle.recovery_attempts
    }
    query_order = tuple(item.query_id for item in bundle.memory_queries)
    query_ids = _unique_ids(
        "memory queries",
        query_order,
    )
    _unique_ids(
        "transition assessments",
        [item.assessment_id for item in bundle.transition_assessments],
    )
    _unique_ids(
        "recovery assessments",
        [item.assessment_id for item in bundle.recovery_assessments],
    )
    for decision in bundle.decisions:
        if decision.observation_id not in observation_ids:
            raise ValueError(f"decision references missing observation: {decision.decision_id}")
    for rejection in bundle.parse_rejections:
        if rejection.observation_id not in observation_ids:
            raise ValueError(
                "pre-action parse rejection references a missing observation: "
                f"{rejection.request_id}"
            )
    all_decision_ids = decision_ids | recovery_decision_ids
    for action in bundle.actions:
        if action.source_decision_id not in all_decision_ids:
            raise ValueError(f"action references missing decision: {action.action_id}")
        if action.recovery_attempt_id and action.recovery_attempt_id not in attempt_ids:
            raise ValueError(f"action references missing recovery attempt: {action.action_id}")
    execution_order = tuple(item.action_id for item in bundle.executions)
    execution_ids = _unique_ids(
        "executions",
        execution_order,
    )
    if execution_ids != action_ids | parse_rejection_ids:
        raise ValueError(
            "action/execution cardinality or order differs from the canonical package"
        )
    action_execution_order = tuple(
        request_id for request_id in execution_order if request_id in action_ids
    )
    rejection_execution_order = tuple(
        request_id
        for request_id in execution_order
        if request_id in parse_rejection_ids
    )
    if (
        action_execution_order != action_order
        or rejection_execution_order != parse_rejection_order
    ):
        raise ValueError(
            "action/execution cardinality or order differs from the canonical package"
        )
    expected_executor_steps = tuple(range(1, len(bundle.executions) + 1))
    if tuple(item.executor_step for item in bundle.executions) != expected_executor_steps:
        raise ValueError("executor steps are not a contiguous one-based sequence")
    if len(bundle.executions) != bundle.summary.executor_steps:
        raise ValueError("execution count disagrees with runtime summary")
    executions_by_id = {
        item.action_id: item for item in bundle.executions
    }
    for rejection in bundle.parse_rejections:
        execution = executions_by_id[rejection.request_id]
        if (
            execution.status is not ExecutionStatus.REJECTED
            or execution.error_kind != "pre_action_parse_rejected"
        ):
            raise ValueError(
                "pre-action parse rejection execution receipt is inconsistent"
            )

    # Reconstruct the only causal history that policy inputs may carry from
    # runtime-owned action, execution, and completed post-observation records.
    # No verifier receipt participates in this replay.
    post_observations = tuple(
        item for item in bundle.observations if item.prior_action_id is not None
    )
    post_action_order = tuple(
        cast(str, item.prior_action_id) for item in post_observations
    )
    if post_action_order != action_order:
        raise ValueError(
            "action/post-observation cardinality or order differs from the "
            "canonical package"
        )
    expected_history_by_observation_id: dict[
        str,
        tuple[CausalHistoryEntry, ...],
    ] = {}
    canonical_history: list[CausalHistoryEntry] = []
    for canonical_observation in bundle.observations:
        if canonical_observation.prior_action_id is not None:
            canonical_action = actions_by_id.get(
                canonical_observation.prior_action_id
            )
            if canonical_action is None:
                raise ValueError(
                    "post-observation references a missing canonical action"
                )
            canonical_execution = executions_by_id.get(canonical_action.action_id)
            if canonical_execution is None:
                raise ValueError(
                    "post-observation action has no canonical execution"
                )
            canonical_history.append(
                completed_causal_history_entry(
                    history_index=len(canonical_history) + 1,
                    action=canonical_action,
                    execution=canonical_execution,
                    post_observation=canonical_observation,
                )
            )
        expected_history_by_observation_id[
            canonical_observation.observation_id
        ] = tuple(canonical_history)

    normal_action_source_order = tuple(
        item.source_decision_id
        for item in bundle.actions
        if item.recovery_attempt_id is None
    )
    if normal_action_source_order != decision_order:
        raise ValueError(
            "normal decision/action cardinality or order differs from the canonical package"
        )

    def require_policy_observation(
        value: PolicyObservation,
        *,
        context: str,
    ) -> None:
        if value.task_id != bundle.summary.task_id:
            raise ValueError(f"{context} belongs to another task")
        canonical = observations_by_id.get(value.observation_id)
        if canonical is None:
            raise ValueError(f"{context} is missing")
        if (
            _policy_observation_projection_sha256(value)
            != _observation_projection_sha256(canonical)
        ):
            raise ValueError(
                f"{context} differs from its canonical observation projection"
            )
        expected_history = expected_history_by_observation_id.get(
            value.observation_id
        )
        if expected_history is None:
            raise ValueError(f"{context} has no canonical causal-history prefix")
        if tuple(item.record_sha256 for item in value.causal_history) != tuple(
            item.record_sha256 for item in expected_history
        ):
            raise ValueError(
                f"{context} causal history differs from completed canonical events"
            )

    transition_action_order: list[str] = []
    for transition in bundle.transition_inputs:
        if transition.task_id != bundle.summary.task_id:
            raise ValueError("transition belongs to another task")
        require_policy_observation(
            transition.pre_observation,
            context="transition pre-observation",
        )
        require_policy_observation(
            transition.post_observation,
            context="transition post-observation",
        )
        if (
            observation_positions[transition.pre_observation.observation_id]
            >= observation_positions[transition.post_observation.observation_id]
        ):
            raise ValueError("transition observations are not in causal order")
        canonical_action = actions_by_id.get(transition.executed_action.action_id)
        if canonical_action is None:
            raise ValueError("transition action is missing")
        if transition.executed_action.record_sha256 != canonical_action.record_sha256:
            raise ValueError("transition embedded action differs from canonical action")
        canonical_execution = executions_by_id.get(canonical_action.action_id)
        if canonical_execution is None:
            raise ValueError("transition execution result is missing")
        if (
            transition.execution_result.record_sha256
            != canonical_execution.record_sha256
        ):
            raise ValueError(
                "transition embedded execution differs from canonical execution"
            )
        canonical_post = observations_by_id[transition.post_observation.observation_id]
        if canonical_post.prior_action_id != canonical_action.action_id:
            raise ValueError("transition post-observation cites another action")
        transition_action_order.append(canonical_action.action_id)
    _unique_ids("transition inputs", transition_action_order)
    action_positions = {
        action_id: index for index, action_id in enumerate(action_order)
    }
    if tuple(action_positions[item] for item in transition_action_order) != tuple(
        sorted(action_positions[item] for item in transition_action_order)
    ):
        raise ValueError("transition inputs are not in canonical action order")

    transition_keys = tuple(
        (
            item.pre_observation.observation_id,
            item.post_observation.observation_id,
            item.executed_action.action_id,
        )
        for item in bundle.transition_inputs
    )
    assessment_keys = tuple(
        (
            item.pre_observation_id,
            item.post_observation_id,
            item.executed_action_id,
        )
        for item in bundle.transition_assessments
    )
    if assessment_keys != transition_keys:
        raise ValueError(
            "transition input/assessment cardinality or order differs"
        )
    for assessment in bundle.transition_assessments:
        if assessment.pre_observation_id not in observation_ids:
            raise ValueError("transition assessment pre-observation is missing")
        if assessment.post_observation_id not in observation_ids:
            raise ValueError("transition assessment post-observation is missing")
        if assessment.executed_action_id not in action_ids:
            raise ValueError("transition assessment action is missing")
    incident_attempt_positions: dict[str, int] = {}
    for episode_index, attempt in enumerate(bundle.recovery_attempts, start=1):
        if attempt.episode_attempt_index != episode_index:
            raise ValueError("recovery attempts are not in canonical episode order")
        incident_index = incident_attempt_positions.get(attempt.incident_id, 0) + 1
        incident_attempt_positions[attempt.incident_id] = incident_index
        if attempt.incident_attempt_index != incident_index:
            raise ValueError("recovery attempts are not in canonical incident order")
        canonical_attempt_actions = tuple(
            action.action_id
            for action in bundle.actions
            if action.recovery_attempt_id == attempt.attempt_id
        )
        if tuple(attempt.action_ids) != canonical_attempt_actions:
            raise ValueError(
                f"recovery attempt {attempt.attempt_id} action cardinality/order differs"
            )

    recovery_transition_order: list[str] = []
    for transition in bundle.recovery_transitions:
        attempt = attempts_by_id.get(transition.attempt_id)
        if attempt is None:
            raise ValueError("recovery transition references missing attempt")
        if transition.incident_id != attempt.incident_id:
            raise ValueError("recovery transition incident differs from its attempt")
        require_policy_observation(
            transition.pre_recovery_observation,
            context="recovery transition pre-observation",
        )
        require_policy_observation(
            transition.post_recovery_observation,
            context="recovery transition post-observation",
        )
        if (
            observation_positions[transition.pre_recovery_observation.observation_id]
            >= observation_positions[transition.post_recovery_observation.observation_id]
        ):
            raise ValueError("recovery transition observations are not in causal order")
        embedded_action_ids = tuple(
            item.action_id for item in transition.recovery_actions
        )
        if embedded_action_ids != tuple(attempt.action_ids):
            raise ValueError(
                "recovery transition action cardinality/order differs from its attempt"
            )
        for embedded in transition.recovery_actions:
            canonical_action = actions_by_id.get(embedded.action_id)
            if canonical_action is None:
                raise ValueError("recovery transition action is missing")
            if embedded.record_sha256 != canonical_action.record_sha256:
                raise ValueError(
                    "recovery transition embedded action differs from canonical action"
                )
        canonical_post = observations_by_id[
            transition.post_recovery_observation.observation_id
        ]
        if canonical_post.prior_action_id != embedded_action_ids[-1]:
            raise ValueError("recovery post-observation cites another action")
        recovery_transition_order.append(transition.attempt_id)
    _unique_ids("recovery transitions", recovery_transition_order)
    attempt_positions = {
        attempt_id: index for index, attempt_id in enumerate(attempt_order)
    }
    if tuple(attempt_positions[item] for item in recovery_transition_order) != tuple(
        sorted(attempt_positions[item] for item in recovery_transition_order)
    ):
        raise ValueError("recovery transitions are not in canonical attempt order")

    recovery_assessment_order = tuple(
        item.attempt_id for item in bundle.recovery_assessments
    )
    if recovery_assessment_order != tuple(recovery_transition_order):
        raise ValueError(
            "recovery transition/assessment cardinality or order differs"
        )
    recovery_assessments_by_attempt = {
        item.attempt_id: item for item in bundle.recovery_assessments
    }
    recovery_transitions_by_attempt = {
        item.attempt_id: item for item in bundle.recovery_transitions
    }
    for attempt_id, assessment in recovery_assessments_by_attempt.items():
        transition = recovery_transitions_by_attempt[attempt_id]
        if (
            assessment.incident_id != transition.incident_id
            or assessment.pre_recovery_observation_id
            != transition.pre_recovery_observation.observation_id
            or assessment.post_recovery_observation_id
            != transition.post_recovery_observation.observation_id
            or tuple(assessment.recovery_action_ids)
            != tuple(item.action_id for item in transition.recovery_actions)
        ):
            raise ValueError(
                "recovery assessment differs from its canonical transition"
            )
        attempt = attempts_by_id[attempt_id]
        if attempt.predicted_assessment_id != assessment.assessment_id:
            raise ValueError(
                "recovery attempt cites another predicted assessment"
            )
    for attempt in bundle.recovery_attempts:
        if (
            attempt.predicted_assessment_id is not None
            and attempt.attempt_id not in recovery_assessments_by_attempt
        ):
            raise ValueError("recovery attempt assessment is missing")

    result_order = tuple(item.query_id for item in bundle.memory_results)
    if result_order != query_order:
        raise ValueError("memory query/result cardinality or order differs")
    transitions_by_action_and_post = {
        (
            transition.executed_action.action_id,
            transition.post_observation.observation_id,
        ): transition
        for transition in bundle.transition_inputs
    }
    for query in bundle.memory_queries:
        if query.episode_id != bundle.episode_id:
            raise ValueError("memory query belongs to another episode")
        if query.task_id != bundle.summary.task_id:
            raise ValueError("memory query belongs to another task")
        post_failure = observations_by_id.get(query.post_failure_observation_id)
        if post_failure is None:
            raise ValueError("memory query post-failure observation is missing")
        if query.failed_action_id not in action_ids:
            raise ValueError("memory query failed action is missing")
        if (
            query.post_failure_observation_sha256 is not None
            and query.post_failure_observation_sha256 != post_failure.record_sha256
        ):
            raise ValueError(
                "memory query post-failure observation hash differs from canonical record"
            )
        if query.post_action_input_sha256 is not None:
            transition = transitions_by_action_and_post.get(
                (query.failed_action_id, query.post_failure_observation_id)
            )
            if (
                transition is None
                or query.post_action_input_sha256 != transition.record_sha256
            ):
                raise ValueError(
                    "memory query transition hash differs from canonical record"
                )

    normal_executions = len(bundle.parse_rejections) + sum(
        action.recovery_attempt_id is None
        for action in bundle.actions
        if action.action_id in execution_ids
    )
    recovery_executions = sum(
        action.recovery_attempt_id is not None
        for action in bundle.actions
        if action.action_id in execution_ids
    )
    if normal_executions != bundle.summary.normal_actions:
        raise ValueError("normal-action count disagrees with runtime summary")
    if recovery_executions != bundle.summary.recovery_actions:
        raise ValueError("recovery-action count disagrees with runtime summary")
    if len(bundle.recovery_attempts) != bundle.summary.recovery_attempts:
        raise ValueError("recovery-attempt count disagrees with runtime summary")
    if len(bundle.memory_queries) != bundle.summary.memory_queries:
        raise ValueError("memory-query count disagrees with runtime summary")
    interventions = sum(
        result.changed_strategy or result.changed_target_or_parameters
        for result in bundle.memory_results
    )
    if interventions != bundle.summary.memory_interventions:
        raise ValueError("memory-intervention count disagrees with runtime summary")
    return EpisodeForeignKeyValidation(
        episode_id=bundle.episode_id,
        status="PASS",
        observations=len(bundle.observations),
        actions=len(bundle.actions),
        executions=len(bundle.executions),
        transition_assessments=len(bundle.transition_assessments),
        recovery_attempts=len(bundle.recovery_attempts),
        recovery_assessments=len(bundle.recovery_assessments),
        memory_queries=len(bundle.memory_queries),
    )


def probability_map(labels: Sequence[str], selected: str) -> dict[str, float]:
    """Convenience helper for deterministic fixture policies."""
    if selected not in labels:
        raise ValueError(f"selected label {selected!r} is absent")
    return {label: 1.0 if label == selected else 0.0 for label in labels}
