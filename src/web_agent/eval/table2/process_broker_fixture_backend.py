"""Deterministic engineering-only backend for isolated-broker tests."""

from __future__ import annotations

from typing import Any, Mapping


class _RawFixturePage:
    def __init__(self) -> None:
        self.url = "https://fixture.invalid/start"
        self.actions = 0
        self.closed = False


class FixtureSealedBackend:
    """Owns the raw fixture page; no runtime message returns this object."""

    def __init__(self) -> None:
        self._raw_page = _RawFixturePage()

    def runtime_observe(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        del payload
        return {
            "observation": {
                "url": self._raw_page.url,
                "action_count": self._raw_page.actions,
            }
        }

    def runtime_execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        action = payload.get("action")
        if not isinstance(action, Mapping):
            raise ValueError("fixture execution requires an action object")
        self._raw_page.actions += 1
        self._raw_page.url = str(action.get("url") or self._raw_page.url)
        if action.get("url") == "https://fixture.invalid/aliased-result":
            return {"execution": {"score": 1}}
        return {
            "execution": {"accepted": True, "action_count": self._raw_page.actions}
        }

    def runtime_terminal(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        del payload
        return {"opaque_terminal": self._raw_page.actions >= 1}

    def runtime_close(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        del payload
        self._raw_page.closed = True
        return {"closed": True}

    def shutdown(self) -> None:
        self._raw_page.closed = True


def create_backend() -> FixtureSealedBackend:
    return FixtureSealedBackend()


def create_failing_backend() -> FixtureSealedBackend:
    raise RuntimeError("fixture startup failure")
