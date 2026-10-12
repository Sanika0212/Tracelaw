"""Runs and steps: what Tracelaw learns from and checks.

A *run* is one execution of an agent or workflow: an ordered list of *steps* and an
optional outcome. A step has a name (its type: the node, tool or model call), and any of
four records Tracelaw can learn laws about:

- ``state_before`` and ``state_after``: the agent's state around the step;
- ``inputs``: the step's arguments (a tool call's arguments, a prompt's variables);
- ``outputs``: what the step returned.

Values may be nested dictionaries and lists of JSON-like values. Runs are stored as JSON
Lines, one run per line.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Step:
    name: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    state_before: dict[str, Any] | None = None
    state_after: dict[str, Any] | None = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Step:
        return cls(
            name=str(data["name"]),
            inputs=dict(data.get("inputs") or {}),
            outputs=_as_dict(data.get("outputs")),
            state_before=data.get("state_before"),
            state_after=data.get("state_after"),
            error=data.get("error"),
            meta=dict(data.get("meta") or {}),
        )


@dataclass
class Run:
    id: str
    steps: list[Step]
    passed: bool | None = None  # outcome, when known
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.steps]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Run:
        return cls(
            id=str(data.get("id", "")),
            steps=[Step.from_dict(s) for s in data.get("steps", [])],
            passed=data.get("passed"),
            meta=dict(data.get("meta") or {}),
        )

    @classmethod
    def from_states(
        cls,
        id: str,
        states: list[dict[str, Any]],
        names: list[str],
        *,
        passed: bool | None = None,
    ) -> Run:
        """A run from the state after each step (``states[0]`` is the initial state)."""
        if len(states) != len(names) + 1:
            raise ValueError("give one state before the first step and one after each step")
        steps = [
            Step(name, state_before=states[i], state_after=states[i + 1])
            for i, name in enumerate(names)
        ]
        return cls(id, steps, passed)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    return {"value": value}


def load_runs(path: str | Path) -> list[Run]:
    """Runs from a JSON Lines file (one run per line) or a JSON list."""
    text = Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        return [Run.from_dict(r) for r in json.loads(text)]
    return [Run.from_dict(json.loads(line)) for line in text.splitlines() if line.strip()]


def iter_runs(paths: Iterable[str | Path]) -> Iterator[Run]:
    for path in paths:
        yield from load_runs(path)


def save_runs(runs: Iterable[Run], path: str | Path) -> Path:
    out = Path(path)
    with out.open("w", encoding="utf-8") as f:
        for run in runs:
            f.write(json.dumps(run.to_dict(), default=str) + "\n")
    return out
