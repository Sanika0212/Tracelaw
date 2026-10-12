"""Turn traces you already have into runs.

- :func:`from_messages`: a chat transcript with tool calls (OpenAI message format), as
  agent frameworks and benchmarks such as τ-bench log them. Each tool call becomes a
  step with its arguments and parsed result; the agent's and user's turns become
  ``reply`` and ``user`` steps. The state is what the conversation has established so
  far: every id-like value the user mentioned or a tool returned (``ids``), and whether
  the user's last message agreed to something (``user_agreed``), so that laws such as
  "the agent only uses order ids it has seen" and "it acts only after the user agrees"
  can be learned.
- :func:`from_otel`: OpenTelemetry spans (OTLP JSON) that follow the GenAI semantic
  conventions; ``execute_tool`` and ``chat`` spans become steps.
- :func:`from_counterfact`: runs recorded by Counterfact (its SQLite store).
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from tracelaw.model import Run, Step

ID_PATTERN = re.compile(r"#?[A-Z]\d{5,}|\b[a-z]+(?:_[a-z]+)+_\d{2,}\b|\b\d{6,}\b|\b[A-Z0-9]{6}\b")


AGREEMENT = re.compile(
    r"(yes|yeah|yep|sure|correct|confirm|confirmed|that's (right|correct)|please (go ahead|proceed)"
    r"|go ahead|proceed|ok(ay)?\b)",
    re.IGNORECASE,
)


def ids_in(text: str) -> list[str]:
    """Id-like tokens: ``#W2378156``, ``yusuf_rossi_9620``, ``credit_card_9513926``,
    ``1151293680``, ``HAT0EM``."""
    return ID_PATTERN.findall(text)


def _parse(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        text = content.strip()
        if text[:1] in "{[":
            try:
                value = json.loads(text)
            except ValueError:
                pass
            else:
                return value if isinstance(value, dict) else {"value": value}
        return {"value": text if len(text) <= 200 else text[:200]}
    return {"value": content}


def from_messages(
    run_id: str,
    messages: Sequence[dict[str, Any]],
    *,
    passed: bool | None = None,
    meta: dict[str, Any] | None = None,
) -> Run:
    known: list[str] = []
    steps: list[Step] = []
    agreed = [False]  # did the user's last message agree to something?
    results = {m.get("tool_call_id"): m for m in messages if m.get("role") == "tool"}

    def state() -> dict[str, Any]:
        return {
            "ids": sorted(set(known)),
            "calls": sum(s.name not in ("reply", "user") for s in steps),
            "user_agreed": agreed[0],
        }

    for message in messages:
        role = message.get("role")
        content = message.get("content") or ""
        if role == "user":
            before = state()
            known.extend(ids_in(str(content)))
            agreed[0] = bool(AGREEMENT.match(str(content).strip()))
            steps.append(Step("user", {}, {"chars": len(str(content))}, before, state()))
        elif role == "assistant":
            if content:
                steps.append(Step("reply", {}, {"chars": len(str(content))}, state(), state()))
            for call in message.get("tool_calls") or []:
                fn = call.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {"_unparsed": fn.get("arguments")}
                result = results.get(call.get("id"), {})
                output = result.get("content", "")
                before = state()
                error = str(output) if str(output).startswith("Error") else None
                known.extend(ids_in(str(output)))
                steps.append(
                    Step(
                        fn.get("name", "tool"),
                        args if isinstance(args, dict) else {"value": args},
                        _parse(output),
                        before,
                        state(),
                        error,
                    )
                )
    return Run(run_id, steps, passed, dict(meta or {}))


def from_taubench(path: str | Path) -> list[Run]:
    """τ-bench trajectories (a JSON list with ``task_id``, ``trial``, ``reward``, ``traj``)."""
    data = json.loads(Path(path).read_text())
    return [
        from_messages(
            f"task{r['task_id']}-trial{r.get('trial', 0)}",
            r["traj"],
            passed=r.get("reward", 0) >= 1.0,
            meta={"task_id": r["task_id"], "trial": r.get("trial", 0)},
        )
        for r in data
    ]


# ------------------------------------------------------------------- OpenTelemetry


def _attributes(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        value = item.get("value", {})
        out[item["key"]] = next(iter(value.values()), None) if isinstance(value, dict) else value
    return out


def from_otel(path: str | Path) -> list[Run]:
    """OTLP JSON (``resourceSpans``): one run per trace, steps in start order."""
    data = json.loads(Path(path).read_text())
    spans: dict[str, list[dict[str, Any]]] = {}
    for resource in data.get("resourceSpans", []):
        for scope in resource.get("scopeSpans", []):
            for span in scope.get("spans", []):
                spans.setdefault(span["traceId"], []).append(span)
    runs = []
    for trace, items in spans.items():
        steps = []
        for span in sorted(items, key=lambda s: int(s.get("startTimeUnixNano", 0))):
            attrs = _attributes(span.get("attributes", []))
            op = attrs.get("gen_ai.operation.name")
            if op not in ("execute_tool", "chat"):
                continue
            name = (
                attrs.get("gen_ai.tool.name") or attrs.get("gen_ai.request.model") or span["name"]
            )
            inputs = _parse(attrs.get("gen_ai.tool.call.arguments") or {})
            outputs = _parse(attrs.get("gen_ai.tool.call.result") or {})
            if op == "chat":
                outputs = {
                    k.split(".")[-1]: v for k, v in attrs.items() if k.startswith("gen_ai.usage.")
                }
            error = (
                (span.get("status") or {}).get("message")
                if (span.get("status") or {}).get("code") == 2
                else None
            )
            steps.append(Step(str(name), inputs, outputs, None, None, error))
        runs.append(Run(trace, steps))
    return runs


# --------------------------------------------------------------------- Counterfact


def from_counterfact(path: str | Path, *, parent_runs: bool = False) -> list[Run]:
    """Runs from a Counterfact recording (``.cf.sqlite``). Replays and interventions are
    left out unless ``parent_runs`` is true; the outcome (``passed``) is kept."""
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    blobs = _blob_reader(con)
    query = "SELECT run_id, passed FROM runs" + (
        "" if parent_runs else " WHERE parent_run_id IS NULL"
    )
    runs = []
    for row in con.execute(query + " ORDER BY started_ns"):
        steps = []
        for span in con.execute(
            "SELECT name, kind, status, error, input_ref, output_ref FROM spans "
            "WHERE run_id = ? ORDER BY start_ns",
            (row["run_id"],),
        ):
            if span["kind"] not in ("execute_tool", "chat", "step"):
                continue
            steps.append(
                Step(
                    span["name"],
                    _parse(blobs(span["input_ref"])),
                    _parse(blobs(span["output_ref"])),
                    None,
                    None,
                    span["error"] if span["status"] == "error" else None,
                )
            )
        passed = None if row["passed"] is None else bool(row["passed"])
        runs.append(Run(row["run_id"], steps, passed))
    con.close()
    return runs


_CHAIN = "__cf_chain__"  # Counterfact stores long message lists as linked blobs


def _blob_reader(con: sqlite3.Connection) -> Any:
    import zlib

    def raw(ref: str) -> Any:
        found = con.execute("SELECT data FROM blobs WHERE hash = ?", (ref,)).fetchone()
        if found is None:
            return None
        return json.loads(zlib.decompress(found["data"]).decode("utf-8"))

    def join(value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) == {_CHAIN}:
                items, node = [], value[_CHAIN]
                while node is not None:
                    link = raw(node)
                    items.append(join(raw(link["m"])))
                    node = link["p"]
                return items[::-1]
            return {k: join(v) for k, v in value.items()}
        if isinstance(value, list):
            return [join(v) for v in value]
        return value

    def read(ref: str | None) -> Any:
        return join(raw(ref)) if ref else None

    return read
