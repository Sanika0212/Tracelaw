"""Benchmark 1: planted state bugs in executable agent workflows.

For each workflow (:mod:`tracelaw.bench.workflows`) a lawbook is learned from ``--train``
good runs and checked on fresh good runs and on runs with one planted fault each.

Two scenarios:

1. **Monitoring.** Every run is checked as it finishes. Reported: the false-alarm rate on
   good runs (the target is ``--alpha``), the share of faulty runs flagged, and the share
   flagged *at the right step* (the faulty step, or inside a loop's span).
2. **Debugging.** The run is known to have failed; which step broke it? Reported: the
   share of failing runs where the method's top step is the faulty one.

Methods compared: Tracelaw; its two halves alone (laws only, trajectory only); the
single-run rules common in agent tracing tools (a step repeated three or more times in
a row, a run longer than any good run, an error); blaming the last step; a random step.
A learning curve repeats the benchmark with fewer good runs to learn from.

    tracelaw bench faults --train 300 --test 200
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from tracelaw.bench.workflows import FAULTS, WORKFLOWS, faulty_runs, good_runs
from tracelaw.lawbook import Lawbook
from tracelaw.model import Run

Localizer = Callable[[Run], int | None]


def rules(longest: int) -> Callable[[Run], int | None]:
    """Single-run rules: the first step that trips one, else None (not flagged)."""

    def check(run: Run) -> int | None:
        names = run.names
        for i, step in enumerate(run.steps):
            if step.error is not None:
                return i
            if i >= 2 and names[i] == names[i - 1] == names[i - 2]:
                return i
            if i >= longest:
                return i
        return None

    return check


def ablation(book: Lawbook, part: str) -> Callable[[Run], tuple[float, list[float]]]:
    """Step scores from only the laws or only the trajectory of a lawbook."""

    def score(run: Run) -> tuple[float, list[float]]:
        _, steps, findings = book.score(run)
        law = [0.0] * len(steps)
        for f in findings:
            if f.kind == "law":
                law[f.step] += f.score
        values = law if part == "laws" else [s - x for s, x in zip(steps, law, strict=True)]
        return max(values), values

    return score


def _hit(index: int | None, truth: int, span: int) -> bool:
    return index is not None and truth <= index < truth + span


def evaluate(workflow: str, train_n: int, test_n: int, alpha: float, seed: int) -> dict[str, Any]:
    train = good_runs(workflow, train_n, seed=seed)
    book = Lawbook.learn(train, alpha=alpha, seed=seed)
    good = good_runs(workflow, test_n, seed=seed + 1)
    bad = faulty_runs(workflow, test_n, seed=seed + 2)
    longest = max(len(r) for r in train)
    rule = rules(longest)
    rng = random.Random(seed)

    # thresholds for the ablations: the (1 - alpha) quantile of good held-out scores
    parts = {p: ablation(book, p) for p in ("laws", "trajectory")}
    cut = {}
    for p, part_score in parts.items():
        held = sorted(part_score(r)[0] for r in good_runs(workflow, test_n, seed=seed + 3))
        cut[p] = held[min(len(held) - 1, int((1 - alpha) * len(held)))] + 1e-9

    monitors: dict[str, Localizer] = {
        "Tracelaw": lambda r: book.check(r).first_step,
        "laws only": lambda r: _first(parts["laws"](r)[1], cut["laws"]),
        "trajectory only": lambda r: _first(parts["trajectory"](r)[1], cut["trajectory"]),
        "single-run rules": rule,
    }
    debuggers: dict[str, Localizer] = {
        "Tracelaw": book.localize,
        "laws only": lambda r: _argmax(parts["laws"](r)[1]),
        "trajectory only": lambda r: _argmax(parts["trajectory"](r)[1]),
        "single-run rules": lambda r: rule(r) if rule(r) is not None else len(r) - 1,
        "last step": lambda r: len(r) - 1,
        "random step": lambda r: rng.randrange(len(r)),
    }
    out: dict[str, Any] = {"workflow": workflow, "train": train_n, "faulty": len(bad)}
    monitoring: dict[str, Any] = {}
    for name, fn in monitors.items():
        alarms = [fn(r) is not None for r in good]
        by_fault: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
        for run, truth, fault in bad:
            index = fn(run)
            span = run.meta.get("fault_span", 1)
            by_fault[fault.kind].append((index is not None, _hit(index, truth, span)))
        everything = [x for v in by_fault.values() for x in v]
        monitoring[name] = {
            "false_alarms": sum(alarms) / len(alarms),
            "flagged": sum(f for f, _ in everything) / len(everything),
            "flagged_at_fault": sum(h for _, h in everything) / len(everything),
            "by_fault": {k: sum(h for _, h in v) / len(v) for k, v in sorted(by_fault.items())},
        }
    debugging: dict[str, Any] = {}
    failing = [(r, t) for r, t, _ in bad if r.passed is False]
    for name, fn in debuggers.items():
        hits = [_hit(fn(r), t, r.meta.get("fault_span", 1)) for r, t in failing]
        debugging[name] = sum(hits) / len(hits) if hits else float("nan")
    out.update(
        {
            "monitoring": monitoring,
            "debugging": debugging,
            "failing": len(failing),
            "laws": len(book.laws),
        }
    )
    return out


def _first(scores: Sequence[float], threshold: float) -> int | None:
    return next((i for i, s in enumerate(scores) if s >= threshold), None)


def _argmax(scores: Sequence[float]) -> int:
    return max(range(len(scores)), key=scores.__getitem__)


def learning_curve(sizes: Sequence[int], test_n: int, alpha: float, seed: int) -> dict[str, Any]:
    curve: dict[str, Any] = {}
    for n in sizes:
        rows = [evaluate(w, n, test_n, alpha, seed) for w in WORKFLOWS]
        curve[str(n)] = {
            "flagged_at_fault": sum(r["monitoring"]["Tracelaw"]["flagged_at_fault"] for r in rows)
            / len(rows),
            "false_alarms": sum(r["monitoring"]["Tracelaw"]["false_alarms"] for r in rows)
            / len(rows),
            "debugging": sum(r["debugging"]["Tracelaw"] for r in rows) / len(rows),
        }
        print(f"  {n:4d} good runs: {curve[str(n)]}", flush=True)
    return curve


def markdown(results: list[dict[str, Any]]) -> str:
    methods = list(results[0]["monitoring"])
    lines = [
        "| Workflow | Method | False alarms | Flagged | Flagged at the faulty step | "
        "Debugging: top step is the faulty one |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        for m in methods:
            mon = r["monitoring"][m]
            lines.append(
                f"| {r['workflow']} | {m} | {mon['false_alarms']:.1%} | {mon['flagged']:.1%} | "
                f"{mon['flagged_at_fault']:.1%} | {r['debugging'][m]:.1%} |"
            )
        for m in ("last step", "random step"):
            lines.append(f"| {r['workflow']} | {m} | | | | {r['debugging'][m]:.1%} |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="tracelaw bench faults", description=__doc__)
    parser.add_argument("--train", type=int, default=300)
    parser.add_argument("--test", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--curve", nargs="*", type=int, default=[20, 50, 100, 300])
    parser.add_argument("--out", default="runs/faults")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for workflow in WORKFLOWS:
        r = evaluate(workflow, args.train, args.test, args.alpha, args.seed)
        mon = r["monitoring"]["Tracelaw"]
        print(
            f"{workflow}: false alarms {mon['false_alarms']:.1%}, flagged at the fault "
            f"{mon['flagged_at_fault']:.1%}, debugging {r['debugging']['Tracelaw']:.1%}",
            flush=True,
        )
        results.append(r)
    curve = learning_curve(args.curve, args.test, args.alpha, args.seed) if args.curve else {}
    summary = {"alpha": args.alpha, "faults": list(FAULTS), "results": results, "curve": curve}
    (out / "results.json").write_text(json.dumps(summary, indent=1))
    table = markdown(results)
    (out / "results.md").write_text(table)
    print(table)
    return 0
