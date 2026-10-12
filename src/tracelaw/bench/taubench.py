"""Benchmark 2: real agent conversations from τ-bench, passing and failing.

τ-bench (Yao et al., 2024) has a customer-service agent use tools on a retail or an
airline database while a simulated user asks for changes; a run passes when the
database ends in the right state. Its authors publish the trajectories of their runs
(MIT license), downloaded here on first use: gpt-4o on 115 retail and 50 airline tasks,
four trials each.

Each conversation becomes a run (:func:`tracelaw.importers.from_messages`). Tracelaw
learns only from passing runs and must rank failing runs above passing ones it has not
seen, in two settings:

1. **New tasks**: tasks are split into five folds; each fold is scored by a lawbook
   learned from the passing runs of the other tasks (the hard setting: new users,
   orders and requests).
2. **Recurring tasks**: learned from the passing runs of trials 0 and 1, tested on
   trials 2 and 3 of the same tasks (monitoring a deployed agent whose requests recur).

Reported: AUROC of the run score for "this run failed", the share of passing and of
failing runs flagged at ``alpha``, and the same AUROC for the laws alone, the trajectory
alone, the single-run rules, and the run's length.

    tracelaw bench taubench --alpha 0.1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tracelaw.bench.faults import rules
from tracelaw.importers import from_taubench
from tracelaw.lawbook import Lawbook
from tracelaw.model import Run

SOURCE = "https://raw.githubusercontent.com/sierra-research/tau-bench/main/historical_trajectories/"
DOMAINS = ("gpt-4o-retail", "gpt-4o-airline")


def data_dir() -> Path:
    base = Path(os.environ.get("TRACELAW_CACHE", Path.home() / ".cache" / "tracelaw"))
    return base / "taubench"


def load(name: str) -> list[Run]:
    path = data_dir() / f"{name}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOURCE + f"{name}.json", path)
    return from_taubench(path)


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Mann-Whitney AUROC with tied scores given their average rank."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for value in np.unique(s):
        tied = s == value
        ranks[tied] = ranks[tied].mean()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _scores(book: Lawbook, run: Run, longest: int) -> dict[str, float]:
    top, steps, findings = book.score(run)
    laws = sum(f.score for f in findings if f.kind == "law")
    hits = 0
    rule = rules(longest)
    rest = run
    while (i := rule(rest)) is not None and hits < 50:  # count every rule hit
        hits += 1
        rest = Run(rest.id, rest.steps[i + 1 :])
    return {
        "Tracelaw": top,
        "laws only": laws,
        "trajectory only": sum(steps) - laws,
        "single-run rules": float(hits),
        "run length": float(len(run)),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [r["failed"] for r in rows]
    methods = list(rows[0]["scores"])
    flags = np.array([r["flagged"] for r in rows])
    fail = np.array(failed)
    return {
        "runs": len(rows),
        "failing": float(fail.mean()),
        "auroc": {m: auroc([r["scores"][m] for r in rows], failed) for m in methods},
        "flagged_passing": float(flags[~fail].mean()),
        "flagged_failing": float(flags[fail].mean()),
    }


def new_tasks(runs: list[Run], alpha: float, folds: int, seed: int) -> dict[str, Any]:
    tasks = sorted({r.meta["task_id"] for r in runs})
    random.Random(seed).shuffle(tasks)
    rows = []
    for f in range(folds):
        held = set(tasks[f::folds])
        train = [r for r in runs if r.meta["task_id"] not in held and r.passed]
        book = Lawbook.learn(train, alpha=alpha, seed=seed)
        longest = max(len(r) for r in train)
        for run in runs:
            if run.meta["task_id"] in held:
                rows.append(
                    {
                        "failed": not run.passed,
                        "flagged": book.check(run).flagged,
                        "scores": _scores(book, run, longest),
                    }
                )
    return _summarize(rows)


def recurring_tasks(runs: list[Run], alpha: float, seed: int) -> dict[str, Any]:
    train = [r for r in runs if r.meta["trial"] in (0, 1) and r.passed]
    book = Lawbook.learn(train, alpha=alpha, seed=seed)
    longest = max(len(r) for r in train)
    rows = [
        {
            "failed": not r.passed,
            "flagged": book.check(r).flagged,
            "scores": _scores(book, r, longest),
        }
        for r in runs
        if r.meta["trial"] in (2, 3)
    ]
    return _summarize(rows)


def markdown(results: dict[str, Any]) -> str:
    lines = [
        "| Domain | Setting | Runs (failing) | Tracelaw AUROC | Laws only | Trajectory only | "
        "Single-run rules | Run length | Flagged: passing | Flagged: failing |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for domain, by in results.items():
        for setting, r in by.items():
            a = r["auroc"]
            lines.append(
                f"| {domain} | {setting} | {r['runs']} ({r['failing']:.0%}) | "
                f"{a['Tracelaw']:.3f} | {a['laws only']:.3f} | {a['trajectory only']:.3f} | "
                f"{a['single-run rules']:.3f} | {a['run length']:.3f} | "
                f"{r['flagged_passing']:.1%} | {r['flagged_failing']:.1%} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="tracelaw bench taubench", description=__doc__)
    parser.add_argument("--domains", nargs="+", default=list(DOMAINS))
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/taubench")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for domain in args.domains:
        runs = load(domain)
        results[domain] = {
            "new tasks": new_tasks(runs, args.alpha, args.folds, args.seed),
            "recurring tasks": recurring_tasks(runs, args.alpha, args.seed),
        }
        for setting, r in results[domain].items():
            print(
                f"{domain} ({setting}): AUROC {r['auroc']['Tracelaw']:.3f}, flagged "
                f"{r['flagged_failing']:.1%} of failing and {r['flagged_passing']:.1%} of "
                "passing runs",
                flush=True,
            )
    (out / "results.json").write_text(
        json.dumps({"alpha": args.alpha, "results": results}, indent=1)
    )
    table = markdown(results)
    (out / "results.md").write_text(table)
    print(table)
    return 0
