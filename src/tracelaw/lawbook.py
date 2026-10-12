"""Learn a lawbook from good runs; check a new run against it.

Every step of a run gets a *score*: the surprisal of the step under the trajectory
model plus, for each law it breaks, the law's strength (``log(1 + support)``, the
surprisal of an event seen in none of ``support`` good observations). Unknown step
names and per-run counts outside the good range add to the score too. A run's score is
its highest step score.

**Calibrated alarms.** Alarms are calibrated on the good runs themselves, by
cross-conformal prediction: the runs are split into folds, a lawbook is learned on all
but one fold and scores the runs of that fold. A new run is scored by every fold's
lawbook and compared with the runs that lawbook did not see; the share of good runs
scoring at least as high is its p-value, and it raises an alarm when the p-value is at
most ``alpha``. A new good run then raises one with probability close to ``alpha`` (at
most ``2 alpha`` in theory, Vovk 2015), so "5% false alarms" is a setting, not a hope.
The lawbook learned on every run gives the findings and the step threshold.

**Localisation.** The first step whose score reaches the threshold is where the run
went wrong, with the evidence: each broken law, how many good observations it held in,
and what this run did instead.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tracelaw.flatten import variables
from tracelaw.laws import Law, MiningOptions, mine
from tracelaw.model import Run
from tracelaw.trajectory import Trajectory


@dataclass
class Finding:
    step: int  # index of the step; len(run) for something missing at the end
    name: str  # step name, or "<end>"
    kind: str  # "law", "transition", "unknown_step", "count"
    message: str
    score: float
    law: dict[str, Any] | None = None


@dataclass
class Verdict:
    run_id: str
    score: float  # the run's highest step score
    threshold: float
    p_value: float  # share of good calibration runs scoring at least this high
    flagged: bool
    first_step: int | None  # where the run first broke a law (None if not flagged)
    step_scores: list[float]
    findings: list[Finding] = field(default_factory=list)

    @property
    def culprit(self) -> list[Finding]:
        """Findings at the first flagged step."""
        return [f for f in self.findings if f.step == self.first_step]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Lawbook:
    laws: list[Law]
    trajectory: Trajectory
    threshold: float = math.inf
    alpha: float = 0.05
    calibration: list[float] = field(default_factory=list)
    runs: int = 0
    transition_weight: float = 1.0
    # cross-conformal: each fold's lawbook and the scores of the runs it did not see
    folds: list[Lawbook] = field(default_factory=list)
    fold_scores: list[list[float]] = field(default_factory=list)

    # ------------------------------------------------------------- learning

    @classmethod
    def fit(
        cls,
        runs: Sequence[Run],
        *,
        order: int = 3,
        options: MiningOptions | None = None,
        transition_weight: float = 1.0,
    ) -> Lawbook:
        """Mine laws and the trajectory model from good runs (no calibration)."""
        observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
        seen: set[tuple[str, str]] = set()
        for run in runs:
            for step in run.steps:
                values = variables(step)
                # the same event repeated (the same task run again) is one piece of
                # evidence, not several: count identical steps (state, inputs and outputs) once
                event = json.dumps(values, sort_keys=True, default=str)
                if (step.name, event) in seen:
                    continue
                seen.add((step.name, event))
                observations[step.name].append(values)
        laws = mine(observations, options)
        trajectory = Trajectory(order=order).fit([r.names for r in runs])
        return cls(laws, trajectory, runs=len(runs), transition_weight=transition_weight)

    @classmethod
    def learn(
        cls,
        runs: Sequence[Run],
        *,
        alpha: float = 0.05,
        folds: int = 5,
        order: int = 3,
        options: MiningOptions | None = None,
        transition_weight: float = 1.0,
        seed: int = 0,
    ) -> Lawbook:
        """Learn from good runs and calibrate the alarm threshold by cross-conformal."""
        good = [r for r in runs if r.passed is not False]
        if len(good) < 2 * folds:
            raise ValueError(f"need at least {2 * folds} good runs, got {len(good)}")
        order_ = list(range(len(good)))
        random.Random(seed).shuffle(order_)
        fold_books: list[Lawbook] = []
        fold_scores: list[list[float]] = []
        for f in range(folds):
            test = [good[i] for j, i in enumerate(order_) if j % folds == f]
            train = [good[i] for j, i in enumerate(order_) if j % folds != f]
            fold = cls.fit(train, order=order, options=options, transition_weight=transition_weight)
            fold_books.append(fold)
            fold_scores.append(sorted(fold.score(r)[0] for r in test))
        book = cls.fit(good, order=order, options=options, transition_weight=transition_weight)
        held_out = [x for scores in fold_scores for x in scores]
        book.alpha = alpha
        book.calibration = sorted(held_out)
        book.threshold = conformal_threshold(held_out, alpha)
        book.folds, book.fold_scores = fold_books, fold_scores
        return book

    # ------------------------------------------------------------- checking

    def score(self, run: Run) -> tuple[float, list[float], list[Finding]]:
        by_step: dict[str, list[Law]] = defaultdict(list)
        for law in self.laws:
            by_step[law.step].append(law)
        names = run.names
        surprisal = self.trajectory.surprisal(names)
        scores = [0.0] * (len(names) + 1)
        findings: list[Finding] = []
        for i, step in enumerate(run.steps):
            s = self.transition_weight * surprisal[i]
            scores[i] += s
            if step.name not in self.trajectory.vocab:
                findings.append(
                    Finding(i, step.name, "unknown_step", "this step never occurs in a good run", s)
                )
                continue
            if s >= math.log(20):  # rarer than 1 in 20 after this context: worth saying
                prev = names[i - 1] if i else "the start"
                findings.append(
                    Finding(
                        i,
                        step.name,
                        "transition",
                        f"{step.name} after {prev} is rare in good runs "
                        f"(probability {math.exp(-s):.3f})",
                        s,
                    )
                )
            values = variables(step)
            for law in by_step.get(step.name, []):
                if step.error is not None and any(v.startswith("out.") for v in law.vars):
                    continue  # the error explains missing or odd outputs: one finding, not many
                problem = law.check(values)
                if problem is not None:
                    scores[i] += law.strength
                    findings.append(
                        Finding(
                            i,
                            step.name,
                            "law",
                            f"{problem} (held in {law.support} good observations)",
                            law.strength,
                            law.to_dict(),
                        )
                    )
        end = self.transition_weight * surprisal[-1]
        scores[-1] += end
        if end >= math.log(20):
            findings.append(
                Finding(
                    len(names),
                    "<end>",
                    "transition",
                    f"good runs rarely end after {names[-1] if names else 'no steps'} "
                    f"(probability {math.exp(-end):.3f})",
                    end,
                )
            )
        seen: dict[str, int] = defaultdict(int)
        for i, name in enumerate(names):
            seen[name] += 1
            lo_hi = self.trajectory.occurrences.get(name)
            if lo_hi and seen[name] == lo_hi[1] + 1:
                strength = math.log1p(self.trajectory.runs)
                scores[i] += strength
                findings.append(
                    Finding(
                        i,
                        name,
                        "count",
                        f"{name} ran {lo_hi[1] + 1} times; good runs run it at most "
                        f"{lo_hi[1]} times",
                        strength,
                    )
                )
        for name, (lo, _) in self.trajectory.occurrences.items():
            if seen[name] < lo:
                strength = math.log1p(self.trajectory.runs)
                scores[-1] += strength
                findings.append(
                    Finding(
                        len(names),
                        "<end>",
                        "count",
                        f"{name} ran {seen[name]} times; every good run runs it at "
                        f"least {lo} times",
                        strength,
                    )
                )
        findings.sort(key=lambda f: (f.step, -f.score))
        return max(scores), scores, findings

    def p_value(self, run: Run, top: float | None = None) -> float:
        """Cross-conformal p-value: each fold's lawbook scores the run and compares it with
        the good runs that fold did not see (Vovk's cross-conformal predictor)."""
        if self.folds:
            n = sum(len(f) for f in self.fold_scores)
            above = 0
            for book, scores in zip(self.folds, self.fold_scores, strict=True):
                score = book.score(run)[0]
                above += sum(c >= score for c in scores)
            return (1 + above) / (n + 1)
        n = len(self.calibration)
        top = self.score(run)[0] if top is None else top
        return (1 + sum(c >= top for c in self.calibration)) / (n + 1) if n else 1.0

    def check(self, run: Run) -> Verdict:
        """Flag the run when its p-value is at most ``alpha``; point at the first step that
        reaches the threshold (or the highest-scoring step)."""
        top, scores, findings = self.score(run)
        p = self.p_value(run, top)
        flagged = p <= self.alpha if (self.folds or self.calibration) else top >= self.threshold
        first = None
        if flagged:
            first = next((i for i, s in enumerate(scores) if s >= self.threshold), None)
            if first is None:
                first = max(range(len(scores)), key=scores.__getitem__)
        return Verdict(run.id, top, self.threshold, p, flagged, first, scores, findings)

    def localize(self, run: Run) -> int:
        """The step most likely to be where the run went wrong (even below the threshold)."""
        _, scores, _ = self.score(run)
        first = next((i for i, s in enumerate(scores) if s >= self.threshold), None)
        if first is not None:
            return first
        return max(range(len(scores)), key=scores.__getitem__)

    # ------------------------------------------------------------- storage

    def to_dict(self) -> dict[str, Any]:
        return {
            "laws": [law.to_dict() for law in self.laws],
            "trajectory": self.trajectory.to_dict(),
            "threshold": self.threshold if math.isfinite(self.threshold) else None,
            "alpha": self.alpha,
            "calibration": self.calibration,
            "runs": self.runs,
            "transition_weight": self.transition_weight,
            "folds": [f.to_dict() for f in self.folds],
            "fold_scores": self.fold_scores,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Lawbook:
        threshold = data.get("threshold")
        return cls(
            laws=[Law.from_dict(x) for x in data["laws"]],
            trajectory=Trajectory.from_dict(data["trajectory"]),
            threshold=math.inf if threshold is None else float(threshold),
            alpha=data.get("alpha", 0.05),
            calibration=list(data.get("calibration", [])),
            runs=data.get("runs", 0),
            transition_weight=data.get("transition_weight", 1.0),
            folds=[cls.from_dict(f) for f in data.get("folds", [])],
            fold_scores=[list(x) for x in data.get("fold_scores", [])],
        )

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.write_text(json.dumps(self.to_dict(), indent=1))
        return out

    @classmethod
    def load(cls, path: str | Path) -> Lawbook:
        return cls.from_dict(json.loads(Path(path).read_text()))


def conformal_threshold(scores: Sequence[float], alpha: float) -> float:
    """Smallest threshold that at most ``alpha`` of new good runs reach (split conformal)."""
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return math.inf
    ordered = sorted(scores)
    # a run alarms when its score reaches the threshold, so step just past the k-th score
    return math.nextafter(ordered[k - 1], math.inf)
