"""What normally happens next: a variable-order Markov model over step names.

Each step's probability given the steps before it is estimated from good runs with
Witten-Bell smoothing, backing off from the longest context seen (up to ``order``
steps) to shorter ones and finally to a uniform distribution that leaves room for step
names never seen before. A step's *surprisal* is ``-log P(step | context)`` in nats;
the end of the run is scored too, so a run that stops early is surprising at its end.

This catches what single-run rules miss: an ``A B A B`` loop where ``A A A`` rules see
nothing, a step skipped or added, a run that ends half-way. It does not flag legitimate
repetition (paging through results) when good runs repeat the same way.

Counts per run are learned alongside: how many times each step name occurs in a good
run (``checkout`` exactly once, ``search`` one to four times).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

START, END = "<start>", "<end>"


@dataclass
class Trajectory:
    order: int = 3
    counts: dict[tuple[str, ...], Counter[str]] = field(default_factory=dict)
    vocab: set[str] = field(default_factory=set)
    occurrences: dict[str, tuple[int, int]] = field(default_factory=dict)  # name: (min, max)
    runs: int = 0

    def fit(self, sequences: Sequence[Sequence[str]]) -> Trajectory:
        table: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        per_run: list[Counter[str]] = []
        for names in sequences:
            seq = [START, *names, END]
            for i in range(1, len(seq)):
                for k in range(0, self.order + 1):
                    if i - k < 0:
                        break
                    table[tuple(seq[i - k : i])][seq[i]] += 1
            per_run.append(Counter(names))
            self.vocab.update(names)
        self.counts = dict(table)
        self.runs = len(sequences)
        self.occurrences = {
            name: (min(c[name] for c in per_run), max(c[name] for c in per_run))
            for name in self.vocab
        }
        return self

    def probability(self, context: Sequence[str], symbol: str) -> float:
        """Witten-Bell interpolation from the longest seen context down to uniform."""
        size = len(self.vocab) + 2  # names, the end marker, and one unseen name
        p = 1.0 / size
        for k in range(0, min(self.order, len(context)) + 1):
            ctx = tuple(context[len(context) - k :]) if k else ()
            seen = self.counts.get(ctx)
            if not seen:
                break
            total = sum(seen.values())
            distinct = len(seen)
            lam = total / (total + distinct)
            p = lam * seen.get(symbol, 0) / total + (1 - lam) * p
        return p

    def surprisal(self, names: Sequence[str]) -> list[float]:
        """``-log P`` of each step, then of the run ending where it did (one longer)."""
        seq = [START, *names, END]
        return [
            -math.log(max(self.probability(seq[max(0, i - self.order) : i], seq[i]), 1e-12))
            for i in range(1, len(seq))
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "counts": [[list(k), dict(v)] for k, v in self.counts.items()],
            "vocab": sorted(self.vocab),
            "occurrences": {k: list(v) for k, v in self.occurrences.items()},
            "runs": self.runs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trajectory:
        return cls(
            order=data["order"],
            counts={tuple(k): Counter(v) for k, v in data["counts"]},
            vocab=set(data["vocab"]),
            occurrences={k: (v[0], v[1]) for k, v in data["occurrences"].items()},
            runs=data["runs"],
        )
