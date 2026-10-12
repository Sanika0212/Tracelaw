"""Benchmarks: ``faults`` (planted state bugs in executable workflows) and ``taubench``
(real agent conversations from τ-bench, passing and failing).

Run with ``tracelaw bench SUITE [options]``; each suite documents its options.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence


def run(suite: str, argv: Sequence[str]) -> int:
    module = importlib.import_module(f"tracelaw.bench.{suite}")
    code: int = module.main(list(argv))
    return code
