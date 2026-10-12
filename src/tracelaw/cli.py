"""Command line: ``tracelaw learn``, ``check``, ``explain`` and ``bench``.

tracelaw learn good-runs.jsonl --out laws.json
tracelaw check new-runs.jsonl --laws laws.json --html report.html
tracelaw explain laws.json --step checkout
tracelaw bench faults
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from tracelaw import __version__
from tracelaw.model import Run

FORMATS = ("jsonl", "taubench", "otel", "counterfact")


def read(paths: Sequence[str], fmt: str) -> list[Run]:
    from tracelaw import importers
    from tracelaw.model import load_runs

    readers = {
        "jsonl": load_runs,
        "taubench": importers.from_taubench,
        "otel": importers.from_otel,
        "counterfact": importers.from_counterfact,
    }
    return [run for path in paths for run in readers[fmt](path)]


def learn_command(args: argparse.Namespace) -> int:
    from tracelaw.lawbook import Lawbook

    runs = read(args.runs, args.format)
    good = [r for r in runs if r.passed is not False]
    book = Lawbook.learn(good, alpha=args.alpha, folds=args.folds)
    book.save(args.out)
    print(
        f"learned {len(book.laws)} laws from {len(good)} good runs "
        f"({len(runs) - len(good)} failing runs left out); alarms at {args.alpha:.0%}, "
        f"threshold {book.threshold:.2f}; saved to {args.out}"
    )
    return 0


def check_command(args: argparse.Namespace) -> int:
    from tracelaw.lawbook import Lawbook
    from tracelaw.report import write_report
    from tracelaw.testing import explain

    book = Lawbook.load(args.laws)
    runs = read(args.runs, args.format)
    flagged = 0
    verdicts = []
    for run in runs:
        verdict = book.check(run)
        verdicts.append(verdict.to_dict())
        if verdict.flagged:
            flagged += 1
            print(explain(verdict, run, limit=args.limit))
            print()
        if args.html:
            path = Path(args.html)
            target = (
                path if len(runs) == 1 else path.with_name(f"{path.stem}-{run.id}{path.suffix}")
            )
            write_report(run, verdict, book, target)
    print(f"{flagged} of {len(runs)} runs break the laws of good runs")
    if args.json:
        Path(args.json).write_text(json.dumps(verdicts, indent=1))
    return 1 if flagged and args.fail else 0


def explain_command(args: argparse.Namespace) -> int:
    from tracelaw.lawbook import Lawbook
    from tracelaw.laws import PRIORITY

    book = Lawbook.load(args.laws)
    laws = [law for law in book.laws if args.step in (None, law.step)]
    laws.sort(key=lambda law: (law.step, PRIORITY.get(law.kind, 9), -law.support))
    for law in laws[: args.limit]:
        print(f"{law.describe()}  (held in {law.support} good observations)")
    if len(laws) > args.limit:
        print(f"... and {len(laws) - args.limit} more")
    if args.step is None:
        print()
        for name, (lo, hi) in sorted(book.trajectory.occurrences.items()):
            print(f"{name}: runs {lo} to {hi} times per good run")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["bench"]:
        from tracelaw.bench import run

        if len(argv) < 2:
            print("usage: tracelaw bench {faults,taubench} [options]")
            return 2
        return run(argv[1], argv[2:])
    parser = argparse.ArgumentParser(prog="tracelaw", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"tracelaw {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    lp = sub.add_parser("learn", help="learn a lawbook from good runs")
    lp.add_argument("runs", nargs="+")
    lp.add_argument("--format", choices=FORMATS, default="jsonl")
    lp.add_argument("--out", default="laws.json")
    lp.add_argument("--alpha", type=float, default=0.05, help="target false-alarm rate")
    lp.add_argument("--folds", type=int, default=5)
    lp.set_defaults(func=learn_command)
    cp = sub.add_parser("check", help="check runs against a lawbook")
    cp.add_argument("runs", nargs="+")
    cp.add_argument("--laws", required=True)
    cp.add_argument("--format", choices=FORMATS, default="jsonl")
    cp.add_argument("--html", help="write an HTML report (one per run if several)")
    cp.add_argument("--json", help="write every verdict as JSON")
    cp.add_argument("--limit", type=int, default=8, help="findings to print per run")
    cp.add_argument("--fail", action="store_true", help="exit 1 if any run is flagged")
    cp.set_defaults(func=check_command)
    ep = sub.add_parser("explain", help="list the laws in a lawbook")
    ep.add_argument("laws")
    ep.add_argument("--step")
    ep.add_argument("--limit", type=int, default=40)
    ep.set_defaults(func=explain_command)
    sub.add_parser("bench", help="run a benchmark suite: faults or taubench")
    args = parser.parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
