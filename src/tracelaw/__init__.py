"""Tracelaw: learn the laws an agent's good runs obey, then find the first step where a
new run breaks one."""

from importlib.metadata import PackageNotFoundError, version

from tracelaw.lawbook import Finding, Lawbook, Verdict
from tracelaw.laws import Law, MiningOptions
from tracelaw.model import Run, Step, load_runs, save_runs

try:
    __version__ = version("tracelaw")
except PackageNotFoundError:  # pragma: no cover - only when running from a source tree
    __version__ = "0.0.0"

__all__ = [
    "Finding",
    "Law",
    "Lawbook",
    "MiningOptions",
    "Run",
    "Step",
    "Verdict",
    "__version__",
    "load_runs",
    "save_runs",
]
