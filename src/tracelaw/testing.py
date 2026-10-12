"""Use a lawbook as regression tests.

.. code-block:: python

    from tracelaw.testing import assert_obeys

    def test_checkout_agent(lawbook):          # fixture from the pytest plugin
        run = record_my_agent("buy two lamps")  # however you capture a run
        assert_obeys(run, lawbook)

``assert_obeys`` fails with the broken laws spelled out, starting with the first step
that broke one. Point the plugin at a lawbook with ``--tracelaw-laws laws.json`` or the
``tracelaw_laws`` ini option.
"""

from __future__ import annotations

from tracelaw.lawbook import Lawbook, Verdict
from tracelaw.model import Run


def explain(verdict: Verdict, run: Run, *, limit: int = 8) -> str:
    lines = [
        f"run {verdict.run_id} breaks the laws of good runs "
        f"(score {verdict.score:.2f}, threshold {verdict.threshold:.2f}, p = {verdict.p_value:.3f})"
    ]
    if verdict.first_step is not None:
        name = run.steps[verdict.first_step].name if verdict.first_step < len(run) else "<end>"
        lines.append(f"first broken at step {verdict.first_step} ({name}):")
    for f in verdict.findings[:limit]:
        lines.append(f"  step {f.step} {f.name}: {f.message}")
    if len(verdict.findings) > limit:
        lines.append(f"  ... and {len(verdict.findings) - limit} more")
    return "\n".join(lines)


def assert_obeys(run: Run, lawbook: Lawbook, *, strict: bool = False) -> Verdict:
    """Fail if ``run`` raises an alarm; with ``strict``, fail on any broken law at all."""
    verdict = lawbook.check(run)
    broken = verdict.flagged or (strict and any(f.kind == "law" for f in verdict.findings))
    if broken:
        raise AssertionError(explain(verdict, run))
    return verdict
