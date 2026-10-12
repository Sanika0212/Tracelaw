"""Single-file HTML report of a checked run.

No server and no network: styles and data are inline. The page shows the verdict, then
the run step by step: each step's score as a bar against the alarm threshold, the first
step that broke a law marked, and under every step the laws it broke, how many good
observations each held in, and what this run did instead. It follows the system
light/dark preference.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from tracelaw import __version__
from tracelaw.lawbook import Lawbook, Verdict
from tracelaw.model import Run

KINDS = {
    "law": ("✗", "Broken law", "bad"),
    "transition": ("↷", "Unusual step order", "warn"),
    "unknown_step": ("?", "Step never seen", "bad"),
    "count": ("#", "Unusual count", "warn"),
}

_CSS = """
:root { color-scheme: light;
  --bg:#f9f9f7; --raised:#ffffff; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --line:#e1e0d9;
  --good:#0ca30c; --warn:#c98500; --bad:#d03b3b; --accent:#2a78d6; --track:#ecebe6;
  --bad-bg:rgba(208,59,59,.10); --warn-bg:rgba(250,178,25,.18); --good-bg:rgba(12,163,12,.10); }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) { color-scheme: dark;
  --bg:#0d0d0d; --raised:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --line:#2c2c2a; --accent:#3987e5;
  --warn:#eda100; --track:#262624; --bad-bg:rgba(208,59,59,.24); --warn-bg:rgba(250,178,25,.16);
  --good-bg:rgba(12,163,12,.20); } }
:root[data-theme="dark"] { color-scheme: dark; --bg:#0d0d0d; --raised:#1a1a19; --ink:#ffffff;
  --ink2:#c3c2b7; --line:#2c2c2a; --accent:#3987e5; --warn:#eda100; --track:#262624;
  --bad-bg:rgba(208,59,59,.24); --warn-bg:rgba(250,178,25,.16); --good-bg:rgba(12,163,12,.20); }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
main, header { max-width:1000px; margin:0 auto; padding:0 20px; }
header { padding-top:22px; }
h1 { font-size:21px; margin:0 0 6px; } h2 { font-size:16px; margin:24px 0 10px; }
.sub { color:var(--ink2); font-size:13.5px; }
.badge { display:inline-flex; gap:6px; align-items:center; border-radius:6px; padding:2px 9px; font-size:13px; font-weight:600; }
.good { color:var(--good); background:var(--good-bg); } .warn { color:var(--warn); background:var(--warn-bg); }
.bad { color:var(--bad); background:var(--bad-bg); }
.step { display:grid; grid-template-columns:42px 220px 1fr; gap:10px; align-items:center;
  padding:7px 10px; border-bottom:1px solid var(--line); }
.step.first { background:var(--bad-bg); border-radius:8px; }
.idx { color:var(--muted); font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px; }
.name { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:13.5px; overflow-wrap:anywhere; }
.track { position:relative; height:10px; background:var(--track); border-radius:5px; }
.fill { position:absolute; left:0; top:0; bottom:0; border-radius:5px; background:var(--accent); }
.fill.over { background:var(--bad); }
.cut { position:absolute; top:-4px; bottom:-4px; width:2px; background:var(--ink); }
.findings { grid-column:2 / 4; margin:4px 0 2px; font-size:13.5px; color:var(--ink2); }
.findings div { margin:3px 0; }
.legend { display:flex; gap:14px; flex-wrap:wrap; font-size:12.5px; color:var(--ink2); margin-top:8px; }
footer { color:var(--muted); font-size:12.5px; padding:30px 20px; text-align:center; }
"""


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def render(run: Run, verdict: Verdict, book: Lawbook, *, title: str | None = None) -> str:
    title = title or f"Run {run.id}"
    if verdict.flagged:
        status = '<span class="badge bad">✗ Breaks the laws of good runs</span>'
    else:
        status = '<span class="badge good">✓ Within the laws of good runs</span>'
    where = ""
    if verdict.first_step is not None:
        name = run.steps[verdict.first_step].name if verdict.first_step < len(run) else "the end"
        where = f" First broken at step {verdict.first_step} ({_e(name)})."
    summary = (
        f"p-value {verdict.p_value:.3f} against {len(book.calibration)} held-out good runs "
        f"(alarm at {book.alpha:.0%}); highest step score {verdict.score:.2f}, threshold "
        f"{verdict.threshold:.2f}.{where}"
    )
    top = max([*verdict.step_scores, verdict.threshold if verdict.threshold < 1e9 else 0, 1.0])
    rows = []
    names = [*run.names, "<end>"]
    for i, score in enumerate(verdict.step_scores):
        mine = [f for f in verdict.findings if f.step == i]
        over = score >= verdict.threshold
        cut = (
            f'<div class="cut" style="left:{100 * verdict.threshold / top:.1f}%"></div>'
            if verdict.threshold < 1e9
            else ""
        )
        details = "".join(
            f'<div><span class="badge {KINDS[f.kind][2]}">{KINDS[f.kind][0]} {KINDS[f.kind][1]}</span> '
            f"{_e(f.message)}</div>"
            for f in mine
        )
        rows.append(
            f'<div class="step{" first" if i == verdict.first_step else ""}">'
            f'<div class="idx">{i}</div><div class="name">{_e(names[i])}</div>'
            f'<div class="track"><div class="fill{" over" if over else ""}" '
            f'style="width:{100 * min(score, top) / top:.1f}%"></div>{cut}</div>'
            + (f'<div class="findings">{details}</div>' if details else "")
            + "</div>"
        )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(title)}</title><style>{_CSS}</style></head><body>
<header><h1>{_e(title)}</h1><div>{status}</div><p class="sub">{_e(summary)}</p>
<div class="legend"><span>Bar: how surprising the step is to a lawbook learned from {book.runs} good runs</span>
<span>Black line: alarm threshold</span></div></header>
<main><h2>Step by step</h2>{"".join(rows)}</main>
<footer>Tracelaw {_e(__version__)}: {len(book.laws)} laws learned from {book.runs} good runs.</footer>
</body></html>
"""


def write_report(run: Run, verdict: Verdict, book: Lawbook, path: str | Path) -> Path:
    out = Path(path)
    out.write_text(render(run, verdict, book), encoding="utf-8")
    return out
