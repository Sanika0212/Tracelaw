"""Build every README figure.

    uv run --extra viz python scripts/make_figures.py

Inputs: ``runs/faults/results.json`` and ``runs/taubench/results.json`` (from
``tracelaw bench``). The animation and the report screenshot check a shop run with a
planted bug against a lawbook learned here. Outputs go to ``docs/assets/`` as
``<name>-light.*`` and ``<name>-dark.*``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

ASSETS = Path("docs/assets")
DPI = 200
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    raised: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    neutral: str
    series: tuple[str, ...]
    bad: str


THEMES = {
    "light": Theme(
        "light",
        "#fcfcfb",
        "#ffffff",
        "#0b0b0b",
        "#52514e",
        "#898781",
        "#e1e0d9",
        "#c3c2b7",
        "#a8a79f",
        ("#2a78d6", "#eb6834", "#1baf7a", "#eda100"),
        "#d03b3b",
    ),
    "dark": Theme(
        "dark",
        "#1a1a19",
        "#222220",
        "#ffffff",
        "#c3c2b7",
        "#898781",
        "#2c2c2a",
        "#383835",
        "#64635e",
        ("#3987e5", "#d95926", "#199e70", "#c98500"),
        "#e05252",
    ),
}
METHODS = {
    "Tracelaw": 0,
    "laws only": 1,
    "trajectory only": 2,
    "single-run rules": 3,
}


def use_theme(mode: str) -> Theme:
    theme = THEMES[mode]
    plt.rcParams.update(
        {
            "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9.5,
            "figure.facecolor": theme.surface,
            "axes.facecolor": theme.surface,
            "savefig.facecolor": theme.surface,
            "text.color": theme.ink,
            "axes.labelcolor": theme.ink2,
            "axes.edgecolor": theme.axis,
            "xtick.color": theme.muted,
            "ytick.color": theme.muted,
            "xtick.labelcolor": theme.ink2,
            "ytick.labelcolor": theme.ink2,
            "grid.color": theme.grid,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )
    return theme


def _title(fig: Figure, title: str, subtitle: str, theme: Theme) -> None:
    fig.text(0.012, 0.975, title, ha="left", va="top", fontsize=12.5, weight="bold")
    fig.text(
        0.012,
        0.975 - 0.32 / fig.get_figheight(),
        subtitle,
        ha="left",
        va="top",
        fontsize=9,
        color=theme.ink2,
    )


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(path)
    return path


# ------------------------------------------------------------------------ banner


def banner(mode: str) -> str:
    t = THEMES[mode]
    width, height = 1200, 260
    x0, y = 610, 150
    shapes = []
    heights = [18, 26, 14, 30, 22, 96, 70, 58]
    for i, h in enumerate(heights):
        bad = i >= 5
        color = t.bad if bad else t.series[0]
        shapes.append(
            f'<rect x="{x0 + i * 62}" y="{y - h}" width="40" height="{h}" rx="6" fill="{color}" '
            f'fill-opacity="{0.95 if bad else 0.75}"/>'
        )
        shapes.append(
            f'<rect x="{x0 + i * 62}" y="{y + 14}" width="40" height="40" rx="8" fill="{t.raised}" '
            f'stroke="{t.bad if i == 5 else t.axis}" stroke-width="{2.5 if i == 5 else 1.5}"/>'
        )
    shapes.append(
        f'<line x1="{x0 - 14}" y1="{y - 46}" x2="{x0 + 8 * 62}" y2="{y - 46}" stroke="{t.ink}" '
        'stroke-width="2" stroke-dasharray="6 5"/>'
    )
    shapes.append(
        f'<text x="{x0 + 8 * 62 - 4}" y="{y - 54}" font-family="{FONT}" font-size="13" '
        f'text-anchor="end" fill="{t.ink2}">alarm threshold (5% false alarms)</text>'
    )
    shapes.append(
        f'<text x="{x0 + 5 * 62 + 20}" y="{y + 76}" font-family="{FONT}" font-size="14" '
        f'text-anchor="middle" fill="{t.bad}">first broken law</text>'
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Tracelaw: learn the laws an agent's good runs obey, then find the first step where a new run breaks one">
  <rect width="{width}" height="{height}" rx="18" fill="{t.surface}"/>
  <text x="56" y="118" font-family="{FONT}" font-size="64" font-weight="700" fill="{t.ink}">Tracelaw</text>
  <text x="58" y="162" font-family="{FONT}" font-size="22" fill="{t.ink2}">Learn the laws your agent's good runs obey,</text>
  <text x="58" y="192" font-family="{FONT}" font-size="22" fill="{t.ink2}">then find the first step a run breaks one.</text>
  {"".join(shapes)}
</svg>
"""


# ----------------------------------------------------------------------- charts


def plot_faults(summary: dict[str, Any], out: Path, mode: str) -> Path:
    theme = use_theme(mode)
    results = summary["results"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9), sharey=True)
    _title(
        fig,
        "Which step broke the run?",
        "Four executable agents, 200 runs each with one planted state bug, and 200 good runs. "
        "Left: flagged at the\nfaulty step with alarms at 5% false alarms. Right: the run is "
        "known to have failed; is the top step the faulty one?",
        theme,
    )
    names = [r["workflow"] for r in results]
    for ax, key in zip(axes, ("monitoring", "debugging"), strict=True):
        for name, slot in METHODS.items():
            xs = []
            for r in results:
                value = r[key][name]
                xs.append(value["flagged_at_fault"] if key == "monitoring" else value)
            ys = [i + 0.27 - 0.18 * slot for i in range(len(names))][::-1]
            ax.scatter(
                xs,
                ys,
                s=64 if slot == 0 else 40,
                color=theme.series[slot],
                edgecolors=theme.surface,
                linewidths=1.2,
                zorder=3,
                label=name,
            )
        if key == "debugging":
            for i, r in enumerate(results[::-1]):
                ax.scatter(
                    [r["debugging"]["random step"]],
                    [i - 0.27],
                    s=30,
                    marker="|",
                    color=theme.ink2,
                    zorder=3,
                )
        ax.set_xlim(-0.02, 1.04)
        ax.set_xticks([0, 0.5, 1.0], ["0%", "50%", "100%"])
        ax.set_yticks(range(len(names)), names[::-1])
        ax.grid(True, axis="x")
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=5)
        ax.set_title(
            "monitoring" if key == "monitoring" else "debugging",
            loc="left",
            fontsize=9.5,
            color=theme.ink,
        )
    handles = [
        Line2D([0], [0], color=theme.series[s], lw=0, marker="o", ms=7) for s in METHODS.values()
    ]
    handles.append(Line2D([0], [0], color=theme.ink2, lw=0, marker="|", ms=8))
    fig.legend(
        handles, [*METHODS, "random step"], loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0)
    )
    fig.subplots_adjust(left=0.1, right=0.98, top=1 - 1.25 / 3.9, bottom=0.2, wspace=0.08)
    return _save(fig, out)


def plot_curve(summary: dict[str, Any], out: Path, mode: str) -> Path:
    theme = use_theme(mode)
    curve = summary["curve"]
    sizes = sorted(int(k) for k in curve)
    fig = plt.figure(figsize=(7.6, 3.2))
    _title(
        fig,
        "How many good runs does it need?",
        "Average over the four agents. A few dozen good runs are enough to find most planted "
        "bugs at the step they happened.",
        theme,
    )
    ax = fig.add_axes((0.08, 0.16, 0.6, 0.6))
    series = [
        ("debugging", "debugging: top step is the faulty one", 0),
        ("flagged_at_fault", "monitoring: flagged at the faulty step", 2),
        ("false_alarms", "false alarms (target 5%)", 1),
    ]
    for key, label, slot in series:
        ys = [curve[str(n)][key] for n in sizes]
        ax.plot(
            sizes,
            ys,
            color=theme.series[slot],
            lw=2,
            marker="o",
            ms=5,
            markeredgecolor=theme.surface,
            label=label,
        )
    ax.axhline(0.05, color=theme.neutral, lw=1, ls=(0, (3, 2)))
    ax.set_xscale("log")
    ax.set_xticks(sizes, [str(n) for n in sizes])
    ax.minorticks_off()
    ax.set_ylim(0, 1.04)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("good runs learned from")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=4)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="center left",
        bbox_to_anchor=(0.7, 0.45),
        labelspacing=1.2,
    )
    return _save(fig, out)


def plot_taubench(results: dict[str, Any], out: Path, mode: str) -> Path:
    theme = use_theme(mode)
    labels = {"gpt-4o-retail": "retail", "gpt-4o-airline": "airline"}
    methods = ["Tracelaw", "trajectory only", "laws only", "single-run rules", "run length"]
    rows = [(f"{labels.get(d, d)}, {s}", r) for d, by in results.items() for s, r in by.items()]
    fig = plt.figure(figsize=(7.6, 0.55 * len(rows) + 1.9))
    _title(
        fig,
        "Real agent runs: τ-bench, gpt-4o",
        "AUROC for telling failing runs from passing ones Tracelaw has not seen (50% = "
        "chance). Learned only from passing runs.",
        theme,
    )
    top = 1 - 1.0 / fig.get_figheight()
    ax = fig.add_axes((0.27, 1.0 / fig.get_figheight(), 0.7, top - 1.0 / fig.get_figheight()))
    colors = [theme.series[0], theme.series[2], theme.series[1], theme.series[3], theme.neutral]
    for i, (_, r) in enumerate(rows[::-1]):
        for j, m in enumerate(methods):
            ax.scatter(
                [r["auroc"][m]],
                [i + 0.24 - 0.12 * j],
                s=60 if j == 0 else 36,
                color=colors[j],
                edgecolors=theme.surface,
                linewidths=1.2,
                zorder=3,
            )
    ax.set_yticks(range(len(rows)), [n for n, _ in rows][::-1])
    ax.set_xlim(0.45, 0.8)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8], ["50%", "60%", "70%", "80%"])
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=5)
    handles = [Line2D([0], [0], color=c, lw=0, marker="o", ms=7) for c in colors]
    fig.legend(handles, methods, loc="lower center", ncol=5, bbox_to_anchor=(0.6, 0.0))
    return _save(fig, out)


# -------------------------------------------------------------------- animation


def animate_check(run: Any, verdict: Any, out: Path, mode: str) -> Path:
    """Steps light up one by one; the bar crosses the threshold at the first broken law."""
    theme = use_theme(mode)
    names = [*run.names, "<end>"]
    scores = verdict.step_scores
    n = len(scores)
    fig = plt.figure(figsize=(7.6, 0.32 * n + 1.9))
    _title(
        fig,
        "Checking a run, step by step",
        "A shop agent with a planted bug: checkout charges twice the cart total. Bars: how "
        "surprising each step is to the\nlawbook of 200 good runs. Dashed line: the alarm "
        "threshold.",
        theme,
    )
    top = 1 - 1.05 / fig.get_figheight()
    ax = fig.add_axes((0.2, 0.04, 0.42, top - 0.04))
    note = fig.add_axes((0.65, 0.04, 0.33, top - 0.04))
    note.axis("off")
    hi = max(max(scores), verdict.threshold) * 1.1
    ax.set_xlim(0, hi)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(range(n), names[::-1])
    ax.set_xticks([])
    ax.tick_params(length=0, pad=4)
    ax.spines["bottom"].set_visible(False)
    ax.axvline(verdict.threshold, color=theme.ink, lw=1.2, ls=(0, (4, 3)))
    bars = ax.barh(range(n)[::-1], [0] * n, height=0.6, color=theme.series[0])
    text = note.text(0.0, 0.98, "", va="top", ha="left", fontsize=8.6, color=theme.ink, wrap=True)
    frames = list(range(n)) + [n - 1] * 10

    def draw(k: int) -> list[Any]:
        for i in range(n):
            score = scores[i] if i <= frames[k] else 0
            bars[i].set_width(score)
            bars[i].set_color(theme.bad if score >= verdict.threshold else theme.series[0])
        first = verdict.first_step
        if first is not None and frames[k] >= first:
            found = [f.message for f in verdict.findings if f.step == first][:2]
            text.set_text(
                f"first broken at step {first} ({names[first]}):\n\n" + "\n\n".join(found)
            )
        return []

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=400)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=3), dpi=110)
    plt.close(fig)
    print(out)
    return out


def screenshot(page: str, name: str) -> None:
    if not CHROME.exists():
        print(f"skipping screenshots: {CHROME} not found")
        return
    with tempfile.TemporaryDirectory() as tmp:
        for mode in ("light", "dark"):
            themed = Path(tmp) / f"{name}-{mode}.html"
            themed.write_text(page.replace("<html", f'<html data-theme="{mode}"', 1))
            target = ASSETS / f"{name}-{mode}.png"
            subprocess.run(
                [
                    str(CHROME),
                    "--headless=new",
                    "--hide-scrollbars",
                    "--force-device-scale-factor=2",
                    "--window-size=1000,900",
                    f"--screenshot={target.resolve()}",
                    themed.as_uri(),
                ],
                check=True,
                capture_output=True,
            )
            print(target)


def demo() -> tuple[Any, Any, Any]:
    from tracelaw import Lawbook
    from tracelaw.bench.workflows import Fault, good_runs, shop

    book = Lawbook.learn(good_runs("shop", 200, seed=1), alpha=0.05)
    import random

    rng = random.Random(11)
    run, _ = shop(rng, Fault("duplicate", "checkout", 0))
    return run, book.check(run), book


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faults", default="runs/faults/results.json")
    parser.add_argument("--taubench", default="runs/taubench/results.json")
    args = parser.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    for mode in THEMES:
        (ASSETS / f"banner-{mode}.svg").write_text(banner(mode))
        print(ASSETS / f"banner-{mode}.svg")
    if Path(args.faults).exists():
        summary = json.loads(Path(args.faults).read_text())
        for mode in THEMES:
            plot_faults(summary, ASSETS / f"faults-{mode}.png", mode)
            plot_curve(summary, ASSETS / f"curve-{mode}.png", mode)
    if Path(args.taubench).exists():
        results = json.loads(Path(args.taubench).read_text())["results"]
        for mode in THEMES:
            plot_taubench(results, ASSETS / f"taubench-{mode}.png", mode)
    run, verdict, book = demo()
    for mode in THEMES:
        animate_check(run, verdict, ASSETS / f"check-{mode}.gif", mode)
    from tracelaw.report import render

    screenshot(render(run, verdict, book, title="A shop run that charged twice"), "report")


if __name__ == "__main__":
    main()
