# Contributing

Thanks for your interest in Tracelaw.

## Development setup

```bash
git clone https://github.com/Sanika0212/Tracelaw.git
cd Tracelaw
uv sync --extra bench --extra viz
```

## Checks

Every pull request must pass the same checks CI runs:

```bash
uv run ruff check src tests scripts examples
uv run ruff format --check src tests scripts examples
uv run mypy
uv run coverage run -m pytest && uv run coverage report --fail-under=85
```

## Ground rules

1. **Tests never touch the network.** Use the workflows in `tracelaw.bench.workflows` and
   small hand-made runs; the τ-bench loader is replaced in tests.
2. **Numbers come from runs.** Every number in the README must be reproducible with
   `tracelaw bench` and `scripts/make_figures.py`, and
   [docs/benchmarks.md](docs/benchmarks.md) must list the exact commands.
3. **Every alarm explains itself.** A new law or signal must say what held in good runs,
   how often, and what the run did instead; a bare score is not a finding.
4. **False alarms stay calibrated.** A change to scoring must keep the measured
   false-alarm rate near `alpha` on held-out good runs.
5. Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`,
   `docs:`, `test:`, `chore:`, `ci:`).

## License of contributions

By contributing, you agree that your contributions are licensed under the Apache
License 2.0, the license of this project.
