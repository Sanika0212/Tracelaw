# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-12

### Added

1. Law mining over the state, arguments and outputs of every step type: presence, type,
   range, allowed values, unchanged, delta, monotone, equality, sums, products,
   membership in values already seen, and laws over list elements; each with its
   support.
2. A variable-order Markov model of step sequences with per-run step counts.
3. `Lawbook`: cross-conformal alarms at a chosen false-alarm rate, a p-value per run,
   and the first step that broke a law, with every broken law spelled out.
4. Importers for JSON Lines, chat transcripts with tool calls (and τ-bench), OpenTelemetry
   GenAI spans and Counterfact recordings.
5. A single-file HTML report, `assert_obeys` with a pytest fixture, and the
   `tracelaw learn | check | explain | bench` command line.
6. Benchmarks: planted state bugs in four executable workflows with baselines and a
   learning curve, and real τ-bench conversations.
