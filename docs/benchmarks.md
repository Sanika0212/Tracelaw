# Benchmarks

Every number in the README comes from these two suites; both run on a laptop CPU in a
few minutes. Runs write `results.json` and `results.md` under `runs/`.

## 1. Planted state bugs (`bench faults`)

```bash
uv run tracelaw bench faults --train 300 --test 200
```

1. **Workflows.** Four executable agents in `tracelaw.bench.workflows`, each a sequence
   of tool steps over a mutable state, with the randomness real agents have:
   1. `shop`: one to three searches before each of one to four cart additions, an
      optional coupon, checkout, a receipt;
   2. `travel`: a flight search, zero to three result pages, selection, booking for one
      to four passengers, payment, confirmation;
   3. `support`: a ticket read and classified as a refund, a question or a bug, then the
      matching branch (order lookup and refund; one to three answers; a filed bug);
   4. `research`: a plan, two to five rounds of search, read and note, an answer citing
      the notes.
2. **Faults.** One per run, at a chosen step; the run continues from the corrupted
   state:
   1. `wrong_update` (a total that ignores the quantity, a page counter that skips);
   2. `missed_update` (a total or a status not updated);
   3. `duplicate` (a tool effect applied twice);
   4. `wrong_argument` (a receipt to another address, one passenger too many);
   5. `loop` (two steps repeated as A, B, A, B);
   6. `skipped_step` (no order lookup before a refund, no receipt).

   Each workflow checks its final state like a test would; about 80% of faulty runs fail
   it, the rest are silent faults.
3. **Scenarios.**
   1. *Monitoring*: a lawbook learned from 300 good runs checks 200 new good runs (false
      alarms; the target is 5%) and 200 faulty runs (flagged, and flagged at the faulty
      step; for a loop, any step inside it counts).
   2. *Debugging*: for faulty runs that failed, is the method's top step the faulty one?
4. **Methods.** Tracelaw; laws alone and the trajectory alone (each with its own 5%
   threshold); single-run rules (an error, a step three times in a row, a run longer than
   any good run); the last step; a random step.
5. **Learning curve.** The same with 20, 50, 100 and 300 good runs to learn from.

## 2. Real agent conversations (`bench taubench`)

```bash
uv run tracelaw bench taubench --alpha 0.1
```

1. **Data.** τ-bench's published trajectories (MIT license, downloaded on first use):
   gpt-4o as a customer-service agent with tools on a retail database (115 tasks) and an
   airline database (50 tasks), four trials each; a run passes when the database ends in
   the right state. Conversations are converted by `tracelaw.importers.from_taubench`.
2. **Settings.**
   1. *New tasks*: five folds by task; a lawbook learned from the passing runs of four
      folds' tasks checks every run of the fifth.
   2. *Recurring tasks*: learned from the passing runs of trials 0 and 1, checked on
      trials 2 and 3 of the same tasks.
3. **Measures.** AUROC of the run score for "this run failed"; the share of passing and
   of failing runs flagged at `alpha = 0.1`; the same AUROC for the laws alone, the
   trajectory alone, the number of single-run rule hits and the run's length.

## Limits

1. The workflows are written for this benchmark; real agents have messier state. The
   τ-bench suite is there to show what happens on real runs, including where Tracelaw
   adds little.
2. τ-bench records no state, so its laws are about arguments, outputs and the
   conversation.
