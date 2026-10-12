# How Tracelaw works

An agent that fails rarely fails loudly. It charges the wrong amount, books one seat
too many, cites a note it never wrote, or loops politely through the same two tools.
Each step looks plausible on its own, and the error surfaces several steps later, if at
all. Tracing tools record everything but judge each run alone, with rules such as "the
same step three times in a row is a loop", so they miss what is wrong for *this* agent
and flag what is normal for it.

Tracelaw judges a run against the agent's own good runs. It learns the laws those runs
obey, checks a new run against them, and points at the first step that broke one.

```text
good runs ──▶ variables per step ──▶ laws (with support) ─┐
          └─▶ step sequences ──────▶ trajectory model ────┼─▶ lawbook ──▶ check a new run
          └─▶ held-out folds ──────▶ calibration ─────────┘              (alarm, first step, evidence)
```

## 1. Runs, steps and variables

A run is a list of steps. Each step has a name and up to four records: the state before
and after it, its inputs (a tool's arguments) and its outputs. Nested values are
flattened into named variables (`tracelaw.flatten`):

1. `pre.cart.total`, `post.cart.total`: state before and after;
2. `in.user_id`, `out.status`: an argument and an output;
3. lists are summarised, because their length varies: `post.cart.items#len` (the
   length), `post.cart.items[].price` (the prices as a list), `post.ids[]` (a list of
   scalars).

Runs come from JSON Lines, chat transcripts with tool calls (`from_messages`, and
τ-bench through it), OpenTelemetry GenAI spans (`from_otel`) or Counterfact recordings
(`from_counterfact`). Chat transcripts get a conversation state: every id-like value
seen so far and whether the user's last message agreed to something.

## 2. Laws

For every step name, candidate laws are tested against every good occurrence of that
step and kept only if they held in all of them, at least five times
(`tracelaw.laws`):

| Law | Example |
|---|---|
| present, kind | `post.booking` is always set, and is always a mapping |
| range, one of | `in.qty` is within [0, 4]; `out.category` is refund, question or bug |
| unchanged, delta, monotone | checkout leaves the cart unchanged; `post.page == pre.page + 1`; a balance never decreases |
| equal | `in.amount == pre.cart.total`; `in.user_id == pre.user.id` |
| sum of, plus, product | `total == sum(items[].line)`; `post.balance == pre.balance + in.amount`; `line == price × qty` |
| member, subset | `in.order_id` is one of the ids seen so far; every `in.item_ids[]` is |
| each (list elements) | no element of `post.notes[].url` is empty |

Four rules keep the laws honest:

1. **Support.** Each law records how many good observations it held in. A broken law's
   strength is `log(1 + support)`, the surprisal of an event never seen in that many
   tries, so breaking a law seen 500 times outweighs breaking one seen 5 times.
2. **Independent evidence.** Identical tool events (the same inputs and outputs, as when
   a benchmark repeats a task) count once.
3. **Open-ended values.** A field gets a "one of" law only if every allowed value was
   seen at least twice; values seen once mean the next one is likely new (the
   Good-Turing argument). Ranges are widened by half their spread on each side, so a
   value just past the largest seen is not an alarm, and a value never seen below zero
   stays non-negative.
4. **One fact, one law.** Variables that are always equal form a group; value laws and
   relations are mined on one representative, so a single wrong value counts once.

When a step fails with an error, laws about its outputs are skipped: the error is the
finding, not every missing field.

## 3. The trajectory

A variable-order Markov model over step names (`tracelaw.trajectory`) learns what
normally comes next, with Witten-Bell smoothing from the longest context seen (up to
three steps) down to a uniform distribution that keeps room for unseen step names. The
end of the run is a symbol too, so a run that stops early is surprising at its end.
The model also learns how many times each step runs in a good run.

This catches what fixed rules cannot: an `A B A B` loop, a skipped step, an extra one,
a run cut short. It does not flag repetition that good runs also have (three searches
before adding an item, paging through results).

## 4. Scores, alarms and the first broken step

Each step's score is its surprisal under the trajectory model plus the strength of every
law it breaks, plus a count finding when a step runs more often than in any good run. A
run's score is its highest step score.

Alarms are calibrated by **cross-conformal prediction** (`tracelaw.lawbook`): the good
runs are split into five folds; a lawbook learned on four folds scores the runs of the
fifth. A new run is scored by every fold's lawbook and compared with the runs that
lawbook did not see; the share of good runs scoring at least as high is its p-value. It
raises an alarm when the p-value is at most `alpha`. A new good run then raises one with
probability close to `alpha` (at most `2 alpha` in theory), so the false-alarm rate is a
setting, not a hope.

The first step whose score reaches the threshold is where the run went wrong. Each
finding says which law broke, how many good observations it held in, and what this run
did instead:

```text
step 11 checkout: post.order.paid = 588.62 but pre.cart.total = 294.31 (held in 200 good observations)
```

## Limitations

1. **Laws need the right variables.** State bugs are visible only if the state is
   recorded. Failures that are a wrong choice among valid options (the wrong item from a
   correct search) break no law; τ-bench retail is mostly such failures.
2. **New kinds of tasks look unusual.** A lawbook learned on one mix of requests flags
   more runs from a very different mix. Learn from the runs you monitor.
3. **Calibration assumes exchangeable good runs.** If the agent, its tools or its users
   change, learn a new lawbook.
4. **Laws are exact.** A field that is right in 999 runs and legitimately different in
   the 1000th breaks its law in that run; such laws drop out when relearned on runs that
   include the exception.
