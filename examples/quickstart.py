"""Learn what good runs of an agent look like, then catch a buggy run at the step it broke.

    uv run python examples/quickstart.py

The "agent" is a tiny refund workflow. Good runs refund the order's amount; the buggy
run refunds the amount twice. Nothing in the buggy run errors: every step looks fine on
its own, and only the laws of good runs say otherwise.
"""

import random

from tracelaw import Lawbook, Run, Step


def refund_run(rng: random.Random, *, double: bool = False) -> Run:
    order = {"id": f"o{rng.randint(10000, 99999)}", "amount": round(rng.uniform(5, 300), 2)}
    balance = round(rng.uniform(0, 500), 2)
    state = {"ticket": "open", "balance": balance, "refunds": 0}
    steps = [
        Step(
            "lookup_order",
            inputs={"order_id": order["id"]},
            outputs=order,
            state_before=dict(state),
            state_after=dict(state),
        )
    ]
    before = dict(state)
    times = 2 if double else 1
    state["balance"] = round(state["balance"] + times * order["amount"], 2)
    state["refunds"] += times
    steps.append(
        Step(
            "refund",
            inputs={"order_id": order["id"], "amount": order["amount"]},
            outputs={"ok": True},
            state_before=before,
            state_after=dict(state),
        )
    )
    before = dict(state)
    state["ticket"] = "closed"
    steps.append(Step("close_ticket", state_before=before, state_after=dict(state)))
    return Run(f"run-{order['id']}", steps)


rng = random.Random(0)
book = Lawbook.learn([refund_run(rng) for _ in range(100)], alpha=0.05)
for law in book.laws:
    if law.step == "refund" and law.kind in ("plus", "delta", "equal"):
        print("learned:", law.describe())

verdict = book.check(refund_run(rng, double=True))
print(
    f"\nflagged: {verdict.flagged} (p = {verdict.p_value:.3f}), first broken at step "
    f"{verdict.first_step}"
)
for finding in verdict.culprit:
    print("  ", finding.message)
