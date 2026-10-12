"""Executable agent workflows with planted state bugs.

Four small agents, each a sequence of named steps over a mutable state, with the
randomness real agents have: a varying number of searches and retries, optional steps,
paging, different branches per ticket. Every step is recorded with the state before and
after it, its inputs and its outputs.

A *fault* corrupts one step of an otherwise good run, and the run carries on from the
corrupted state, as a real bug would:

==================  ===============================================================
fault               what happens at the chosen step
==================  ===============================================================
``wrong_update``    a state field is updated wrongly (a total ignores the quantity)
``missed_update``   a state update is skipped (the total is not updated)
``duplicate``       a tool's effect is applied twice (a refund paid twice)
``wrong_argument``  a tool is called with a wrong argument (the receipt to another user)
``loop``            two steps repeat as A, B, A, B before the run continues
``skipped_step``    a required step is skipped (no order lookup before the refund)
==================  ===============================================================

The ground truth is the index of the first faulty step (for a skipped step, the step
that ran in its place). Each workflow also checks its final state, so a run's outcome
(passed or failed) is decided the way a test would decide it.
"""

from __future__ import annotations

import copy
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tracelaw.model import Run, Step

FAULTS = ("wrong_update", "missed_update", "duplicate", "wrong_argument", "loop", "skipped_step")


@dataclass(frozen=True)
class Fault:
    kind: str
    step: str
    occurrence: int = 0  # which occurrence of the step (0 = first)


class Recorder:
    def __init__(self, state: dict[str, Any], rng: random.Random, fault: Fault | None) -> None:
        self.state = state
        self.rng = rng
        self.fault = fault
        self.steps: list[Step] = []
        self.counts: Counter[str] = Counter()
        self.fault_index: int | None = None
        self.fault_span = 1  # steps that make up the fault (a loop spans several)

    def active(self, name: str, kinds: tuple[str, ...]) -> str | None:
        """The fault kind if this occurrence of ``name`` is the one to corrupt."""
        f = self.fault
        if f and f.step == name and f.occurrence == self.counts[name] and f.kind in kinds:
            return f.kind
        return None

    def skip(self, name: str) -> bool:
        if self.active(name, ("skipped_step",)):
            self.counts[name] += 1
            self.fault_index = len(self.steps)
            return True
        return False

    def step(self, name: str, fn: Callable[..., dict[str, Any]], **inputs: Any) -> dict[str, Any]:
        fault = self.active(name, ("wrong_update", "missed_update", "duplicate", "wrong_argument"))
        if fault == "wrong_argument":
            inputs = WRONG_ARGUMENTS[name](inputs, self.state, self.rng)
        if fault is not None and self.fault_index is None:
            self.fault_index = len(self.steps)
        before = copy.deepcopy(self.state)
        outputs = fn(self.state, self.rng, fault, **inputs)
        self.steps.append(
            Step(name, dict(inputs), dict(outputs), before, copy.deepcopy(self.state))
        )
        self.counts[name] += 1
        return outputs

    def loop(
        self,
        steps: list[tuple[str, Callable[..., dict[str, Any]], dict[str, Any]]],
        repeats: int = 3,
    ) -> None:
        """Run ``steps`` again ``repeats`` times (an A, B, A, B loop)."""
        if self.fault_index is None:
            self.fault_index = len(self.steps)
        self.fault_span = repeats * len(steps)
        for _ in range(repeats):
            for name, fn, inputs in steps:
                before = copy.deepcopy(self.state)
                outputs = fn(self.state, self.rng, None, **inputs)
                self.steps.append(
                    Step(name, dict(inputs), dict(outputs), before, copy.deepcopy(self.state))
                )

    def looping(self, name: str) -> bool:
        """True right after the occurrence of ``name`` chosen for a loop fault."""
        f = self.fault
        return bool(
            f and f.kind == "loop" and f.step == name and self.counts[name] == f.occurrence + 1
        )


def _finish(run: Run, recorder: Recorder) -> tuple[Run, int | None]:
    if recorder.fault_index is not None:
        run.meta["fault_span"] = recorder.fault_span
    return run, recorder.fault_index


# ------------------------------------------------------------------- shop

CATALOG = [(f"sku{i:03d}", round(random.Random(i).uniform(2, 90), 2)) for i in range(60)]
WORDS = "lamp chair desk mug pen notebook cable charger shelf rug".split()


def _search(
    state: dict[str, Any], rng: random.Random, fault: str | None, query: str
) -> dict[str, Any]:
    state["queries"] += 1
    return {"results": rng.randint(0, 40)}


def _add_to_cart(
    state: dict[str, Any], rng: random.Random, fault: str | None, sku: str, price: float, qty: int
) -> dict[str, Any]:
    line = round(price * qty, 2)
    item = {"sku": sku, "price": price, "qty": qty, "line": line}
    state["adds"] += 1
    if fault == "missed_update":
        state["cart"]["items"].append(item)  # total not updated
        return {"line": line}
    times = 2 if fault == "duplicate" else 1
    for _ in range(times):
        state["cart"]["items"].append(dict(item))
        added = price if fault == "wrong_update" else line  # forgets the quantity
        state["cart"]["total"] = round(state["cart"]["total"] + added, 2)
    return {"line": line}


def _apply_coupon(
    state: dict[str, Any], rng: random.Random, fault: str | None, code: str, percent: int
) -> dict[str, Any]:
    state["coupon"] = code
    if fault != "missed_update":
        state["cart"]["total"] = round(state["cart"]["total"] * (1 - percent / 100), 2)
    return {"applied": True}


def _checkout(
    state: dict[str, Any], rng: random.Random, fault: str | None, amount: float, user_id: str
) -> dict[str, Any]:
    if fault == "missed_update":
        return {"charged": amount}
    state["order"]["paid"] = round(amount * (2 if fault == "duplicate" else 1), 2)
    state["order"]["status"] = "paid" if fault != "wrong_update" else "pending"
    return {"charged": amount}


def _send_receipt(
    state: dict[str, Any], rng: random.Random, fault: str | None, email: str, amount: float
) -> dict[str, Any]:
    state["receipts"] += 2 if fault == "duplicate" else 1
    state["receipt_to"] = email if fault != "wrong_update" else None
    return {"sent": True}


def shop(rng: random.Random, fault: Fault | None = None) -> tuple[Run, int | None]:
    uid = rng.randint(1000, 9999)
    state: dict[str, Any] = {
        "user": {"id": f"u{uid}", "email": f"user{uid}@example.com"},
        "cart": {"items": [], "total": 0.0},
        "coupon": None,
        "order": {"status": "open", "paid": 0.0},
        "queries": 0,
        "receipts": 0,
        "receipt_to": None,
        "adds": 0,
    }
    r = Recorder(state, rng, fault)
    for _ in range(rng.randint(1, 4)):
        for _ in range(rng.randint(1, 3)):
            r.step("search", _search, query=rng.choice(WORDS))
        sku, price = rng.choice(CATALOG)
        r.step("add_to_cart", _add_to_cart, sku=sku, price=price, qty=rng.randint(1, 3))
        if r.looping("add_to_cart"):
            r.loop(
                [
                    ("search", _search, {"query": "lamp"}),
                    ("add_to_cart", _add_to_cart, {"sku": sku, "price": price, "qty": 1}),
                ]
            )
    if rng.random() < 0.3:
        r.step("apply_coupon", _apply_coupon, code="SAVE10", percent=10)
    r.step("checkout", _checkout, amount=state["cart"]["total"], user_id=state["user"]["id"])
    if not r.skip("send_receipt"):
        r.step(
            "send_receipt",
            _send_receipt,
            email=state["user"]["email"],
            amount=state["order"]["paid"],
        )
    lines = round(sum(i["line"] for i in state["cart"]["items"]), 2)
    expected = round(lines * 0.9, 2) if state["coupon"] else lines
    passed = (
        abs(state["order"]["paid"] - expected) < 0.02
        and state["order"]["status"] == "paid"
        and state["receipt_to"] == state["user"]["email"]
        and state["receipts"] == 1
        and state["adds"] == len(state["cart"]["items"])
    )
    return _finish(Run(f"shop-{uid}-{rng.random():.6f}", r.steps, passed), r)


# ----------------------------------------------------------------- travel

CITIES = "BOS JFK SFO SEA ORD DEN ATL LAX MIA AUS".split()


def _search_flights(
    state: dict[str, Any], rng: random.Random, fault: str | None, origin: str, dest: str
) -> dict[str, Any]:
    state["search"] = {"page": 1, "pages": rng.randint(1, 4)}
    return {"results": rng.randint(5, 30)}


def _next_page(state: dict[str, Any], rng: random.Random, fault: str | None) -> dict[str, Any]:
    step = 2 if fault == "wrong_update" else 0 if fault == "missed_update" else 1
    state["search"]["page"] += step
    return {"results": rng.randint(5, 30)}


def _select_flight(
    state: dict[str, Any], rng: random.Random, fault: str | None, flight: str
) -> dict[str, Any]:
    price = round(rng.uniform(80, 900), 2)
    state["selected"] = {"flight": flight, "price": price}
    return {"price": price}


def _book(
    state: dict[str, Any], rng: random.Random, fault: str | None, flight: str, passengers: int
) -> dict[str, Any]:
    unit = state["selected"]["price"]
    count = passengers - 1 if fault == "wrong_update" else passengers
    booking = {
        "flight": flight,
        "passengers": passengers,
        "price": round(unit * count, 2),
        "status": "held",
    }
    if fault != "missed_update":
        state["bookings"].append(booking)
        if fault == "duplicate":
            state["bookings"].append(dict(booking))
        state["booking"] = booking
    return {"reference": f"R{rng.randint(10000, 99999)}"}


def _pay(
    state: dict[str, Any], rng: random.Random, fault: str | None, amount: float
) -> dict[str, Any]:
    if fault != "missed_update" and state.get("booking"):
        state["booking"]["status"] = "paid"
        state["charged"] = round(state["charged"] + amount * (2 if fault == "duplicate" else 1), 2)
    return {"ok": True}


def _confirm(
    state: dict[str, Any], rng: random.Random, fault: str | None, email: str
) -> dict[str, Any]:
    state["confirmed_to"] = email
    return {"sent": True}


def travel(rng: random.Random, fault: Fault | None = None) -> tuple[Run, int | None]:
    uid = rng.randint(1000, 9999)
    origin, dest = rng.sample(CITIES, 2)
    state: dict[str, Any] = {
        "user": {"id": f"u{uid}", "email": f"traveller{uid}@example.com"},
        "trip": {"origin": origin, "dest": dest, "passengers": rng.randint(1, 4)},
        "search": {"page": 0, "pages": 0},
        "selected": None,
        "booking": None,
        "bookings": [],
        "charged": 0.0,
        "confirmed_to": None,
    }
    r = Recorder(state, rng, fault)
    r.step("search_flights", _search_flights, origin=origin, dest=dest)
    for _ in range(rng.randint(0, state["search"]["pages"] - 1)):
        r.step("next_page", _next_page)
    if r.looping("search_flights") or r.looping("next_page"):
        r.loop(
            [
                ("search_flights", _search_flights, {"origin": origin, "dest": dest}),
                ("next_page", _next_page, {}),
            ]
        )
    flight = f"{origin}{dest}{rng.randint(100, 999)}"
    r.step("select_flight", _select_flight, flight=flight)
    r.step("book", _book, flight=flight, passengers=state["trip"]["passengers"])
    booking = state.get("booking") or {"price": 0.0}
    r.step("pay", _pay, amount=booking["price"])
    if not r.skip("confirm"):
        r.step("confirm", _confirm, email=state["user"]["email"])
    expected = round(state["selected"]["price"] * state["trip"]["passengers"], 2)
    passed = (
        state["booking"] is not None
        and len(state["bookings"]) == 1
        and abs(state["booking"]["price"] - expected) < 0.02
        and state["booking"]["status"] == "paid"
        and abs(state["charged"] - expected) < 0.02
        and state["booking"]["passengers"] == state["trip"]["passengers"]
        and state["confirmed_to"] == state["user"]["email"]
    )
    return _finish(Run(f"travel-{uid}-{rng.random():.6f}", r.steps, passed), r)


# ---------------------------------------------------------------- support

CATEGORIES = ("refund", "question", "bug")


def _read_ticket(state: dict[str, Any], rng: random.Random, fault: str | None) -> dict[str, Any]:
    return {"chars": rng.randint(40, 900)}


def _classify(state: dict[str, Any], rng: random.Random, fault: str | None) -> dict[str, Any]:
    state["category"] = state["ticket"]["kind"]
    return {"category": state["category"]}


def _lookup_order(
    state: dict[str, Any], rng: random.Random, fault: str | None, order_id: str
) -> dict[str, Any]:
    amount = state["orders"].get(order_id, round(rng.uniform(5, 300), 2))
    state["order"] = {"id": order_id, "amount": amount}
    return {"amount": amount}


def _refund(
    state: dict[str, Any], rng: random.Random, fault: str | None, order_id: str, amount: float
) -> dict[str, Any]:
    if fault == "missed_update":
        return {"ok": True}
    paid = amount * 10 if fault == "wrong_update" else amount
    times = 2 if fault == "duplicate" else 1
    for _ in range(times):
        state["account"]["balance"] = round(state["account"]["balance"] + paid, 2)
        state["account"]["refunds"] += 1
    return {"ok": True}


def _answer(state: dict[str, Any], rng: random.Random, fault: str | None) -> dict[str, Any]:
    state["replies"] += 1
    return {"words": rng.randint(30, 200)}


def _file_bug(
    state: dict[str, Any], rng: random.Random, fault: str | None, severity: str
) -> dict[str, Any]:
    state["bugs"] += 1
    return {"issue": rng.randint(1000, 9999)}


def _close(state: dict[str, Any], rng: random.Random, fault: str | None) -> dict[str, Any]:
    if fault != "missed_update":
        state["ticket"]["status"] = "closed"
    return {"ok": True}


def support(rng: random.Random, fault: Fault | None = None) -> tuple[Run, int | None]:
    tid = rng.randint(10000, 99999)
    order_id = f"o{rng.randint(100000, 999999)}"
    amount = round(rng.uniform(5, 300), 2)
    kind = rng.choices(CATEGORIES, (0.5, 0.3, 0.2))[0]
    if fault and fault.step in ("lookup_order", "refund"):
        kind = "refund"
    state: dict[str, Any] = {
        "ticket": {"id": tid, "order_id": order_id, "kind": kind, "status": "open"},
        "orders": {order_id: amount},
        "category": None,
        "order": None,
        "account": {"balance": round(rng.uniform(0, 500), 2), "refunds": 0},
        "replies": 0,
        "bugs": 0,
    }
    start_balance = state["account"]["balance"]
    r = Recorder(state, rng, fault)
    r.step("read_ticket", _read_ticket)
    r.step("classify", _classify)
    if kind == "refund":
        if not r.skip("lookup_order"):
            r.step("lookup_order", _lookup_order, order_id=state["ticket"]["order_id"])
        refund_amount = (state["order"] or {"amount": round(rng.uniform(5, 300), 2)})["amount"]
        r.step("refund", _refund, order_id=order_id, amount=refund_amount)
        if r.looping("refund"):
            r.loop(
                [
                    ("lookup_order", _lookup_order, {"order_id": order_id}),
                    ("refund", _refund, {"order_id": order_id, "amount": refund_amount}),
                ]
            )
    elif kind == "question":
        for _ in range(rng.randint(1, 3)):
            r.step("answer", _answer)
    else:
        r.step("file_bug", _file_bug, severity=rng.choice(["low", "medium", "high"]))
        r.step("answer", _answer)
    r.step("close_ticket", _close)
    passed = state["ticket"]["status"] == "closed"
    if kind == "refund":
        passed = (
            passed
            and abs(state["account"]["balance"] - (start_balance + amount)) < 0.02
            and state["account"]["refunds"] == 1
        )
    return _finish(Run(f"support-{tid}-{rng.random():.6f}", r.steps, passed), r)


# --------------------------------------------------------------- research


def _plan(state: dict[str, Any], rng: random.Random, fault: str | None) -> dict[str, Any]:
    state["planned"] = True
    return {"subquestions": rng.randint(2, 5)}


def _web_search(
    state: dict[str, Any], rng: random.Random, fault: str | None, query: str
) -> dict[str, Any]:
    state["searches"] += 1
    return {"hits": rng.randint(1, 10)}


def _read(state: dict[str, Any], rng: random.Random, fault: str | None, url: str) -> dict[str, Any]:
    state["read"] += 1
    return {"chars": rng.randint(500, 9000)}


def _note(state: dict[str, Any], rng: random.Random, fault: str | None, url: str) -> dict[str, Any]:
    times = 0 if fault == "missed_update" else 2 if fault == "duplicate" else 1
    for _ in range(times):
        state["notes"].append({"url": url, "words": rng.randint(20, 120)})
    if fault == "wrong_update":
        state["notes"] = [*state["notes"][:-1], {"url": "", "words": 0}]
    return {"notes": len(state["notes"])}


def _write(
    state: dict[str, Any], rng: random.Random, fault: str | None, citations: int
) -> dict[str, Any]:
    state["answer"] = {"citations": citations, "words": rng.randint(150, 600)}
    return {"ok": True}


def research(rng: random.Random, fault: Fault | None = None) -> tuple[Run, int | None]:
    rid = rng.randint(10000, 99999)
    state: dict[str, Any] = {
        "planned": False,
        "searches": 0,
        "read": 0,
        "notes": [],
        "answer": None,
    }
    r = Recorder(state, rng, fault)
    r.step("plan", _plan)
    for i in range(rng.randint(2, 5)):
        url = f"https://source{rid}-{i}.example.org/page"
        r.step("web_search", _web_search, query=f"subquestion {i}")
        r.step("read", _read, url=url)
        if r.looping("read"):
            r.loop(
                [
                    ("web_search", _web_search, {"query": f"subquestion {i} again"}),
                    ("read", _read, {"url": url}),
                ],
                repeats=4,
            )
        if not r.skip("note"):
            r.step("note", _note, url=url)
    r.step("write_answer", _write, citations=len(state["notes"]))
    urls = [n["url"] for n in state["notes"]]
    passed = (
        state["answer"] is not None
        and state["answer"]["citations"] == len(state["notes"])
        and len(urls) == len(set(urls))
        and all(urls)
        and len(state["notes"]) == state["read"]
    )
    return _finish(Run(f"research-{rid}-{rng.random():.6f}", r.steps, passed), r)


# ---------------------------------------------------------- wrong arguments


def _other_email(
    inputs: dict[str, Any], state: dict[str, Any], rng: random.Random
) -> dict[str, Any]:
    return {**inputs, "email": f"someone{rng.randint(1000, 9999)}@example.com"}


WRONG_ARGUMENTS: dict[
    str, Callable[[dict[str, Any], dict[str, Any], random.Random], dict[str, Any]]
] = {
    "send_receipt": _other_email,
    "confirm": _other_email,
    "checkout": lambda i, s, r: {
        **i,
        "amount": round(i["amount"] + r.choice([-1, 1]) * r.uniform(1, 20), 2),
    },
    "book": lambda i, s, r: {**i, "passengers": i["passengers"] + 1},
    "lookup_order": lambda i, s, r: {**i, "order_id": f"o{r.randint(100000, 999999)}"},
    "refund": lambda i, s, r: {**i, "amount": round(i["amount"] * r.uniform(1.5, 3), 2)},
    "write_answer": lambda i, s, r: {**i, "citations": i["citations"] + r.randint(1, 3)},
    "add_to_cart": lambda i, s, r: {**i, "qty": i["qty"] + 5},
}

WORKFLOWS: dict[str, Callable[[random.Random, Fault | None], tuple[Run, int | None]]] = {
    "shop": shop,
    "travel": travel,
    "support": support,
    "research": research,
}

# Which faults make sense at which steps, per workflow.
TARGETS: dict[str, dict[str, tuple[str, ...]]] = {
    "shop": {
        "wrong_update": ("add_to_cart", "checkout", "send_receipt"),
        "missed_update": ("add_to_cart", "apply_coupon", "checkout"),
        "duplicate": ("add_to_cart", "checkout", "send_receipt"),
        "wrong_argument": ("checkout", "send_receipt", "add_to_cart"),
        "loop": ("add_to_cart",),
        "skipped_step": ("send_receipt",),
    },
    "travel": {
        "wrong_update": ("next_page", "book"),
        "missed_update": ("next_page", "book", "pay"),
        "duplicate": ("book", "pay"),
        "wrong_argument": ("book", "confirm"),
        "loop": ("search_flights",),
        "skipped_step": ("confirm",),
    },
    "support": {
        "wrong_update": ("refund",),
        "missed_update": ("refund", "close_ticket"),
        "duplicate": ("refund",),
        "wrong_argument": ("lookup_order", "refund"),
        "loop": ("refund",),
        "skipped_step": ("lookup_order",),
    },
    "research": {
        "wrong_update": ("note",),
        "missed_update": ("note",),
        "duplicate": ("note",),
        "wrong_argument": ("write_answer",),
        "loop": ("read",),
        "skipped_step": ("note",),
    },
}


def good_runs(workflow: str, n: int, seed: int = 0) -> list[Run]:
    rng = random.Random(seed)
    fn = WORKFLOWS[workflow]
    return [fn(rng, None)[0] for _ in range(n)]


def faulty_runs(workflow: str, n: int, seed: int = 0) -> list[tuple[Run, int, Fault]]:
    """``n`` runs with one planted fault each: (run, index of the first faulty step, fault)."""
    rng = random.Random(seed)
    fn = WORKFLOWS[workflow]
    out: list[tuple[Run, int, Fault]] = []
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        kind = rng.choice(FAULTS)
        step = rng.choice(TARGETS[workflow][kind])
        fault = Fault(kind, step, rng.choice([0, 0, 1]))
        run, index = fn(rng, fault)
        if index is None:  # the targeted occurrence never ran in this run
            continue
        run.meta.update({"fault": kind, "fault_step": step, "fault_index": index})
        out.append((run, index, fault))
    return out
