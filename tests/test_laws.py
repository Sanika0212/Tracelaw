import math
import random
from pathlib import Path

import pytest

from tracelaw import Law, Lawbook, MiningOptions, Run, Step, load_runs, save_runs
from tracelaw.flatten import kind, variables
from tracelaw.lawbook import conformal_threshold
from tracelaw.laws import mine_step
from tracelaw.trajectory import Trajectory


def cart_step(rng: random.Random, *, bad: str | None = None) -> Step:
    items = [
        {"sku": f"s{rng.randint(1, 50)}", "price": round(rng.uniform(1, 50), 2), "tag": "new"}
        for _ in range(rng.randint(0, 4))
    ]
    total = round(sum(i["price"] for i in items), 2)
    price = round(rng.uniform(1, 50), 2)
    qty = rng.randint(2, 3) if bad == "total" else rng.randint(1, 3)  # qty 1 hides the bug
    line = round(price * qty, 2)
    after_items = [*items, {"sku": "s99", "price": line, "tag": "new"}]
    after_total = round(total + line, 2)
    count = rng.randint(0, 9)
    user = f"u{rng.randint(100, 999)}"
    if bad == "total":
        after_total = round(total + price, 2)
    count_after = count + 2 if bad == "count" else count + 1
    return Step(
        "add",
        inputs={"price": price, "qty": qty, "user": user, "mode": rng.choice(["web", "app"])},
        outputs={"line": line},
        state_before={"cart": {"items": items, "total": total}, "adds": count, "user": user},
        state_after={
            "cart": {"items": after_items, "total": after_total},
            "adds": count_after,
            "user": "someone" if bad == "user" else user,
        },
    )


@pytest.fixture
def laws() -> list[Law]:
    rng = random.Random(0)
    return mine_step("add", [variables(cart_step(rng)) for _ in range(60)], MiningOptions())


def described(laws: list[Law]) -> set[str]:
    return {law.describe() for law in laws}


def test_flatten_variables() -> None:
    step = Step(
        "s",
        inputs={"a": 1, "nested": {"b": [1, 2]}},
        outputs={"ok": True},
        state_before={"items": [{"p": 1.5}, {"p": 2}], "tags": ["x", "y"]},
        state_after=None,
        error="boom",
    )
    v = variables(step)
    assert v["in.a"] == 1
    assert v["in.nested.b#len"] == 2
    assert v["in.nested.b[]"] == [1, 2]
    assert v["pre.items[].p"] == [1.5, 2]
    assert v["pre.tags[]"] == ["x", "y"]
    assert v["out.ok"] is True
    assert v["#error"] is True
    assert [kind(x) for x in (None, True, 3, 2.5, "s", [1])] == [
        "null",
        "bool",
        "number",
        "number",
        "str",
        "list",
    ]


def test_mined_laws_read_like_the_workflow(laws: list[Law]) -> None:
    text = described(laws)
    assert "add: post.cart.total equals the sum of post.cart.items[].price" in text
    assert "add: adds changes by exactly 1" in text
    assert "add: user is left unchanged" in text
    assert "add: post.cart.items#len equals pre.cart.items#len + in.qty" not in text
    assert "add: out.line equals in.price × in.qty" in text
    assert "add: post.cart.total equals pre.cart.total + out.line" in text
    assert "add: in.user equals pre.user" in text
    assert "add: in.mode is one of ['app', 'web']" in text
    assert "add: every element of post.cart.items[].tag is one of ['new']" in text
    kinds = {law.kind for law in laws}
    assert {"present", "kind", "range", "one_of", "unchanged", "delta", "equal"} <= kinds
    assert {"sum_of", "product", "plus", "each_one_of", "each_range", "each_kind"} <= kinds
    assert all(law.support >= 5 for law in laws)


@pytest.mark.parametrize(
    ("bad", "expected"),
    [
        ("total", "post.cart.total"),
        ("count", "adds changed by 2"),
        ("user", "user changed from"),
    ],
)
def test_violations_name_what_went_wrong(laws: list[Law], bad: str, expected: str) -> None:
    values = variables(cart_step(random.Random(5), bad=bad))
    problems = [p for law in laws if (p := law.check(values))]
    assert any(expected in p for p in problems)
    clean = variables(cart_step(random.Random(5)))
    assert [law.describe() for law in laws if law.check(clean)] == []


def test_law_round_trip_and_strength(laws: list[Law]) -> None:
    law = laws[0]
    again = Law.from_dict(law.to_dict())
    assert again == law
    assert law.strength == pytest.approx(math.log1p(law.support))


def test_few_observations_give_no_laws() -> None:
    rng = random.Random(1)
    assert mine_step("add", [variables(cart_step(rng)) for _ in range(3)], MiningOptions()) == []


def test_trajectory_scores_loops_and_early_stops() -> None:
    good = [["a", "b", "c"]] * 30 + [["a", "b", "b", "c"]] * 10
    t = Trajectory(order=2).fit(good)
    normal = t.surprisal(["a", "b", "c"])
    assert max(normal) < 1.0
    looped = t.surprisal(["a", "b", "c", "b", "c", "b", "c"])
    assert max(looped) > 2.0
    early = t.surprisal(["a", "b"])
    assert early[-1] > 2.0  # ending after b is rare
    unknown = t.surprisal(["a", "zzz", "c"])
    assert unknown[1] > 3.0
    assert t.occurrences["b"] == (1, 2)
    again = Trajectory.from_dict(t.to_dict())
    assert again.surprisal(["a", "b", "c"]) == pytest.approx(normal)


def test_conformal_threshold() -> None:
    scores = list(range(1, 20))  # 19 calibration scores
    t = conformal_threshold(scores, 0.1)
    assert sum(s >= t for s in scores) / len(scores) <= 0.1
    assert conformal_threshold([1.0, 2.0], 0.01) == math.inf


def runs(n: int, seed: int, bad: str | None = None) -> list[Run]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        steps = [Step("start", outputs={"ok": True}), cart_step(rng)]
        if bad and i % 2 == 0:
            steps[1] = cart_step(rng, bad=bad)
        steps.append(Step("finish", outputs={"ok": True}))
        out.append(Run(f"r{seed}-{i}", steps, passed=None))
    return out


def test_lawbook_learn_check_and_store(tmp_path: Path) -> None:
    book = Lawbook.learn(runs(120, 0), alpha=0.1, folds=4)
    assert book.folds
    assert len(book.fold_scores) == 4
    good = [book.check(r) for r in runs(60, 1)]
    assert sum(v.flagged for v in good) / len(good) <= 0.2
    bad = [book.check(r) for r in runs(20, 2, bad="total")[::2]]
    assert all(v.flagged for v in bad)
    assert all(v.first_step == 1 for v in bad)
    assert any("post.cart.total" in f.message for f in bad[0].culprit)
    assert bad[0].p_value <= 0.1
    assert book.localize(runs(2, 3, bad="count")[0]) == 1
    path = book.save(tmp_path / "laws.json")
    loaded = Lawbook.load(path)
    assert loaded.check(runs(2, 2, bad="total")[0]).flagged
    assert loaded.to_dict()["threshold"] == book.to_dict()["threshold"]
    with pytest.raises(ValueError, match="good runs"):
        Lawbook.learn(runs(5, 0))


def test_run_shapes_and_storage(tmp_path: Path) -> None:
    run = Run.from_states("x", [{"n": 0}, {"n": 1}, {"n": 2}], ["inc", "inc"], passed=True)
    assert run.names == ["inc", "inc"]
    assert run.steps[1].state_before == {"n": 1}
    with pytest.raises(ValueError, match="one state before"):
        Run.from_states("x", [{"n": 0}], ["inc"])
    path = save_runs([run], tmp_path / "runs.jsonl")
    assert load_runs(path)[0].steps[1].state_after == {"n": 2}
    listed = tmp_path / "runs.json"
    listed.write_text("[" + path.read_text().strip() + "]")
    assert load_runs(listed)[0].passed is True
    assert Step.from_dict({"name": "s", "outputs": 3}).outputs == {"value": 3}
    assert len(run) == 2
