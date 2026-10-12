"""Mine the laws that every good observation of a step obeyed (Daikon-style inference).

For each step name, every good run's occurrences of that step are flattened into
variables (:mod:`tracelaw.flatten`) and candidate laws are kept when they held in every
observation and were seen at least ``min_support`` times:

==============  ====================================================================
law             holds when
==============  ====================================================================
``present``     the variable always exists (a missing one is a missed update)
``kind``        the variable's type is always one of a few (number, str, bool, null)
``range``       a number stays within the range seen in good runs
``one_of``      a string or flag takes one of a few values
``unchanged``   the step never changes this part of the state (``post == pre``)
``delta``       the step always moves a number by the same amount (``post == pre + c``)
``monotone``    the step never decreases (or never increases) a number
``equal``       two varying variables are always equal (``in.user_id == pre.user.id``)
``sum_of``      a number always equals the sum of a list's values (a total and its items)
``product``     a number always equals the product of two others (``price == unit * qty``)
``plus``        a number always equals the sum of two others (``post.balance == pre.balance +
                in.amount``)
``each_*``      every element of a list has a kind, a range, a value from a few, or is not
                empty (``post.notes[].url`` is never empty)
``member``      an argument is always one of the values the state already holds
                (``in.order_id`` is among the ids seen so far: no invented ids)
``subset``      every element of a list argument is among them (``in.item_ids[]``)
==============  ====================================================================

Every law keeps its *support* (how many observations it held in). A violated law's
strength is ``log(support + 1)``: breaking a law that held 500 times is more surprising
than breaking one that held 5 times. Run-level false-alarm rates are calibrated
separately, on held-out good runs (:mod:`tracelaw.lawbook`).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from tracelaw.flatten import is_number, kind

TOLERANCE = 1e-6


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= TOLERANCE * max(1.0, abs(a), abs(b))


# how interesting a kind of law is to read first: relations, then changes, then values
PRIORITY = {
    "equal": 0, "sum_of": 0, "plus": 0, "product": 0, "member": 0, "subset": 0,
    "delta": 1, "unchanged": 1, "monotone": 1,
    "one_of": 2, "range": 2, "present": 3, "kind": 4,
    "each_one_of": 3, "each_range": 3, "each_nonempty": 3, "each_kind": 4,
}  # fmt: skip


@dataclass
class Law:
    step: str
    kind: str
    vars: tuple[str, ...]
    params: dict[str, Any] = field(default_factory=dict)
    support: int = 0

    @property
    def strength(self) -> float:
        """How surprising a violation is: log(1 + observations the law held in)."""
        return math.log1p(self.support)

    def describe(self) -> str:
        v = self.vars
        p = self.params
        text = {
            "present": f"{v[0]} is always set",
            "kind": f"{v[0]} is always {' or '.join(p.get('kinds', []))}",
            "range": f"{v[0]} stays within [{_fmt(p.get('lo'))}, {_fmt(p.get('hi'))}]",
            "one_of": f"{v[0]} is one of {p.get('values')}",
            "unchanged": f"{v[0]} is left unchanged",
            "delta": f"{v[0]} changes by exactly {_fmt(p.get('c'))}",
            "monotone": f"{v[0]} never {'decreases' if p.get('up') else 'increases'}",
            "equal": f"{v[0]} equals {v[1] if len(v) > 1 else ''}",
            "sum_of": f"{v[0]} equals the sum of {v[1] if len(v) > 1 else ''}",
            "product": f"{v[0]} equals {v[1]} × {v[2]}" if len(v) > 2 else "",
            "plus": f"{v[0]} equals {v[1]} + {v[2]}" if len(v) > 2 else "",
            "each_kind": f"every element of {v[0]} is {' or '.join(p.get('kinds', []))}",
            "each_range": f"every element of {v[0]} is within "
            f"[{_fmt(p.get('lo'))}, {_fmt(p.get('hi'))}]",
            "each_nonempty": f"no element of {v[0]} is empty",
            "each_one_of": f"every element of {v[0]} is one of {p.get('values')}",
            "member": f"{v[0]} is one of {v[1] if len(v) > 1 else ''}",
            "subset": f"every element of {v[0]} is in {v[1] if len(v) > 1 else ''}",
        }[self.kind]
        return f"{self.step}: {text}"

    def check(self, values: dict[str, Any]) -> str | None:
        """``None`` if the law holds (or does not apply); otherwise what went wrong."""
        return _CHECKS[self.kind](self, values)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["vars"] = list(self.vars)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Law:
        return cls(
            data["step"], data["kind"], tuple(data["vars"]), dict(data["params"]), data["support"]
        )


def _num(value: Any) -> float | None:
    return float(value) if is_number(value) else None


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return repr(value) if isinstance(value, str) else str(value)


def _post_pre(path: str) -> tuple[str, str]:
    return f"post.{path}", f"pre.{path}"


# --------------------------------------------------------------------- checks


def _check_present(law: Law, values: dict[str, Any]) -> str | None:
    return None if law.vars[0] in values else f"{law.vars[0]} is missing"


def _check_kind(law: Law, values: dict[str, Any]) -> str | None:
    if law.vars[0] not in values:
        return None
    k = kind(values[law.vars[0]])
    return (
        None if k in law.params["kinds"] else f"{law.vars[0]} is {k}: {_fmt(values[law.vars[0]])}"
    )


def _check_range(law: Law, values: dict[str, Any]) -> str | None:
    value = _num(values.get(law.vars[0]))
    if value is None:
        return None
    lo, hi = law.params["lo"], law.params["hi"]
    if value < lo - TOLERANCE * max(1, abs(lo)) or value > hi + TOLERANCE * max(1, abs(hi)):
        return f"{law.vars[0]} = {_fmt(value)}, outside [{_fmt(lo)}, {_fmt(hi)}]"
    return None


def _check_one_of(law: Law, values: dict[str, Any]) -> str | None:
    if law.vars[0] not in values:
        return None
    value = values[law.vars[0]]
    return None if value in law.params["values"] else f"{law.vars[0]} = {_fmt(value)}"


def _check_unchanged(law: Law, values: dict[str, Any]) -> str | None:
    post, pre = _post_pre(law.vars[0])
    if post not in values or pre not in values:
        return None
    if values[post] == values[pre] or (
        is_number(values[post]) and is_number(values[pre]) and _close(values[post], values[pre])
    ):
        return None
    return f"{law.vars[0]} changed from {_fmt(values[pre])} to {_fmt(values[post])}"


def _check_delta(law: Law, values: dict[str, Any]) -> str | None:
    post, pre = _post_pre(law.vars[0])
    a, b = _num(values.get(post)), _num(values.get(pre))
    if a is None or b is None:
        return None
    c = law.params["c"]
    if _close(a - b, c):
        return None
    return f"{law.vars[0]} changed by {_fmt(a - b)} ({_fmt(b)} to {_fmt(a)}), not {_fmt(c)}"


def _check_monotone(law: Law, values: dict[str, Any]) -> str | None:
    post, pre = _post_pre(law.vars[0])
    a, b = _num(values.get(post)), _num(values.get(pre))
    if a is None or b is None:
        return None
    if law.params["up"] and a < b - TOLERANCE * max(1, abs(b)):
        return f"{law.vars[0]} decreased from {_fmt(b)} to {_fmt(a)}"
    if not law.params["up"] and a > b + TOLERANCE * max(1, abs(b)):
        return f"{law.vars[0]} increased from {_fmt(b)} to {_fmt(a)}"
    return None


def _check_equal(law: Law, values: dict[str, Any]) -> str | None:
    a, b = law.vars
    if a not in values or b not in values:
        return None
    x, y = values[a], values[b]
    if x == y or (is_number(x) and is_number(y) and _close(x, y)):
        return None
    return f"{a} = {_fmt(x)} but {b} = {_fmt(y)}"


def _check_sum_of(law: Law, values: dict[str, Any]) -> str | None:
    total, parts = law.vars
    t, items = _num(values.get(total)), values.get(parts)
    if t is None or not isinstance(items, list) or not all(map(is_number, items)):
        return None
    s = float(sum(items))
    return None if _close(t, s) else f"{total} = {_fmt(t)} but the sum of {parts} is {_fmt(s)}"


def _check_product(law: Law, values: dict[str, Any]) -> str | None:
    t, a, b = (_num(values.get(v)) for v in law.vars)
    if t is None or a is None or b is None:
        return None
    return (
        None
        if _close(t, a * b)
        else f"{law.vars[0]} = {_fmt(t)} but {law.vars[1]} × {law.vars[2]} = {_fmt(a * b)}"
    )


def _elements(law: Law, values: dict[str, Any]) -> list[Any]:
    items = values.get(law.vars[0])
    return items if isinstance(items, list) else []


def _check_each_kind(law: Law, values: dict[str, Any]) -> str | None:
    bad = [e for e in _elements(law, values) if kind(e) not in law.params["kinds"]]
    return f"{law.vars[0]} has an element {_fmt(bad[0])}" if bad else None


def _check_each_range(law: Law, values: dict[str, Any]) -> str | None:
    lo, hi = law.params["lo"], law.params["hi"]
    nums = [x for x in map(_num, _elements(law, values)) if x is not None]
    low, high = lo - TOLERANCE * max(1, abs(lo)), hi + TOLERANCE * max(1, abs(hi))
    bad = [x for x in nums if x < low or x > high]
    return f"{law.vars[0]} has {_fmt(bad[0])}, outside [{_fmt(lo)}, {_fmt(hi)}]" if bad else None


def _check_each_nonempty(law: Law, values: dict[str, Any]) -> str | None:
    items = _elements(law, values)
    return f"{law.vars[0]} has an empty element" if items and not all(items) else None


def _check_each_one_of(law: Law, values: dict[str, Any]) -> str | None:
    bad = [e for e in _elements(law, values) if e not in law.params["values"]]
    return f"{law.vars[0]} has {_fmt(bad[0])}" if bad else None


def _check_member(law: Law, values: dict[str, Any]) -> str | None:
    a, pool = law.vars
    if a not in values or not isinstance(values.get(pool), list):
        return None
    value = values[a]
    return None if value in values[pool] else f"{a} = {_fmt(value)} was never seen in {pool}"


def _check_subset(law: Law, values: dict[str, Any]) -> str | None:
    a, pool = law.vars
    items, known = values.get(a), values.get(pool)
    if not isinstance(items, list) or not isinstance(known, list):
        return None
    missing = [x for x in items if x not in known]
    return f"{a} has {_fmt(missing[0])}, never seen in {pool}" if missing else None


def _check_plus(law: Law, values: dict[str, Any]) -> str | None:
    t, a, b = (_num(values.get(v)) for v in law.vars)
    if t is None or a is None or b is None:
        return None
    return (
        None
        if _close(t, a + b)
        else f"{law.vars[0]} = {_fmt(t)} but {law.vars[1]} + {law.vars[2]} = {_fmt(a + b)}"
    )


_CHECKS = {
    "present": _check_present,
    "kind": _check_kind,
    "range": _check_range,
    "one_of": _check_one_of,
    "unchanged": _check_unchanged,
    "delta": _check_delta,
    "monotone": _check_monotone,
    "equal": _check_equal,
    "sum_of": _check_sum_of,
    "product": _check_product,
    "plus": _check_plus,
    "each_kind": _check_each_kind,
    "each_range": _check_each_range,
    "each_nonempty": _check_each_nonempty,
    "each_one_of": _check_each_one_of,
    "member": _check_member,
    "subset": _check_subset,
}


# --------------------------------------------------------------------- mining


@dataclass
class MiningOptions:
    min_support: int = 5  # observations a law needs before it is kept
    min_range_support: int = 10  # ranges are loose with fewer
    range_margin: float = 0.5  # widen ranges by this share of the observed spread, each side
    max_enum: int = 6  # distinct values for one_of
    enum_ratio: int = 3  # each distinct value seen this many times on average
    max_numeric: int = 60  # cap on variables considered for pairwise relations


def _hashable(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def mine_step(step: str, observations: Sequence[dict[str, Any]], opts: MiningOptions) -> list[Law]:
    """Laws that held in every observation of one step type."""
    n = len(observations)
    if n < opts.min_support:
        return []
    laws: list[Law] = []
    paths = sorted(set().union(*(o.keys() for o in observations)))
    columns: dict[str, list[Any]] = {p: [o[p] for o in observations if p in o] for p in paths}
    aliases = _aliases(columns, n)

    for path, column in columns.items():
        if path == "#error":
            if not any(column):
                laws.append(Law(step, "one_of", (path,), {"values": [False]}, n))
            continue
        if len(column) < opts.min_support:
            continue  # e.g. a key that is an id, different in every run
        if len(column) == n:
            laws.append(Law(step, "present", (path,), {}, n))
        kinds = sorted({kind(v) for v in column})
        if kinds == ["list"]:
            laws.extend(_element_laws(step, path, column, opts))
            continue
        if "list" in kinds:
            continue
        if len(kinds) <= 2:
            laws.append(Law(step, "kind", (path,), {"kinds": kinds}, len(column)))
        if path in aliases:
            continue  # value laws live on the alias group's representative
        if kinds == ["number"] and len(column) >= opts.min_range_support:
            laws.append(
                Law(step, "range", (path,), _fences(column, opts.range_margin), len(column))
            )
        if kinds in (["str"], ["bool"], ["bool", "null"], ["null", "str"]):
            distinct = Counter(column)
            # every allowed value seen twice or more: values seen once (Good-Turing) mean an
            # open-ended field, where the next value is likely to be new
            if (
                len(distinct) <= opts.max_enum
                and len(column) >= opts.enum_ratio * len(distinct)
                and min(distinct.values()) >= 2
            ):
                laws.append(
                    Law(step, "one_of", (path,), {"values": sorted(distinct, key=str)}, len(column))
                )

    laws.extend(_transitions(step, observations, opts))
    laws.extend(_relations(step, observations, columns, opts))
    laws.extend(_memberships(step, columns, observations, opts))
    return laws


def _memberships(
    step: str,
    columns: dict[str, list[Any]],
    observations: Sequence[dict[str, Any]],
    opts: MiningOptions,
) -> list[Law]:
    """Arguments drawn from what the state already holds (``in.order_id in pre.ids[]``)."""
    pools = [p for p, c in columns.items() if p.startswith("pre.") and p.endswith("[]")]
    if not pools:
        return []
    out: list[Law] = []
    for arg, column in columns.items():
        if not arg.startswith("in.") or len(column) < opts.min_support:
            continue
        is_list = all(isinstance(v, list) for v in column)
        if not is_list and not all(isinstance(v, str) for v in column):
            continue
        if len({_hashable(v) for v in column}) < 2:
            continue  # a constant argument says nothing about where it came from
        for pool in pools:
            pairs = [(o[arg], o[pool]) for o in observations if arg in o and pool in o]
            if len(pairs) < opts.min_support:
                continue
            if is_list and all(all(x in known for x in v) for v, known in pairs):
                out.append(Law(step, "subset", (arg, pool), {}, len(pairs)))
            elif not is_list and all(v in known for v, known in pairs):
                out.append(Law(step, "member", (arg, pool), {}, len(pairs)))
    return out


def _fences(values: Sequence[float], margin: float) -> dict[str, float]:
    """The observed range widened by ``margin`` of its spread on each side.

    A new good value falls outside the plain [min, max] of ``n`` values with probability
    about 2/(n + 1), which across many variables is a steady source of false alarms;
    the fences keep range laws for values that are far off (a refund ten times too large).
    """
    lo, hi = float(min(values)), float(max(values))
    pad = margin * (hi - lo)
    # a quantity never seen below zero stays non-negative: a negative price is a bug
    return {"lo": lo - pad if lo < 0 else max(lo - pad, 0.0), "hi": hi + pad}


def _aliases(columns: dict[str, list[Any]], n: int) -> set[str]:
    """Variables equal to another one in every observation (all but one per group)."""
    groups: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for path, column in columns.items():
        if len(column) == n and path != "#error":
            groups[tuple(_hashable(v) for v in column)].append(path)
    return {p for members in groups.values() for p in sorted(members, key=_rank)[1:]}


def _element_laws(step: str, path: str, column: list[Any], opts: MiningOptions) -> list[Law]:
    """Laws every element of a list obeys (``post.notes[].url`` is never empty)."""
    elements = [e for value in column for e in value]
    if len(elements) < opts.min_support:
        return []
    out: list[Law] = []
    kinds = sorted({kind(e) for e in elements})
    if len(kinds) <= 2:
        out.append(Law(step, "each_kind", (path,), {"kinds": kinds}, len(elements)))
    if kinds == ["number"] and len(elements) >= opts.min_range_support:
        bounds = _fences(elements, opts.range_margin)
        out.append(Law(step, "each_range", (path,), bounds, len(elements)))
    if kinds == ["str"] and all(elements):
        out.append(Law(step, "each_nonempty", (path,), {}, len(elements)))
    if kinds in (["str"], ["bool"]):
        distinct = Counter(elements)
        if len(distinct) <= opts.max_enum and len(elements) >= opts.enum_ratio * len(distinct):
            allowed = {"values": sorted(distinct, key=str)}
            out.append(Law(step, "each_one_of", (path,), allowed, len(elements)))
    return out


def _transitions(
    step: str, observations: Sequence[dict[str, Any]], opts: MiningOptions
) -> list[Law]:
    """Laws about how the step changes the state: unchanged, delta, monotone."""
    out: list[Law] = []
    suffixes = sorted(
        {p[5:] for o in observations for p in o if p.startswith("post.")}
        & {p[4:] for o in observations for p in o if p.startswith("pre.")}
    )
    for suffix in suffixes:
        post, pre = _post_pre(suffix)
        pairs = [(o[post], o[pre]) for o in observations if post in o and pre in o]
        if len(pairs) < opts.min_support:
            continue
        if all(_hashable(a) == _hashable(b) for a, b in pairs):
            out.append(Law(step, "unchanged", (suffix,), {}, len(pairs)))
            continue
        if not all(is_number(a) and is_number(b) for a, b in pairs):
            continue
        deltas = [a - b for a, b in pairs]
        if all(_close(d, deltas[0]) for d in deltas):
            out.append(Law(step, "delta", (suffix,), {"c": deltas[0]}, len(pairs)))
        elif all(d >= -TOLERANCE for d in deltas):
            out.append(Law(step, "monotone", (suffix,), {"up": True}, len(pairs)))
        elif all(d <= TOLERANCE for d in deltas):
            out.append(Law(step, "monotone", (suffix,), {"up": False}, len(pairs)))
    return out


def _relations(
    step: str,
    observations: Sequence[dict[str, Any]],
    columns: dict[str, list[Any]],
    opts: MiningOptions,
) -> list[Law]:
    """Equalities, sums and products between variables that vary across observations."""
    n = len(observations)
    full = {p: c for p, c in columns.items() if len(c) == n and p != "#error"}
    varying = {
        p: c
        for p, c in full.items()
        if len({_hashable(v) for v in c}) >= 2 and not any(isinstance(v, list) for v in c)
    }
    out: list[Law] = []
    # equal: group variables with identical value sequences
    groups: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for path, column in varying.items():
        groups[tuple(_hashable(v) for v in column)].append(path)
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(key=_rank)
        anchor = members[0]
        for other in members[1:]:
            if (
                _same_suffix(anchor, other)
                and anchor.startswith("pre.")
                and other.startswith("post.")
            ):
                continue  # already an "unchanged" law
            out.append(Law(step, "equal", (other, anchor), {}, n))
    # relations are mined between one representative of each group of equal variables,
    # so that one broken fact does not count once per alias
    aliases = {other for members in groups.values() for other in sorted(members, key=_rank)[1:]}
    numeric = sorted(
        (p for p, c in varying.items() if p not in aliases and all(is_number(v) for v in c)),
        key=_rank,
    )[: opts.max_numeric]
    lists = {
        p: c
        for p, c in full.items()
        if all(isinstance(v, list) and all(map(is_number, v)) for v in c)
    }
    for total in numeric:
        for parts, column in lists.items():
            if _side(total) != _side(parts):
                continue
            if all(
                _close(t, float(sum(items)))
                for t, items in zip(varying[total], column, strict=True)
            ) and any(len(items) > 1 for items in column):
                out.append(Law(step, "sum_of", (total, parts), {}, n))
    for t in numeric:
        for i, a in enumerate(numeric):
            for b in numeric[i:]:
                if t in (a, b):
                    continue
                if all(
                    _close(x, y * z)
                    for x, y, z in zip(varying[t], varying[a], varying[b], strict=True)
                ) and not all(
                    _close(y, 1.0) or _close(z, 1.0)
                    for y, z in zip(varying[a], varying[b], strict=True)
                ):
                    out.append(Law(step, "product", (t, a, b), {}, n))
                if all(
                    _close(x, y + z)
                    for x, y, z in zip(varying[t], varying[a], varying[b], strict=True)
                ):
                    out.append(Law(step, "plus", (t, a, b), {}, n))
    return out


def _side(path: str) -> str:
    return path.split(".", 1)[0]


def _same_suffix(a: str, b: str) -> bool:
    return a.split(".", 1)[-1] == b.split(".", 1)[-1]


def _rank(path: str) -> tuple[int, str]:
    """Order variables so laws read naturally: state before, inputs, outputs, state after."""
    order = {"pre": 0, "in": 1, "out": 2, "post": 3}
    return order.get(_side(path), 4), path


def mine(
    observations: Mapping[str, Sequence[dict[str, Any]]], opts: MiningOptions | None = None
) -> list[Law]:
    """Laws for every step type, from ``{step name: [variables of each occurrence]}``."""
    opts = opts or MiningOptions()
    laws: list[Law] = []
    for step, obs in sorted(observations.items()):
        laws.extend(mine_step(step, obs, opts))
    return laws
