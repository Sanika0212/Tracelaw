"""Turn a step into named variables that laws can talk about.

Nested values become dotted paths with a prefix for where they came from:

- ``pre.cart.total`` and ``post.cart.total``: state before and after the step;
- ``in.user_id``: an argument; ``out.status``: an output.

Lists are summarised rather than indexed, because their length varies between runs:
``post.cart.items#len`` is the length, ``post.cart.items[].price`` the multiset of the
``price`` field across elements, and ``post.cart.items[]`` the multiset of the elements
themselves when they are scalars.
"""

from __future__ import annotations

from typing import Any

from tracelaw.model import Step

Scalar = bool | int | float | str | None


def flatten(value: Any, prefix: str, out: dict[str, Any], *, depth: int = 6) -> None:
    if depth < 0:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            flatten(item, f"{prefix}.{key}", out, depth=depth - 1)
    elif isinstance(value, list | tuple):
        out[f"{prefix}#len"] = len(value)
        scalars = [v for v in value if not isinstance(v, dict | list | tuple)]
        if scalars and len(scalars) == len(value):
            out[f"{prefix}[]"] = list(scalars)
        dicts = [v for v in value if isinstance(v, dict)]
        if dicts:
            keys = set().union(*(d.keys() for d in dicts))
            for key in sorted(keys):
                column = [d.get(key) for d in dicts]
                if all(not isinstance(v, dict | list | tuple) for v in column):
                    out[f"{prefix}[].{key}"] = column
    else:
        out[prefix] = value


def variables(step: Step) -> dict[str, Any]:
    """Every variable of a step, as ``{path: scalar or list of scalars}``."""
    out: dict[str, Any] = {}
    if step.state_before is not None:
        flatten(step.state_before, "pre", out)
    if step.state_after is not None:
        flatten(step.state_after, "post", out)
    flatten(step.inputs, "in", out)
    flatten(step.outputs, "out", out)
    out["#error"] = step.error is not None
    return out


def is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def kind(value: Any) -> str:
    """A coarse type name: number, bool, str, null or list."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if is_number(value):
        return "number"
    if isinstance(value, list):
        return "list"
    return "str"
