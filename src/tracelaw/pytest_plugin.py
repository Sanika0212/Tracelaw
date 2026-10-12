"""pytest plugin: a ``lawbook`` fixture, from ``--tracelaw-laws`` or the ``tracelaw_laws`` ini."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def pytest_addoption(parser: Any) -> None:
    parser.addoption("--tracelaw-laws", default=None, help="lawbook (laws.json) for tests")
    parser.addini("tracelaw_laws", "lawbook (laws.json) for tests", default="")


@pytest.fixture
def lawbook(request: Any) -> Any:
    from tracelaw.lawbook import Lawbook

    path = request.config.getoption("--tracelaw-laws") or request.config.getini("tracelaw_laws")
    if not path:
        pytest.skip("no lawbook: pass --tracelaw-laws laws.json or set tracelaw_laws")
    return Lawbook.load(Path(str(path)))
