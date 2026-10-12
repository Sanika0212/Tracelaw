"""Importers, command line, report, pytest helpers and benchmarks (offline, small)."""

import json
import sqlite3
import zlib
from pathlib import Path

import pytest

from tracelaw import Lawbook, Run, Step, save_runs
from tracelaw.bench import faults as faults_bench
from tracelaw.bench import run as run_bench
from tracelaw.bench import taubench
from tracelaw.bench.workflows import FAULTS, TARGETS, WORKFLOWS, faulty_runs, good_runs
from tracelaw.importers import from_counterfact, from_messages, from_otel, from_taubench, ids_in
from tracelaw.report import render, write_report
from tracelaw.testing import assert_obeys


def conversation(order: str, agree: bool = True, own: str = "#W1234567") -> list[dict]:
    return [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": f"Cancel my order {own} please"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "a",
                    "function": {
                        "name": "get_order_details",
                        "arguments": json.dumps({"order_id": "#W1234567"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "a",
            "name": "get_order_details",
            "content": json.dumps({"order_id": "#W1234567", "status": "pending", "total": 10.5}),
        },
        {"role": "assistant", "content": "Shall I cancel it?"},
        {"role": "user", "content": "yes" if agree else "hmm, wait"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "b",
                    "function": {
                        "name": "cancel_pending_order",
                        "arguments": json.dumps({"order_id": order}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "b",
            "name": "cancel_pending_order",
            "content": "Error: order not found" if order != "#W1234567" else "{}",
        },
    ]


def test_ids_and_chat_transcripts() -> None:
    assert ids_in("order #W2378156 for yusuf_rossi_9620 with credit_card_9513926") == [
        "#W2378156",
        "yusuf_rossi_9620",
        "credit_card_9513926",
    ]
    run = from_messages("r", conversation("#W1234567"), passed=True)
    assert run.names == ["user", "get_order_details", "reply", "user", "cancel_pending_order"]
    lookup = run.steps[1]
    assert lookup.inputs == {"order_id": "#W1234567"}
    assert lookup.outputs["status"] == "pending"
    assert "#W1234567" in lookup.state_before["ids"]  # type: ignore[index]
    cancel = run.steps[-1]
    assert cancel.state_before["user_agreed"] is True  # type: ignore[index]
    bad = from_messages("r", conversation("#W9999999", agree=False))
    assert bad.steps[-1].error == "Error: order not found"
    assert bad.steps[-1].state_before["user_agreed"] is False  # type: ignore[index]
    orders = [f"#W{2000000 + i}" for i in range(8)]
    book = Lawbook.fit(
        [from_messages(f"r{i}", conversation(o, own=o)) for i, o in enumerate(orders)]
    )
    messages = [f.message for f in book.score(bad)[2]]
    assert any("never seen" in m or "is one of" in m or "#error" in m for m in messages)


def test_taubench_file(tmp_path: Path) -> None:
    path = tmp_path / "tau.json"
    path.write_text(
        json.dumps(
            [
                {"task_id": 3, "trial": 1, "reward": 1.0, "traj": conversation("#W1234567")},
                {"task_id": 3, "trial": 2, "reward": 0.0, "traj": conversation("#W9999999")},
            ]
        )
    )
    runs = from_taubench(path)
    assert [r.passed for r in runs] == [True, False]
    assert runs[1].meta == {"task_id": 3, "trial": 2}


def test_otel_spans(tmp_path: Path) -> None:
    def attr(key: str, value: str) -> dict:
        return {"key": key, "value": {"stringValue": value}}

    spans = [
        {
            "traceId": "t1",
            "name": "tool",
            "startTimeUnixNano": "2",
            "attributes": [
                attr("gen_ai.operation.name", "execute_tool"),
                attr("gen_ai.tool.name", "search"),
                attr("gen_ai.tool.call.arguments", '{"q": "lamp"}'),
                attr("gen_ai.tool.call.result", '{"hits": 3}'),
            ],
        },
        {
            "traceId": "t1",
            "name": "llm",
            "startTimeUnixNano": "1",
            "attributes": [
                attr("gen_ai.operation.name", "chat"),
                attr("gen_ai.request.model", "m"),
                {"key": "gen_ai.usage.input_tokens", "value": {"intValue": 12}},
            ],
        },
        {
            "traceId": "t1",
            "name": "other",
            "startTimeUnixNano": "3",
            "attributes": [],
            "status": {"code": 2, "message": "x"},
        },
    ]
    path = tmp_path / "otel.json"
    path.write_text(json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}))
    run = from_otel(path)[0]
    assert run.names == ["m", "search"]
    assert run.steps[0].outputs == {"input_tokens": 12}
    assert run.steps[1].inputs == {"q": "lamp"}


def test_counterfact_store(tmp_path: Path) -> None:
    path = tmp_path / "rec.cf.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE blobs (hash TEXT PRIMARY KEY, data BLOB, raw_size INTEGER);"
        "CREATE TABLE runs (run_id TEXT, name TEXT, seed INTEGER, started_ns INTEGER, ended_ns "
        "INTEGER, status TEXT, error TEXT, passed INTEGER, parent_run_id TEXT, meta TEXT, "
        "outcome_ref TEXT);"
        "CREATE TABLE spans (span_id TEXT, run_id TEXT, parent_id TEXT, kind TEXT, name TEXT, "
        "start_ns INTEGER, end_ns INTEGER, status TEXT, error TEXT, attributes TEXT, "
        "input_ref TEXT, output_ref TEXT, b_index INTEGER, b_key TEXT, b_fingerprint TEXT, "
        "b_seed INTEGER, b_source TEXT);"
    )

    def blob(key: str, value: object) -> str:
        con.execute(
            "INSERT INTO blobs VALUES (?, ?, 0)", (key, zlib.compress(json.dumps(value).encode()))
        )
        return key

    blob("m1", {"role": "user", "content": "hi"})
    blob("link", {"p": None, "m": "m1"})
    blob("in", {"messages": {"__cf_chain__": "link"}, "q": 1})
    blob("out", {"total": 3})
    con.execute("INSERT INTO runs VALUES ('r1','agent',0,1,2,'ok',NULL,1,NULL,'{}',NULL)")
    con.execute("INSERT INTO runs VALUES ('r2','agent',0,3,4,'ok',NULL,0,'r1','{}',NULL)")
    con.execute(
        "INSERT INTO spans VALUES ('s1','r1',NULL,'execute_tool','add',1,2,'ok',NULL,'{}',"
        "'in','out',NULL,NULL,NULL,NULL,NULL)"
    )
    con.execute(
        "INSERT INTO spans VALUES ('s0','r1',NULL,'invoke_agent','root',0,3,'ok',NULL,'{}',"
        "NULL,NULL,NULL,NULL,NULL,NULL,NULL)"
    )
    con.commit()
    con.close()
    runs = from_counterfact(path)
    assert [r.id for r in runs] == ["r1"]
    assert runs[0].passed is True
    assert runs[0].steps[0].inputs == {"messages": [{"role": "user", "content": "hi"}], "q": 1}
    assert runs[0].steps[0].outputs == {"total": 3}
    assert len(from_counterfact(path, parent_runs=True)) == 2


@pytest.fixture(scope="module")
def shop_book() -> Lawbook:
    return Lawbook.learn(good_runs("shop", 120, seed=1), alpha=0.05)


def test_report_and_assertions(shop_book: Lawbook, tmp_path: Path) -> None:
    bad = next(r for r, _, f in faulty_runs("shop", 30, seed=4) if f.kind == "wrong_update")
    verdict = shop_book.check(bad)
    page = render(bad, verdict, shop_book)
    assert "Breaks the laws of good runs" in page
    assert 'class="step first"' in page
    assert write_report(bad, verdict, shop_book, tmp_path / "r.html").exists()
    with pytest.raises(AssertionError, match="first broken at step"):
        assert_obeys(bad, shop_book)
    good = good_runs("shop", 1, seed=9)[0]
    assert assert_obeys(good, shop_book).run_id == good.id
    assert "Within the laws" in render(good, shop_book.check(good), shop_book)


def test_cli(shop_book: Lawbook, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from tracelaw import cli

    good = save_runs(good_runs("shop", 60, seed=5), tmp_path / "good.jsonl")
    laws = tmp_path / "laws.json"
    assert cli.main(["learn", str(good), "--out", str(laws), "--folds", "3"]) == 0
    assert "learned" in capsys.readouterr().out
    bad = save_runs([r for r, _, _ in faulty_runs("shop", 4, seed=6)], tmp_path / "bad.jsonl")
    code = cli.main(
        [
            "check",
            str(bad),
            "--laws",
            str(laws),
            "--html",
            str(tmp_path / "r.html"),
            "--json",
            str(tmp_path / "v.json"),
            "--fail",
        ]
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "break the laws" in out
    assert len(json.loads((tmp_path / "v.json").read_text())) == 4
    assert list(tmp_path.glob("r-*.html"))
    one = save_runs(good_runs("shop", 1, seed=7), tmp_path / "one.jsonl")
    assert (
        cli.main(["check", str(one), "--laws", str(laws), "--html", str(tmp_path / "one.html")])
        == 0
    )
    assert (tmp_path / "one.html").exists()
    assert cli.main(["explain", str(laws), "--limit", "3"]) == 0
    text = capsys.readouterr().out
    assert "held in" in text
    assert "times per good run" in text
    assert cli.main(["bench"]) == 2


def test_workflows_cover_every_fault() -> None:
    for name in WORKFLOWS:
        runs = good_runs(name, 40, seed=2)
        assert all(r.passed for r in runs)
        assert set(TARGETS[name]) == set(FAULTS)
        bad = faulty_runs(name, 40, seed=3)
        assert {f.kind for _, _, f in bad} == set(FAULTS)
        assert sum(r.passed is False for r, _, _ in bad) >= 25  # some faults stay silent
        assert all(0 <= i <= len(r) for r, i, _ in bad)  # len(r): a final step is missing


def test_fault_benchmark_small(tmp_path: Path) -> None:
    out = tmp_path / "faults"
    code = faults_bench.main(["--train", "40", "--test", "30", "--curve", "20", "--out", str(out)])
    assert code == 0
    summary = json.loads((out / "results.json").read_text())
    shop = summary["results"][0]
    assert shop["monitoring"]["Tracelaw"]["flagged_at_fault"] > 0.5
    assert shop["debugging"]["Tracelaw"] > shop["debugging"]["random step"]
    assert "20" in summary["curve"]
    assert "| shop | Tracelaw |" in (out / "results.md").read_text()


def test_taubench_suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = []
    for task in range(10):
        for trial in range(4):
            ok = (task + trial) % 3 != 0
            own = f"#W{3000000 + task * 10 + trial}"
            run = from_messages(
                f"t{task}-{trial}",
                conversation(own if ok else "#W9999999", own=own),
                passed=ok,
                meta={"task_id": task, "trial": trial},
            )
            runs.append(run)
    monkeypatch.setattr(taubench, "load", lambda name: runs)
    out = tmp_path / "tau"
    assert run_bench("taubench", ["--domains", "toy", "--folds", "2", "--out", str(out)]) == 0
    result = json.loads((out / "results.json").read_text())["results"]["toy"]
    assert result["new tasks"]["auroc"]["Tracelaw"] > 0.5
    assert "| toy | recurring tasks |" in (out / "results.md").read_text()
    assert taubench.auroc([1.0, 1.0], [True, False]) == 0.5


def test_step_shapes() -> None:
    run = Run("x", [Step("a", error="boom")])
    assert run.steps[0].error == "boom"
