import gzip
import json
import math

import pytest

from lob_engine.latency import LatencyBook, LatencyRecorder
from lob_engine.recorder import Recorder, capture_header, read_capture


class TestLatencyRecorder:
    def test_percentiles_are_ordered(self):
        r = LatencyRecorder("x")
        for ns in range(1, 10_001):
            r.record(ns)
        rep = r.report()
        assert rep.count == 10_000
        assert rep.p50_ns < rep.p90_ns < rep.p99_ns < rep.p999_ns <= rep.max_ns

    def test_known_values(self):
        r = LatencyRecorder("x")
        for ns in range(101):
            r.record(ns)
        rep = r.report()
        assert rep.p50_ns == pytest.approx(50.0)
        assert rep.max_ns == 100.0
        assert rep.mean_ns == pytest.approx(50.0)

    def test_warmup_is_discarded(self):
        r = LatencyRecorder("x", warmup=10)
        for ns in range(20):
            r.record(ns)
        assert r.report().count == 10
        assert min(r.samples) == 10

    def test_empty_report_is_nan_not_a_crash(self):
        rep = LatencyRecorder("x").report()
        assert rep.count == 0
        assert math.isnan(rep.p50_ns)
        assert "x" in rep.as_row()

    def test_context_manager_records_something(self):
        r = LatencyRecorder("x")
        with r.time():
            sum(range(1000))
        assert r.report().count == 1
        assert r.samples[0] > 0

    def test_reset(self):
        r = LatencyRecorder("x", warmup=5)
        for ns in range(20):
            r.record(ns)
        r.reset()
        assert r.report().count == 0
        for ns in range(10):
            r.record(ns)
        assert r.report().count == 5  # warmup applies again


class TestLatencyBook:
    def test_creates_recorders_on_demand(self):
        b = LatencyBook()
        b["parse"].record(10)
        b["apply"].record(20)
        assert {r.name for r in b.reports()} == {"parse", "apply"}

    def test_same_name_returns_the_same_recorder(self):
        b = LatencyBook()
        assert b["parse"] is b["parse"]

    def test_render_includes_every_stage(self):
        b = LatencyBook()
        for name in ("a", "b", "c"):
            b[name].record(100)
        text = b.render()
        assert all(n in text for n in "abc")
        assert "microseconds" in text


class TestRecorder:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "cap.jsonl.gz"
        with Recorder(path, venue="okx", symbol="BTC-USDT") as rec:
            rec.record('{"a":1}', 111)
            rec.record('{"a":2}', 222)
        assert list(read_capture(path)) == [(111, '{"a":1}'), (222, '{"a":2}')]

    def test_header_is_readable(self, tmp_path):
        path = tmp_path / "cap.jsonl.gz"
        with Recorder(path, venue="bybit", symbol="BTCUSDT"):
            pass
        h = capture_header(path)
        assert h["venue"] == "bybit"
        assert h["symbol"] == "BTCUSDT"
        assert h["format_version"] == 1

    def test_footer_records_the_count(self, tmp_path):
        path = tmp_path / "cap.jsonl.gz"
        with Recorder(path, venue="okx", symbol="X") as rec:
            for i in range(5):
                rec.record(str(i), i)
        with gzip.open(path, "rt") as fh:
            last = json.loads([ln for ln in fh if ln.strip()][-1])
        assert last["type"] == "footer" and last["count"] == 5

    def test_raw_payload_is_preserved_byte_for_byte(self, tmp_path):
        """Recording parsed objects instead of raw bytes bakes in any
        parser bug permanently."""
        path = tmp_path / "cap.jsonl.gz"
        tricky = '{"weird": "unicode \\u00e9, quotes \\", newline \\n"}'
        with Recorder(path, venue="okx", symbol="X") as rec:
            rec.record(tricky, 1)
        assert list(read_capture(path))[0][1] == tricky

    def test_timestamp_defaults_to_now(self, tmp_path):
        path = tmp_path / "cap.jsonl.gz"
        with Recorder(path, venue="okx", symbol="X") as rec:
            rec.record("{}")
        assert list(read_capture(path))[0][0] > 0

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "cap.jsonl.gz"
        with Recorder(path, venue="okx", symbol="X"):
            pass
        assert path.exists()

    def test_plain_jsonl_also_readable(self, tmp_path):
        path = tmp_path / "cap.jsonl"
        path.write_text(
            json.dumps({"type": "header", "venue": "okx", "symbol": "X"}) + "\n"
            + json.dumps({"type": "msg", "recv_ns": 5, "raw": "hi"}) + "\n"
        )
        assert list(read_capture(path)) == [(5, "hi")]
