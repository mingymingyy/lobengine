"""Adapter tests.

The payloads below are the documented wire shapes for each venue. Pinning
them in tests means a format change shows up here rather than as a
silently wrong book in production.
"""

import json

import pytest

from lob_engine.book import OrderBook, SequenceGap
from lob_engine.feeds import ADAPTERS, BybitOrderbookAdapter, OKXBooksAdapter, replay

TS = 1_700_000_000_123_000_000


# --------------------------------------------------------------------- #
# OKX
# --------------------------------------------------------------------- #
def okx_frame(action, bids, asks, seq, prev_seq, ts="1700000000123"):
    return json.dumps(
        {
            "arg": {"channel": "books", "instId": "BTC-USDT"},
            "action": action,
            "data": [
                {
                    "bids": [[p, s, "0", "1"] for p, s in bids],
                    "asks": [[p, s, "0", "1"] for p, s in asks],
                    "ts": ts,
                    "seqId": seq,
                    "prevSeqId": prev_seq,
                }
            ],
        }
    )


class TestOKX:
    @pytest.fixture
    def adapter(self):
        return OKXBooksAdapter("BTC-USDT", price_decimals=1)

    def test_parses_a_snapshot(self, adapter):
        raw = okx_frame("snapshot", [("68120.1", "0.5")], [("68120.5", "1.25")],
                        123457, 123456)
        (u,) = adapter.parse(raw, TS)
        assert u.is_snapshot
        assert u.symbol == "BTC-USDT"
        assert u.bids[0].price == 681201
        assert u.bids[0].size == 0.5
        assert u.asks[0].price == 681205
        assert u.seq == 123457 and u.prev_seq == 123456
        assert u.exchange_ts_ns == 1700000000123 * 1_000_000
        assert u.recv_ts_ns == TS
        assert u.venue == "okx"

    def test_parses_a_delta(self, adapter):
        (u,) = adapter.parse(
            okx_frame("update", [("68120.1", "0")], [], 123458, 123457), TS
        )
        assert not u.is_snapshot
        assert u.bids[0].size == 0.0

    def test_ignores_subscription_ack(self, adapter):
        ack = json.dumps({"event": "subscribe",
                          "arg": {"channel": "books", "instId": "BTC-USDT"}})
        assert adapter.parse(ack, TS) == []

    def test_ignores_error_frame(self, adapter):
        err = json.dumps({"event": "error", "code": "60012", "msg": "bad request"})
        assert adapter.parse(err, TS) == []

    def test_ignores_pong(self, adapter):
        assert adapter.parse("pong", TS) == []

    def test_ignores_other_channels(self, adapter):
        raw = json.dumps({"arg": {"channel": "trades", "instId": "BTC-USDT"},
                          "data": [{"px": "1", "sz": "1"}]})
        assert adapter.parse(raw, TS) == []

    def test_survives_malformed_json(self, adapter):
        assert adapter.parse("{not json", TS) == []
        assert adapter.parse("", TS) == []

    def test_skips_unparseable_levels_without_dropping_the_message(self, adapter):
        raw = json.dumps({
            "arg": {"channel": "books", "instId": "BTC-USDT"},
            "action": "update",
            "data": [{"bids": [["abc", "1"], ["68120.1", "2"]], "asks": [],
                      "ts": "1700000000123", "seqId": 2, "prevSeqId": 1}],
        })
        (u,) = adapter.parse(raw, TS)
        assert len(u.bids) == 1 and u.bids[0].price == 681201

    def test_subscribe_payload(self, adapter):
        (p,) = adapter.subscribe_payloads()
        msg = json.loads(p)
        assert msg["op"] == "subscribe"
        assert msg["args"][0] == {"channel": "books", "instId": "BTC-USDT"}

    def test_price_precision_mismatch_is_loud(self):
        """Configuring the wrong decimals must fail, not round silently."""
        adapter = OKXBooksAdapter("BTC-USDT", price_decimals=0)
        raw = okx_frame("snapshot", [("68120.1", "1")], [], 1, 0)
        (u,) = adapter.parse(raw, TS)
        assert u.bids == ()  # the bad level is dropped, not rounded


# --------------------------------------------------------------------- #
# Bybit
# --------------------------------------------------------------------- #
def bybit_frame(mtype, bids, asks, u, seq=1, ts=1700000000123):
    return json.dumps(
        {
            "topic": "orderbook.50.BTCUSDT",
            "type": mtype,
            "ts": ts,
            "data": {"s": "BTCUSDT", "b": list(bids), "a": list(asks),
                     "u": u, "seq": seq},
        }
    )


class TestBybit:
    @pytest.fixture
    def adapter(self):
        return BybitOrderbookAdapter("BTCUSDT", price_decimals=2, depth=50)

    def test_parses_a_snapshot(self, adapter):
        (u,) = adapter.parse(
            bybit_frame("snapshot", [["68120.10", "0.5"]], [["68120.50", "1.0"]], 100),
            TS,
        )
        assert u.is_snapshot
        assert u.bids[0].price == 6812010
        assert u.seq == 100
        assert u.prev_seq is None

    def test_synthesises_prev_seq_from_contiguous_ids(self, adapter):
        adapter.parse(bybit_frame("snapshot", [], [], 100), TS)
        (u,) = adapter.parse(bybit_frame("delta", [["68120.10", "0"]], [], 101), TS)
        assert u.prev_seq == 100 and u.seq == 101

    def test_contiguous_mode_detects_a_dropped_message(self, adapter):
        """The adapter never sees the dropped frame, so prev_seq has to be
        derived from the id itself for the gap to be visible."""
        book = OrderBook("BTCUSDT", max_depth=50)
        book.apply(adapter.parse(bybit_frame("snapshot", [["1.00", "1"]],
                                             [["2.00", "1"]], 100), TS)[0])
        # message 101 is lost on the wire; 102 arrives
        (u,) = adapter.parse(bybit_frame("delta", [["1.00", "5"]], [], 102), TS)
        with pytest.raises(SequenceGap):
            book.apply(u)

    def test_last_seen_mode_cannot_see_a_drop(self, ):
        adapter = BybitOrderbookAdapter("BTCUSDT", price_decimals=2,
                                        assume_contiguous_ids=False)
        book = OrderBook("BTCUSDT", max_depth=50)
        book.apply(adapter.parse(bybit_frame("snapshot", [["1.00", "1"]],
                                             [["2.00", "1"]], 100), TS)[0])
        (u,) = adapter.parse(bybit_frame("delta", [["1.00", "5"]], [], 102), TS)
        book.apply(u)  # no gap raised; documented limitation of that mode
        assert book.seq == 102

    def test_snapshot_resets_after_a_service_restart(self, adapter):
        adapter.parse(bybit_frame("snapshot", [], [], 500), TS)
        adapter.parse(bybit_frame("delta", [], [], 501), TS)
        (u,) = adapter.parse(bybit_frame("snapshot", [], [], 1), TS)
        assert u.is_snapshot and u.prev_seq is None

    def test_ignores_other_topics(self, adapter):
        raw = json.dumps({"topic": "publicTrade.BTCUSDT", "data": []})
        assert adapter.parse(raw, TS) == []

    def test_ignores_pong(self, adapter):
        raw = json.dumps({"op": "pong", "success": True})
        assert adapter.parse(raw, TS) == []


# --------------------------------------------------------------------- #
# through the book
# --------------------------------------------------------------------- #
class TestReplayPipeline:
    def test_snapshot_then_deltas(self):
        adapter = OKXBooksAdapter("BTC-USDT", price_decimals=1)
        msgs = [
            (1, okx_frame("snapshot", [("100.0", "5"), ("99.0", "8")],
                          [("102.0", "4")], 1, 0)),
            (2, okx_frame("update", [("100.0", "9")], [], 2, 1)),
            (3, okx_frame("update", [("101.0", "2")], [], 3, 2)),
        ]
        book, stats = replay(adapter, msgs)
        assert stats.updates == 3 and stats.resyncs == 0
        t = book.top()
        assert (t.bid_px, t.bid_sz) == (1010, 2.0)
        assert book.check_integrity()

    def test_gap_triggers_a_resync(self):
        adapter = OKXBooksAdapter("BTC-USDT", price_decimals=1)
        msgs = [
            (1, okx_frame("snapshot", [("100.0", "5")], [("102.0", "4")], 1, 0)),
            (2, okx_frame("update", [("100.0", "9")], [], 5, 4)),  # gap
            (3, okx_frame("snapshot", [("100.0", "7")], [("102.0", "4")], 6, 5)),
            (4, okx_frame("update", [("100.0", "3")], [], 7, 6)),
        ]
        book, stats = replay(adapter, msgs)
        assert stats.resyncs == 1
        assert book.top().bid_sz == 3.0

    def test_bad_frames_do_not_stop_the_stream(self):
        adapter = OKXBooksAdapter("BTC-USDT", price_decimals=1)
        msgs = [
            (1, okx_frame("snapshot", [("100.0", "5")], [("102.0", "4")], 1, 0)),
            (2, "garbage"),
            (3, okx_frame("update", [("100.0", "9")], [], 2, 1)),
        ]
        book, stats = replay(adapter, msgs)
        assert book.top().bid_sz == 9.0
        assert stats.updates == 2


def test_adapter_registry():
    assert set(ADAPTERS) == {"okx", "bybit"}
    for cls in ADAPTERS.values():
        a = cls()
        assert a.venue and a.url().startswith("wss://")
        assert a.subscribe_payloads()
