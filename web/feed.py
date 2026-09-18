"""Venue feed and derived measures, shared by both front ends.

`web/server.py` (websocket dashboard) and `streamlit_app.py` drive the
exact same object. Keeping one implementation matters more than it
looks: the OFI window is destructive to read, and a second copy of that
logic is a second place to get it wrong.

This is read-only public market data. No key, no account, no order ever
leaves this process - the OMS/FIX half of the engine is not wired in.

Configuration, all optional, via environment:

    VENUE           okx (default) | bybit
    SYMBOL          BTC-USDT for okx, BTCUSDT for bybit
    PRICE_DECIMALS  tick size exponent; 1 for okx, 2 for bybit
    DEPTH           ladder rows to publish, default 15
    HZ              tick/broadcast rate, default 10
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # so it runs without `pip install -e .`

from lob_engine.feeds.client import MarketDataClient  # noqa: E402
from lob_engine.feeds.exchanges import (  # noqa: E402
    BybitOrderbookAdapter,
    OKXBooksAdapter,
)
from lob_engine.ofi import OFIAccumulator  # noqa: E402
from lob_engine.types import Side  # noqa: E402

VENUE = os.environ.get("VENUE", "okx").lower()
DEPTH = int(os.environ.get("DEPTH", "15"))
HZ = float(os.environ.get("HZ", "10"))


def build_adapter():
    """Pick a venue adapter from the environment.

    Binance is absent on purpose; see the note in feeds/exchanges.py.
    """
    if VENUE == "bybit":
        return BybitOrderbookAdapter(
            symbol=os.environ.get("SYMBOL", "BTCUSDT"),
            price_decimals=int(os.environ.get("PRICE_DECIMALS", "2")),
            depth=int(os.environ.get("VENUE_DEPTH", "50")),
        )
    if VENUE == "okx":
        return OKXBooksAdapter(
            symbol=os.environ.get("SYMBOL", "BTC-USDT"),
            price_decimals=int(os.environ.get("PRICE_DECIMALS", "1")),
        )
    raise SystemExit(f"unknown VENUE {VENUE!r}: expected 'okx' or 'bybit'")


class Feed:
    """Owns the adapter, the book and the numbers derived from them."""

    def __init__(self) -> None:
        self.adapter = build_adapter()
        self.ofi = OFIAccumulator()
        self.client = MarketDataClient(self.adapter, on_book=self._on_book)
        self.book = self.client.book
        self.started_ns = time.time_ns()
        self._last_msgs = 0
        self._last_rate_ns = self.started_ns
        self._msg_rate = 0.0
        self._ofi_cum = 0.0
        self._ofi_window = 0.0
        self._ofi_events = 0

    # hot path: runs inside the receive loop, once per applied update.
    def _on_book(self, book) -> None:
        self._ofi_cum += self.ofi.update(book.top())

    def tick(self) -> None:
        """Advance the windowed measures. Called only by the broadcaster.

        This is deliberately separate from `snapshot`, which is a pure
        read. Draining the OFI accumulator is destructive, so if it
        happened inside `snapshot` then every hit on the public
        `/api/snapshot` would steal the window from the next broadcast
        and flatten the OFI bar for everyone watching. One uptime monitor
        polling that URL would be enough to break the display.
        """
        stats = self.client.stats
        now_ns = time.time_ns()

        elapsed = (now_ns - self._last_rate_ns) / 1e9
        if elapsed >= 0.5:
            self._msg_rate = (stats.messages - self._last_msgs) / elapsed
            self._last_msgs = stats.messages
            self._last_rate_ns = now_ns

        # OFI since the previous tick, so the bar reads as pressure right
        # now rather than as an ever-growing total.
        self._ofi_window, self._ofi_events = self.ofi.drain()

    def snapshot(self) -> dict:
        book = self.book
        scale = self.adapter.scale
        top = book.top()
        stats = self.client.stats

        now_ns = time.time_ns()
        ofi_window, ofi_events = self._ofi_window, self._ofi_events

        def ladder(side: Side) -> list[list[float]]:
            return [
                [scale.to_float(lv.price), lv.size]
                for lv in book.depth(side, DEPTH)
            ]

        px = scale.to_float
        tick = scale.to_float(1)
        return {
            "venue": self.adapter.venue,
            "symbol": book.symbol,
            "ready": book.ready,
            # Sent explicitly so the page never has to guess precision
            # from a sample price. A top level that happens to land on a
            # round number would otherwise make the whole ladder render
            # as integers and the spread collapse to "0".
            "price_decimals": scale.decimals,
            "seq": book.seq,
            "server_ts_ms": now_ns // 1_000_000,
            "exchange_ts_ms": (
                top.exchange_ts_ns // 1_000_000 if top.exchange_ts_ns else None
            ),
            # How long since this process last applied an update. Local
            # clock on both ends, so it is always meaningful: this is the
            # number that says whether the feed is alive.
            "feed_age_ms": (
                (now_ns - top.recv_ts_ns) / 1e6 if top.recv_ts_ns else None
            ),
            # Venue timestamp to our receive time. Useful, but it spans
            # two machines' clocks, so an unsynchronised host makes it
            # meaningless and it can even come out negative. Reported as
            # measured rather than clamped, and flagged when it inverts.
            "venue_lag_ms": (
                (top.recv_ts_ns - top.exchange_ts_ns) / 1e6
                if top.exchange_ts_ns and top.recv_ts_ns
                else None
            ),
            "bids": ladder(Side.BUY),
            "asks": ladder(Side.SELL),
            "bid_px": px(top.bid_px) if top.bid_px is not None else None,
            "ask_px": px(top.ask_px) if top.ask_px is not None else None,
            "bid_sz": top.bid_sz,
            "ask_sz": top.ask_sz,
            # mid and microprice land between ticks, so they are scaled by
            # the tick size rather than pushed through to_float, which
            # takes a whole number of ticks.
            "mid": top.mid * tick if top.mid is not None else None,
            "microprice": top.microprice * tick if top.microprice is not None else None,
            "spread": top.spread * tick if top.spread is not None else None,
            "spread_ticks": top.spread,
            "ofi_window": ofi_window,
            "ofi_events": ofi_events,
            "ofi_cum": self._ofi_cum,
            "avg_depth": book.average_depth(5),
            "stats": {
                "messages": stats.messages,
                "updates": stats.updates,
                "resyncs": stats.resyncs,
                "reconnects": stats.reconnects,
                "parse_errors": stats.parse_errors,
                "msg_rate": round(self._msg_rate, 1),
                "uptime_s": round((now_ns - self.started_ns) / 1e9),
            },
        }
