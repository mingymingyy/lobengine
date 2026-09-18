"""Public market-data adapters.

Both venues below expose an unauthenticated public book channel. Nothing
here creates an account, signs a request, or sends an order: this is
read-only market data used as a stand-in for the exchange feeds a broker
would licence.

A note for anyone in Singapore reproducing this: MAS added Binance.com to
its Investor Alert List in September 2021 over Payment Services Act
concerns, so Binance is deliberately not one of the adapters here even
for read-only data. If you add an adapter of your own, check the venue's
standing with MAS first.

The two adapters differ in exactly the way that makes the abstraction
worth having:

    OKX   publishes seqId and prevSeqId on every message, so gap
          detection is explicit and given to you.
    Bybit publishes a single monotonically increasing update id `u`, so
          the adapter has to remember the last id it saw and synthesise
          `prev_seq` itself.

Both formats are as documented by the venues at the time of writing.
Message shapes do change, so `parse` is written to return an empty list
rather than raise when it meets something it does not recognise, and the
test suite pins the shapes it expects.
"""

from __future__ import annotations

import json

from ..types import BookUpdate, Level
from .base import FeedAdapter


def _levels(rows, scale, price_idx=0, size_idx=1) -> tuple[Level, ...]:
    out = []
    for row in rows:
        try:
            price = scale.to_ticks(row[price_idx])
            size = float(row[size_idx])
        except (ValueError, IndexError, TypeError):
            continue
        out.append(Level(price, size))
    return tuple(out)


class OKXBooksAdapter(FeedAdapter):
    """OKX v5 public `books` channel (400 levels, incremental).

    Frames look like::

        {"arg":{"channel":"books","instId":"BTC-USDT"},
         "action":"snapshot"|"update",
         "data":[{"asks":[["68120.1","0.5","0","1"],...],
                  "bids":[...],
                  "ts":"1700000000123",
                  "seqId":123457,"prevSeqId":123456}]}

    Each level row is [price, size, deprecated, order_count].
    """

    venue = "okx"
    max_depth = 400
    WS_URL = "wss://ws.okx.com:8443/ws/v5/public"

    def __init__(self, symbol: str = "BTC-USDT", price_decimals: int = 1,
                 channel: str = "books") -> None:
        super().__init__(symbol, price_decimals)
        self.channel = channel

    def url(self) -> str:
        return self.WS_URL

    def subscribe_payloads(self) -> list[str]:
        return [
            json.dumps(
                {
                    "op": "subscribe",
                    "args": [{"channel": self.channel, "instId": self.symbol}],
                }
            )
        ]

    def ping_payload(self) -> str | None:
        # OKX drops connections idle for 30s; it expects the literal
        # string "ping" rather than a websocket ping frame.
        return "ping"

    def parse(self, raw: str, recv_ts_ns: int) -> list[BookUpdate]:
        if raw == "pong":
            return []
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
        if not isinstance(msg, dict) or "data" not in msg:
            return []  # subscription ack, error, etc.
        arg = msg.get("arg", {})
        if arg.get("channel") not in ("books", "books-l2-tbt", "books50-l2-tbt", "books5"):
            return []

        is_snapshot = msg.get("action", "snapshot") == "snapshot"
        symbol = arg.get("instId", self.symbol)
        out: list[BookUpdate] = []
        for d in msg["data"]:
            ts = d.get("ts")
            out.append(
                BookUpdate(
                    symbol=symbol,
                    bids=_levels(d.get("bids", []), self.scale),
                    asks=_levels(d.get("asks", []), self.scale),
                    is_snapshot=is_snapshot,
                    seq=_maybe_int(d.get("seqId")),
                    prev_seq=_maybe_int(d.get("prevSeqId")),
                    exchange_ts_ns=int(ts) * 1_000_000 if ts else None,
                    recv_ts_ns=recv_ts_ns,
                    venue=self.venue,
                )
            )
        return out


class BybitOrderbookAdapter(FeedAdapter):
    """Bybit v5 public `orderbook.{depth}.{symbol}` channel.

    Frames look like::

        {"topic":"orderbook.50.BTCUSDT","type":"snapshot"|"delta",
         "ts":1700000000123,
         "data":{"s":"BTCUSDT","b":[["68120.10","0.5"],...],
                 "a":[...],"u":18521,"seq":7961638724}}

    Bybit sends `u` (update id) but no "previous" id, so the adapter has
    to synthesise one.

    `assume_contiguous_ids` controls how:

        True  (default) prev_seq = u - 1. Bybit documents `u` as always in
              sequence, so a message lost on the network leaves a hole
              that the book will catch. This is the setting you want.
        False prev_seq = the last id this adapter actually saw. That only
              catches reordering, never a genuine drop, because a message
              the adapter never received cannot affect what it remembers.
              Kept as an escape hatch in case a venue's ids are not +1.

    Bybit also resets `u` to 1 when the service restarts, which arrives
    as a snapshot; a snapshot is always an unconditional resync.
    """

    venue = "bybit"
    WS_URL = "wss://stream.bybit.com/v5/public/spot"

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        price_decimals: int = 2,
        depth: int = 50,
        assume_contiguous_ids: bool = True,
    ) -> None:
        super().__init__(symbol, price_decimals)
        self.depth = depth
        self.max_depth = depth
        self.assume_contiguous_ids = assume_contiguous_ids
        self._last_u: int | None = None

    def url(self) -> str:
        return self.WS_URL

    def subscribe_payloads(self) -> list[str]:
        return [
            json.dumps(
                {"op": "subscribe", "args": [f"orderbook.{self.depth}.{self.symbol}"]}
            )
        ]

    def ping_payload(self) -> str | None:
        return json.dumps({"op": "ping"})

    def parse(self, raw: str, recv_ts_ns: int) -> list[BookUpdate]:
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
        if not isinstance(msg, dict):
            return []
        topic = msg.get("topic", "")
        if not topic.startswith("orderbook.") or "data" not in msg:
            return []

        d = msg["data"]
        is_snapshot = msg.get("type") == "snapshot"
        u = _maybe_int(d.get("u"))
        if is_snapshot:
            prev = None
        elif self.assume_contiguous_ids:
            prev = u - 1 if u is not None else None
        else:
            prev = self._last_u
        if u is not None:
            self._last_u = u

        ts = msg.get("ts")
        return [
            BookUpdate(
                symbol=d.get("s", self.symbol),
                bids=_levels(d.get("b", []), self.scale),
                asks=_levels(d.get("a", []), self.scale),
                is_snapshot=is_snapshot,
                seq=u,
                prev_seq=prev,
                exchange_ts_ns=int(ts) * 1_000_000 if ts else None,
                recv_ts_ns=recv_ts_ns,
                venue=self.venue,
            )
        ]


def _maybe_int(v) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


ADAPTERS = {
    "okx": OKXBooksAdapter,
    "bybit": BybitOrderbookAdapter,
}
