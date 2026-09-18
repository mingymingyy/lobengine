"""Live websocket client.

Responsibilities, in the order they bite you in production:

1. **Connect and stay connected.** Reconnect with exponential backoff and
   jitter. Without jitter, every process you run reconnects in lockstep
   after an outage and hammers the venue at the same instant.
2. **Resync on a gap.** When the book raises `SequenceGap`, the only
   correct response is to throw the book away and get a fresh snapshot.
   Carrying on with a book you know is wrong is how a market maker ends
   up quoting through a stale price.
3. **Keep the hot path short.** Receive, stamp, hand off. Recording is a
   buffered append; anything heavier belongs downstream.

`websockets` is an optional dependency: the rest of the package, and the
entire test suite, work without it. Only this module needs it.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass

from ..book import OrderBook, SequenceGap
from ..latency import LatencyBook
from ..recorder import Recorder
from .base import FeedAdapter


@dataclass(slots=True)
class ClientStats:
    messages: int = 0
    updates: int = 0
    resyncs: int = 0
    reconnects: int = 0
    parse_errors: int = 0


class MarketDataClient:
    """Drives one adapter against one book.

    Parameters
    ----------
    on_book
        Called after every successfully applied update, with the book.
        Keep it fast; it runs in the receive loop.
    recorder
        Optional `Recorder`. Records the raw frame with a receive
        timestamp taken before parsing.
    max_reconnect_delay
        Cap on the backoff, in seconds.
    """

    def __init__(
        self,
        adapter: FeedAdapter,
        book: OrderBook | None = None,
        on_book=None,
        recorder: Recorder | None = None,
        latency: LatencyBook | None = None,
        max_reconnect_delay: float = 30.0,
    ) -> None:
        self.adapter = adapter
        self.book = book or OrderBook(adapter.symbol, max_depth=adapter.max_depth)
        self.on_book = on_book or (lambda b: None)
        self.recorder = recorder
        self.latency = latency or LatencyBook(warmup=50)
        self.max_reconnect_delay = max_reconnect_delay
        self.stats = ClientStats()
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------ #
    async def run(self, duration_s: float | None = None) -> None:
        """Connect, subscribe and consume until stopped."""
        try:
            import websockets
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "live capture needs the 'websockets' package: pip install websockets"
            ) from exc

        deadline = time.monotonic() + duration_s if duration_s else None
        attempt = 0

        while not self._stop.is_set():
            if deadline and time.monotonic() >= deadline:
                return
            try:
                async with websockets.connect(
                    self.adapter.url(), open_timeout=10, ping_interval=20
                ) as ws:
                    attempt = 0
                    self.book.reset()
                    for payload in self.adapter.subscribe_payloads():
                        await ws.send(payload)
                    await self._consume(ws, deadline)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect on anything
                if self._stop.is_set():
                    return
                self.stats.reconnects += 1
                attempt += 1
                delay = min(self.max_reconnect_delay, 2.0**attempt * 0.25)
                delay *= 0.5 + random.random()  # jitter, avoid thundering herd
                print(f"[client] {type(exc).__name__}: {exc}; reconnecting in {delay:.1f}s")
                await asyncio.sleep(delay)

    async def _consume(self, ws, deadline: float | None) -> None:
        ping_payload = self.adapter.ping_payload()
        last_ping = time.monotonic()

        while not self._stop.is_set():
            if deadline and time.monotonic() >= deadline:
                self._stop.set()
                return
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            except asyncio.TimeoutError:
                if ping_payload and time.monotonic() - last_ping > 20:
                    await ws.send(ping_payload)
                    last_ping = time.monotonic()
                continue

            recv_ns = time.time_ns()
            t0 = time.perf_counter_ns()
            self.stats.messages += 1

            if self.recorder is not None:
                self.recorder.record(raw, recv_ns)

            try:
                updates = self.adapter.parse(raw, recv_ns)
            except Exception:  # noqa: BLE001 - never let one bad frame kill the feed
                self.stats.parse_errors += 1
                continue
            self.latency["parse"].record(time.perf_counter_ns() - t0)

            for upd in updates:
                t1 = time.perf_counter_ns()
                try:
                    self.book.apply(upd)
                except SequenceGap as gap:
                    self.stats.resyncs += 1
                    print(f"[client] {gap}; resubscribing for a fresh snapshot")
                    self.book.reset()
                    for payload in self.adapter.subscribe_payloads():
                        await ws.send(payload)
                    break
                self.latency["book_apply"].record(time.perf_counter_ns() - t1)
                self.stats.updates += 1
                self.on_book(self.book)


def replay(
    adapter: FeedAdapter,
    messages,
    book: OrderBook | None = None,
    on_book=None,
) -> tuple[OrderBook, ClientStats]:
    """Replay recorded (recv_ns, raw) pairs through the same code path.

    This is the reason `parse` is a pure function: capture once, then
    debug the book against the exact bytes the venue sent, as many times
    as you like, with no network.
    """
    book = book or OrderBook(adapter.symbol, max_depth=adapter.max_depth)
    on_book = on_book or (lambda b: None)
    stats = ClientStats()

    for recv_ns, raw in messages:
        stats.messages += 1
        try:
            updates = adapter.parse(raw, recv_ns)
        except Exception:  # noqa: BLE001
            stats.parse_errors += 1
            continue
        for upd in updates:
            try:
                book.apply(upd)
            except SequenceGap:
                stats.resyncs += 1
                book.reset()
                break
            stats.updates += 1
            on_book(book)
    return book, stats
