#!/usr/bin/env python3
"""Capture live public market data to a replayable file.

Read-only public market data. No account, no API key, no orders.

    python scripts/capture.py --venue okx --symbol BTC-USDT --seconds 300
    python scripts/capture.py --venue bybit --symbol BTCUSDT --decimals 2 --seconds 300

The output feeds straight into scripts/analyse_ofi.py.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lob_engine.book import OrderBook
from lob_engine.feeds import ADAPTERS, MarketDataClient
from lob_engine.recorder import Recorder


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--venue", choices=sorted(ADAPTERS), default="okx")
    ap.add_argument("--symbol", default=None, help="venue-specific instrument id")
    ap.add_argument(
        "--decimals",
        type=int,
        default=None,
        help="price decimal places; must match the venue exactly",
    )
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cls = ADAPTERS[args.venue]
    kwargs = {}
    if args.symbol:
        kwargs["symbol"] = args.symbol
    if args.decimals is not None:
        kwargs["price_decimals"] = args.decimals
    adapter = cls(**kwargs)

    out = Path(
        args.out
        or f"data/{args.venue}_{adapter.symbol.replace('/', '')}_"
        f"{time.strftime('%Y%m%d_%H%M%S')}.jsonl.gz"
    )

    book = OrderBook(adapter.symbol, max_depth=adapter.max_depth)
    recorder = Recorder(out, venue=adapter.venue, symbol=adapter.symbol)
    client = MarketDataClient(adapter, book=book, recorder=recorder)

    last_print = [0.0]

    def on_book(b: OrderBook) -> None:
        now = time.monotonic()
        if now - last_print[0] < 2.0:
            return
        last_print[0] = now
        t = b.top()
        if t.is_two_sided:
            print(
                f"\r{b.symbol}  {t.bid_sz:>9.4f} @ {t.bid_px:<10} | "
                f"{t.ask_px:>10} @ {t.ask_sz:<9.4f}  "
                f"msgs={client.stats.messages:,}  resyncs={client.stats.resyncs}",
                end="",
                flush=True,
            )

    client.on_book = on_book
    print(f"connecting to {adapter.venue} for {adapter.symbol}, writing to {out}")
    print("ctrl-c to stop early\n")

    try:
        asyncio.run(client.run(duration_s=args.seconds))
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        recorder.close()

    s = client.stats
    print(f"\n\nmessages   {s.messages:,}")
    print(f"updates    {s.updates:,}")
    print(f"resyncs    {s.resyncs}")
    print(f"reconnects {s.reconnects}")
    print(f"book gaps  {book.stats.gaps_detected}")
    print(f"crossed    {book.stats.crossed_seen}")
    print(f"\n{client.latency.render()}")
    print(f"\nsaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
