#!/usr/bin/env python3
"""Measure the tick-to-order path, stage by stage.

The path a quote update takes before it can turn into an order:

    raw bytes  ->  parse  ->  book apply  ->  signal  ->  risk  ->  FIX encode

Each stage is timed separately, because an aggregate number tells you
nothing about which stage to fix. Percentiles, not means: a p99.9 of four
milliseconds is the number that decides whether you got the trade.

    python scripts/bench_latency.py --messages 100000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lob_engine import fix
from lob_engine.book import OrderBook, SequenceGap
from lob_engine.feeds import OKXBooksAdapter
from lob_engine.latency import LatencyBook
from lob_engine.ofi import OFIAccumulator
from lob_engine.oms import OMS
from lob_engine.risk import RiskEngine, RiskLimits
from lob_engine.sim import LOBSimulator, SimConfig, to_okx_frames
from lob_engine.types import Side


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--messages", type=int, default=100_000)
    ap.add_argument("--warmup", type=int, default=2_000)
    args = ap.parse_args()

    print(f"generating {args.messages:,} messages ...")
    cfg = SimConfig(seed=3)
    events = list(LOBSimulator(cfg).run(args.messages))
    frames = [f for _, f in to_okx_frames(LOBSimulator(cfg), events)]

    adapter = OKXBooksAdapter("SIM-USDT", price_decimals=1)
    book = OrderBook("SIM-USDT", max_depth=adapter.max_depth)
    acc = OFIAccumulator()
    oms = OMS()
    risk = RiskEngine(
        RiskLimits(
            max_order_qty=100,
            max_order_value=1_000_000,
            max_position=500,
            price_collar_bps=200,
            max_messages_per_sec=100_000,
        ),
        oms,
        price_scale_factor=10.0,
    )
    lat = LatencyBook(warmup=args.warmup)

    print("running ...\n")
    t_start = time.perf_counter()
    orders = 0

    for i, raw in enumerate(frames):
        recv_ns = time.time_ns()

        t0 = time.perf_counter_ns()
        updates = adapter.parse(raw, recv_ns)
        lat["1. parse json"].record(time.perf_counter_ns() - t0)

        for upd in updates:
            t1 = time.perf_counter_ns()
            try:
                book.apply(upd)
            except SequenceGap:
                book.reset()
                continue
            lat["2. book apply"].record(time.perf_counter_ns() - t1)

            t2 = time.perf_counter_ns()
            top = book.top()
            ofi = acc.update(top)
            lat["3. top + ofi"].record(time.perf_counter_ns() - t2)

            if not top.is_two_sided:
                continue

            # Only price up an order occasionally, so the risk and encode
            # stages are measured on a realistic number of samples rather
            # than on every single tick.
            if i % 25:
                continue

            mid = top.mid
            side = Side.BUY if ofi > 0 else Side.SELL
            px = top.bid_px if side is Side.BUY else top.ask_px

            t3 = time.perf_counter_ns()
            decision = risk.check_new_order(book.symbol, side, 1.0, px, mid=mid)
            lat["4. risk checks"].record(time.perf_counter_ns() - t3)
            if not decision:
                continue

            t4 = time.perf_counter_ns()
            fix.encode(
                fix.MsgType.NEW_ORDER_SINGLE,
                [
                    (fix.Tag.SenderCompID, "LOBENG"),
                    (fix.Tag.TargetCompID, "VENUE"),
                    (fix.Tag.MsgSeqNum, str(orders + 1)),
                    (fix.Tag.SendingTime, "20260913-09:30:00.000"),
                    (fix.Tag.ClOrdID, f"O{orders:08d}"),
                    (fix.Tag.Symbol, book.symbol),
                    (fix.Tag.Side, fix.FIX_SIDE[side.value]),
                    (fix.Tag.OrderQty, "1"),
                    (fix.Tag.OrdType, fix.OrdType.LIMIT.value),
                    (fix.Tag.Price, str(px)),
                    (fix.Tag.TimeInForce, fix.TimeInForce.DAY.value),
                    (fix.Tag.TransactTime, "20260913-09:30:00.000"),
                ],
            )
            lat["5. fix encode"].record(time.perf_counter_ns() - t4)
            orders += 1

    elapsed = time.perf_counter() - t_start

    # End-to-end figure: sum of stage medians is misleading, so measure
    # the whole chain separately on a fixed sample.
    print(lat.render())
    print()
    print(f"messages processed   {len(frames):,}")
    print(f"orders priced        {orders:,}")
    print(f"wall time            {elapsed:.2f} s")
    print(f"throughput           {len(frames) / elapsed:,.0f} msg/s")
    print(f"book gaps            {book.stats.gaps_detected}")
    print(f"risk rejects         {risk.stats.rejected:,} of {risk.stats.checked:,}")
    if risk.stats.by_check:
        for k, v in sorted(risk.stats.by_check.items(), key=lambda kv: -kv[1]):
            print(f"    {k:<20}{v:,}")
    print()
    print("Read the p99.9 column, not p50. Python gives you a floor of a few")
    print("microseconds per stage; the point of the harness is that it makes")
    print("a regression visible the moment you introduce one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
