#!/usr/bin/env python3
"""Walk one order through the whole stack, printing each step.

This is the script to run first. It shows, in order:

  * a FIX session logging on, with sequence numbers
  * a NewOrderSingle built, framed and decoded
  * pre-trade risk rejecting bad orders and passing a good one
  * the venue matching it with price-time priority
  * queue position, and why it decides whether you get filled
  * the OMS tracking state, position and P&L

    python scripts/demo_oms.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lob_engine import fix
from lob_engine.fix import OrdType, TimeInForce
from lob_engine.oms import OMS
from lob_engine.risk import RiskEngine, RiskLimits
from lob_engine.session import FixSession, SessionConfig
from lob_engine.types import Side
from lob_engine.venue import MockVenue

RULE = "=" * 72


def head(n: int, title: str) -> None:
    print(f"\n{RULE}\n{n}. {title}\n{RULE}")


def show_fix(raw: str, label: str) -> None:
    print(f"{label}\n  {raw.replace(fix.SOH, '|')}")


def main() -> int:
    # ------------------------------------------------------------------ #
    head(1, "FIX session")
    session = FixSession(
        SessionConfig(sender_comp_id="LOBENG", target_comp_id="SIMVENUE",
                      heartbeat_interval=30)
    )
    show_fix(session.logon(), "outbound Logon (35=A)")

    ack = fix.encode(
        fix.MsgType.LOGON,
        [
            (fix.Tag.SenderCompID, "SIMVENUE"),
            (fix.Tag.TargetCompID, "LOBENG"),
            (fix.Tag.MsgSeqNum, "1"),
            (fix.Tag.SendingTime, fix.utc_timestamp()),
            (fix.Tag.EncryptMethod, "0"),
            (fix.Tag.HeartBtInt, "30"),
        ],
    )
    session.on_message(fix.decode(ack))
    print(f"\nsession state: {session.state.name}")
    print(f"next outbound seq: {session.out_seq}, next expected inbound: {session.in_seq}")

    # A message arriving with seq 5 when we expect 2 means three went
    # missing. The only correct response is to ask for them.
    gapped = fix.encode(
        fix.MsgType.EXECUTION_REPORT,
        [
            (fix.Tag.SenderCompID, "SIMVENUE"),
            (fix.Tag.TargetCompID, "LOBENG"),
            (fix.Tag.MsgSeqNum, "5"),
            (fix.Tag.SendingTime, fix.utc_timestamp()),
            (fix.Tag.ClOrdID, "X"),
            (fix.Tag.ExecType, "0"),
            (fix.Tag.OrdStatus, "0"),
        ],
    )
    out, deliver = session.on_message(fix.decode(gapped))
    print(f"\ninbound seq 5 while expecting 2 -> delivered to app: {deliver}")
    show_fix(out[0], "automatic ResendRequest (35=2)")

    # ------------------------------------------------------------------ #
    head(2, "Venue: starting book")
    oms = OMS(id_prefix="DEMO")
    venue = MockVenue("SIMUSD", on_exec=lambda er: apply_er(oms, er))

    for px, qty in [(9998, 30), (9999, 20), (10000, 12)]:
        venue.add_liquidity(Side.BUY, px, qty)
    for px, qty in [(10001, 15), (10002, 25), (10003, 40)]:
        venue.add_liquidity(Side.SELL, px, qty)

    print_book(venue)

    # ------------------------------------------------------------------ #
    head(3, "Pre-trade risk")
    risk = RiskEngine(
        RiskLimits(
            max_order_qty=50,
            max_order_value=600_000,
            max_position=40,
            max_open_orders=5,
            price_collar_bps=100,
            max_messages_per_sec=20,
            lot_size=1,
            tick_size=1,
        ),
        oms,
        price_scale_factor=1.0,
    )
    mid = venue.mid()
    print(f"mid = {mid}\n")

    trials = [
        ("size fat finger", Side.BUY, 5_000.0, 10000),
        ("price far from mid", Side.BUY, 10.0, 8000),
        ("notional too large", Side.BUY, 40.0, 20000),
        ("good order", Side.BUY, 10.0, 10000),
    ]
    for label, side, qty, px in trials:
        d = risk.check_new_order("SIMUSD", side, qty, px, mid=mid)
        mark = "accept" if d else "REJECT"
        print(f"  {label:<22} {mark:<8} {d.reason}")

    # ------------------------------------------------------------------ #
    head(4, "Resting an order, and queue position")
    order = oms.create("SIMUSD", Side.BUY, 10.0, 10000, OrdType.LIMIT, TimeInForce.DAY)
    print(f"created {order.cl_ord_id}: buy 10 @ 10000, status {order.status.name}")
    venue.submit(order.cl_ord_id, order.side, order.qty, order.price)
    print(f"after venue ack:  status {order.status.name}, leaves {order.leaves_qty}")

    ahead = venue.queue_ahead(order.cl_ord_id)
    print(f"\nsize resting ahead of us at 10000: {ahead}")
    print("A trade printing at 10000 does not fill us until that clears.")

    print("\n-> someone sells 5 into the bid")
    venue.market_order(Side.SELL, 5)
    print(f"   queue ahead now {venue.queue_ahead(order.cl_ord_id)}, "
          f"our cum qty {order.cum_qty}")

    print("\n-> 7 more units ahead of us cancel")
    venue.cancel_ahead(Side.BUY, 10000, 7)
    print(f"   queue ahead now {venue.queue_ahead(order.cl_ord_id)}, "
          f"our cum qty {order.cum_qty}")

    print("\n-> someone sells 6 into the bid")
    venue.market_order(Side.SELL, 6)
    print(f"   our cum qty {order.cum_qty}, avg px {order.avg_px:.1f}, "
          f"status {order.status.name}")

    # ------------------------------------------------------------------ #
    head(5, "Crossing the spread")
    aggressive = oms.create("SIMUSD", Side.BUY, 30.0, 10002, OrdType.LIMIT,
                            TimeInForce.IOC)
    venue.submit(
        aggressive.cl_ord_id, aggressive.side, aggressive.qty,
        aggressive.price, TimeInForce.IOC,
    )
    print("IOC buy 30 @ 10002 (best ask was 10001)")
    print(f"  filled {aggressive.cum_qty} at avg {aggressive.avg_px:.2f}, "
          f"status {aggressive.status.name}")
    print("  walked the book: 15 at 10001 then the rest at 10002")

    # ------------------------------------------------------------------ #
    head(6, "Position and P&L")
    mark = venue.mid() or 10000
    print(f"position   {oms.position('SIMUSD'):+.0f}")
    print(f"fills      {len(oms.fills)}")
    for f in oms.fills:
        print(f"    {f.side.value:<5}{f.qty:>6.0f} @ {f.price}   {f.cl_ord_id}")
    print(f"mark       {mark}")
    print(f"P&L        {oms.realised_pnl('SIMUSD', mark):+.1f}")
    max_long, max_short = oms.exposure("SIMUSD")
    print(f"exposure   if everything working fills: "
          f"{max_short:+.0f} to {max_long:+.0f}")
    print(f"\nillegal state transitions caught: {oms.rejected_transitions}")
    print_book(venue)
    return 0


def apply_er(oms: OMS, er) -> None:
    try:
        oms.on_execution_report(er)
    except Exception as exc:  # noqa: BLE001
        print(f"  [oms] rejected report: {exc}")


def print_book(venue: MockVenue) -> None:
    print("\n        bids                asks")
    bids = sorted((p for p, q in venue.bids.items() if q), reverse=True)
    asks = sorted(p for p, q in venue.asks.items() if q)
    for i in range(max(len(bids), len(asks))):
        left = (
            f"{venue.size_at(Side.BUY, bids[i]):>8.0f} @ {bids[i]:<7}"
            if i < len(bids)
            else " " * 18
        )
        right = (
            f"{asks[i]:>7} @ {venue.size_at(Side.SELL, asks[i]):<8.0f}"
            if i < len(asks)
            else ""
        )
        print(f"  {left}  {right}")


if __name__ == "__main__":
    raise SystemExit(main())
