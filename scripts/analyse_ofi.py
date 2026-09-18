#!/usr/bin/env python3
"""Replicate the price-impact result of Cont, Kukanov & Stoikov (2011).

Two tests:

  1. Mid-price change regressed on order flow imbalance, at several
     bucket widths. The paper's claim is that this is close to linear.
  2. The impact coefficient sorted by market depth. The paper's claim is
     that the slope is inversely proportional to depth.

A third regression, mid-price change on signed trade volume, is run as
the comparison the paper makes: same data, weaker relationship.

    python scripts/analyse_ofi.py --capture data/synthetic_okx.jsonl.gz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lob_engine.book import OrderBook
from lob_engine.feeds import ADAPTERS, replay
from lob_engine.ofi import bucket_stream
from lob_engine.recorder import capture_header, read_capture
from lob_engine.stats import ols


def load(capture: Path, venue: str, symbol: str | None, decimals: int):
    header = capture_header(capture)
    venue = venue or header.get("venue", "okx")
    venue = venue.replace("-simulated", "")
    symbol = symbol or header.get("symbol", "UNKNOWN")

    adapter = ADAPTERS[venue](symbol=symbol, price_decimals=decimals)
    book = OrderBook(symbol, max_depth=adapter.max_depth)

    tops: list = []
    depths: list[float] = []

    def on_book(b: OrderBook) -> None:
        t = b.top()
        if t.is_two_sided:
            tops.append(t)
            depths.append(b.average_depth(5))

    _, stats = replay(adapter, read_capture(capture), book=book, on_book=on_book)
    return tops, depths, stats, book


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", default="data/synthetic_okx.jsonl.gz")
    ap.add_argument("--venue", default=None)
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--decimals", type=int, default=1)
    ap.add_argument(
        "--intervals",
        default="0.5,1,5,10",
        help="bucket widths in seconds, comma separated",
    )
    ap.add_argument("--plot", default=None, help="write a scatter plot to this path")
    args = ap.parse_args()

    capture = Path(args.capture)
    if not capture.exists():
        print(f"no capture at {capture}")
        print("run: python scripts/gen_synthetic.py")
        return 1

    print(f"replaying {capture} ...")
    tops, depths, stats, book = load(capture, args.venue, args.symbol, args.decimals)
    print(
        f"  {stats.messages:,} messages -> {stats.updates:,} book updates, "
        f"{stats.resyncs} resyncs, {stats.parse_errors} parse errors"
    )
    print(f"  book integrity ok: {book.check_integrity()}   "
          f"crossed seen: {book.stats.crossed_seen}   "
          f"gaps: {book.stats.gaps_detected}")
    if len(tops) < 100:
        print("not enough two-sided observations to regress")
        return 1

    span_s = (tops[-1].recv_ts_ns - tops[0].recv_ts_ns) / 1e9
    print(f"  {len(tops):,} two-sided observations over {span_s:,.1f} s\n")

    # ---------------------------------------------------------------- #
    print("1. mid-price change on order flow imbalance")
    print(f"   {'interval':<12}{'buckets':>9}{'R^2':>9}{'beta':>12}"
          f"{'t (HAC)':>10}{'mean depth':>12}")
    print("   " + "-" * 64)

    best = None
    for iv in [float(x) for x in args.intervals.split(",")]:
        buckets = [b for b in bucket_stream(tops, int(iv * 1e9), depths=depths) if b.events]
        if len(buckets) < 30:
            print(f"   {iv:<12.2f}{len(buckets):>9}   too few buckets")
            continue
        x = np.array([b.ofi for b in buckets])
        y = np.array([b.delta_mid for b in buckets])
        d = np.array([b.avg_depth for b in buckets])
        r = ols(y, x)
        print(
            f"   {str(iv) + 's':<12}{len(buckets):>9}{r.r_squared:>9.3f}"
            f"{r.slope:>12.5f}{r.slope_t:>10.1f}{d.mean():>12.2f}"
        )
        if best is None or abs(iv - 1.0) < abs(best[0] - 1.0):
            best = (iv, buckets, x, y, d, r)

    if best is None:
        print("\nno usable interval")
        return 1

    iv, buckets, x, y, d, r = best
    print(f"\n   full output at the {iv}s interval:")
    for line in r.summary(["const", "OFI"]).splitlines():
        print("   " + line)

    # ---------------------------------------------------------------- #
    print("\n2. impact coefficient by depth quintile (paper predicts beta ~ 1/depth)")
    qs = np.quantile(d, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    print(f"   {'quintile':<12}{'mean depth':>12}{'beta':>12}{'R^2':>9}{'n':>8}")
    print("   " + "-" * 53)
    mids, betas = [], []
    for i in range(5):
        mask = (d >= qs[i]) & (d <= qs[i + 1]) if i == 4 else (d >= qs[i]) & (d < qs[i + 1])
        if mask.sum() < 30:
            continue
        ri = ols(y[mask], x[mask])
        if ri.slope <= 0:
            continue
        mids.append(float(d[mask].mean()))
        betas.append(ri.slope)
        print(
            f"   Q{i + 1:<11}{d[mask].mean():>12.2f}{ri.slope:>12.5f}"
            f"{ri.r_squared:>9.3f}{mask.sum():>8}"
        )
    if len(mids) >= 3:
        el = ols(np.log(betas), np.log(mids))
        print(f"\n   elasticity of beta with respect to depth: {el.slope:>6.2f} "
              f"(R^2 {el.r_squared:.3f})")
        print("   the paper's model implies a value near -1.00")

    # ---------------------------------------------------------------- #
    print("\n3. comparison: mid-price change on signed trade volume")
    signed_volume = signed_trade_volume(tops, buckets)
    if signed_volume is not None:
        rv = ols(y, signed_volume)
        print(f"   R^2 = {rv.r_squared:.3f}   beta = {rv.slope:.5f}   "
              f"t = {rv.slope_t:.1f}")
        print(f"   OFI R^2 was {r.r_squared:.3f}; the paper finds the same ordering")
    else:
        print("   skipped: this capture has no trade tape")

    if args.plot:
        make_plot(x, y, Path(args.plot), iv, r)
        print(f"\nwrote {args.plot}")
    return 0


def signed_trade_volume(tops, buckets):
    """Approximate signed trade volume from top-of-book decreases.

    A real implementation uses the venue's trade tape. Book-only feeds do
    not distinguish a market order from a cancellation, so this proxy
    counts every decrease in size at an unchanged best price as a trade.
    It systematically overstates volume, which is exactly the point the
    paper makes: trade-based measures are noisier than OFI.
    """
    vols: list[float] = []
    idx = 0
    for b in buckets:
        v = 0.0
        prev = None
        while idx < len(tops) and tops[idx].recv_ts_ns < b.end_ns:
            t = tops[idx]
            if prev is not None:
                if t.bid_px == prev.bid_px and t.bid_sz < prev.bid_sz:
                    v -= prev.bid_sz - t.bid_sz  # bid hit: seller initiated
                if t.ask_px == prev.ask_px and t.ask_sz < prev.ask_sz:
                    v += prev.ask_sz - t.ask_sz  # ask lifted: buyer initiated
            prev = t
            idx += 1
        vols.append(v)
    return np.array(vols) if vols else None


def make_plot(x, y, path: Path, interval: float, result) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    ax.scatter(x, y, s=6, alpha=0.3, edgecolors="none", color="#1f4e79")
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, result.intercept + result.slope * xs, color="#c0392b", lw=1.6,
            label=f"fit: slope {result.slope:.4f}, $R^2$ {result.r_squared:.3f}")
    ax.axhline(0, color="#999", lw=0.6)
    ax.axvline(0, color="#999", lw=0.6)
    ax.set_xlabel(f"order flow imbalance over {interval}s")
    ax.set_ylabel("mid-price change (ticks)")
    ax.set_title("Price impact of order flow imbalance")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
