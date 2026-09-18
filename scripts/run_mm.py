#!/usr/bin/env python3
"""Compare inventory-aware quoting against a symmetric benchmark.

Reproduces the central comparison in Avellaneda & Stoikov (2006): the
inventory-aware dealer does not make more money on average, but the
distribution of outcomes is far tighter.

    python scripts/run_mm.py --runs 5000 --plot docs/mm_pnl.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lob_engine.strategy import (
    ASParams,
    Summary,
    as_quotes,
    compare,
    run_session,
    symmetric_quotes,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--gamma", type=float, default=0.1, help="risk aversion")
    ap.add_argument("--sigma", type=float, default=2.0, help="mid volatility")
    ap.add_argument("--k", type=float, default=1.5, help="book liquidity")
    ap.add_argument("--A", type=float, default=140.0, help="arrival intensity")
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    p = ASParams(gamma=args.gamma, sigma=args.sigma, k=args.k, A=args.A)
    print(
        f"gamma={p.gamma}  sigma={p.sigma}  k={p.k}  A={p.A}  "
        f"runs={args.runs:,}  steps={args.steps}\n"
    )

    results = compare(p, n_runs=args.runs, steps=args.steps, seed=args.seed)
    print(Summary.header())
    for r in results:
        print(r.as_row())

    inv, sym = results
    print(
        f"\nP&L standard deviation is "
        f"{(1 - inv.pnl_std / sym.pnl_std) * 100:.0f}% lower with inventory skew."
    )
    print(
        f"Final inventory standard deviation is "
        f"{(1 - inv.inv_std / sym.inv_std) * 100:.0f}% lower."
    )
    print(
        f"Mean P&L is {(inv.pnl_mean / sym.pnl_mean - 1) * 100:+.0f}%: "
        "the skew is paid for, it is not free."
    )

    if args.plot:
        plot(p, args, Path(args.plot))
        print(f"\nwrote {args.plot}")
    return 0


def plot(p: ASParams, args, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = {}
    for name, fn in (("inventory-aware", as_quotes), ("symmetric", symmetric_quotes)):
        series[name] = np.array(
            [
                run_session(
                    fn, p, steps=args.steps, rng=np.random.default_rng(args.seed + i)
                ).pnl
                for i in range(args.runs)
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    bins = np.linspace(
        min(s.min() for s in series.values()),
        max(s.max() for s in series.values()),
        70,
    )
    for (name, vals), color in zip(series.items(), ("#1f4e79", "#c0392b"), strict=True):
        ax.hist(
            vals, bins=bins, alpha=0.55, label=f"{name} (sd {vals.std(ddof=1):.1f})",
            color=color, edgecolor="none",
        )
    ax.set_xlabel("terminal P&L")
    ax.set_ylabel("sessions")
    ax.set_title("Same expected edge, very different tails")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
