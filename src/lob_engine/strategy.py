"""Inventory-aware market making.

Implements the quoting rule from:

    Marco Avellaneda and Sasha Stoikov (2006),
    "High-frequency trading in a limit order book".

The problem a market maker actually has is not "what is the fair price".
It is "I am long 400 units and the price could move against me before I
get flat". A dealer who quotes symmetrically around the mid has no
mechanism to get flat, so inventory random-walks away and the P&L
distribution develops fat tails that have nothing to do with skill.

The fix is to quote around a **reservation price** rather than the mid:

    r(s, q, t) = s - q * gamma * sigma^2 * (T - t)

where q is current inventory, gamma is risk aversion, sigma is volatility
and (T - t) is time left. Long inventory pushes the reservation price
down, so both quotes shift down, so you are more likely to be lifted than
hit, so inventory mean-reverts toward zero. The skew is the control.

The total spread is then

    delta_a + delta_b = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / k)

The first term is compensation for inventory risk over the remaining
horizon. The second is the monopolistic-pricing term from the order
arrival model: fills arrive with intensity lambda(delta) = A * exp(-k *
delta), so quoting wider earns more per fill but fills less often, and
this is the optimum of that trade-off.

The paper's headline result is not higher mean P&L. It is **lower
variance** of P&L and of final inventory. That is what the simulation
here measures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ASParams:
    """Avellaneda-Stoikov model parameters."""

    gamma: float = 0.1  # risk aversion
    sigma: float = 2.0  # volatility of the mid, price units per sqrt(time)
    k: float = 1.5  # order book liquidity parameter in lambda = A e^{-k delta}
    A: float = 140.0  # base arrival intensity
    T: float = 1.0  # terminal time (end of the trading session)


def reservation_price(mid: float, inventory: float, t: float, p: ASParams) -> float:
    """Indifference price given current inventory and time remaining."""
    return mid - inventory * p.gamma * p.sigma**2 * max(0.0, p.T - t)


def optimal_spread(t: float, p: ASParams) -> float:
    """Total bid-ask spread the dealer should quote."""
    horizon = max(0.0, p.T - t)
    return p.gamma * p.sigma**2 * horizon + (2.0 / p.gamma) * math.log(
        1.0 + p.gamma / p.k
    )


def as_quotes(mid: float, inventory: float, t: float, p: ASParams) -> tuple[float, float]:
    """(bid, ask) for the inventory-aware strategy."""
    r = reservation_price(mid, inventory, t, p)
    half = optimal_spread(t, p) / 2.0
    return r - half, r + half


def symmetric_quotes(
    mid: float, inventory: float, t: float, p: ASParams
) -> tuple[float, float]:
    """(bid, ask) for the benchmark: fixed spread, always centred on the mid.

    Uses the same average spread as the inventory strategy so the
    comparison is about *skew*, not about who quotes tighter. Quoting a
    different width would confound the two effects.
    """
    half = optimal_spread(0.5 * p.T, p) / 2.0
    return mid - half, mid + half


@dataclass(slots=True)
class RunResult:
    """Outcome of one simulated session."""

    pnl: float
    final_inventory: float
    n_trades: int
    max_abs_inventory: float


@dataclass(slots=True)
class Summary:
    """Distribution of outcomes across many sessions."""

    name: str
    pnl_mean: float
    pnl_std: float
    inv_mean: float
    inv_std: float
    trades_mean: float
    max_inv_mean: float
    pnl_p05: float
    pnl_p95: float

    def as_row(self) -> str:
        return (
            f"{self.name:<22}{self.pnl_mean:>10.2f}{self.pnl_std:>10.2f}"
            f"{self.pnl_p05:>10.2f}{self.pnl_p95:>10.2f}"
            f"{self.inv_std:>15.2f}{self.max_inv_mean:>14.2f}"
            f"{self.trades_mean:>10.1f}"
        )

    @staticmethod
    def header() -> str:
        return (
            f"{'strategy':<22}{'P&L mean':>10}{'P&L std':>10}"
            f"{'P&L p05':>10}{'P&L p95':>10}{'std final |q|':>15}"
            f"{'mean max |q|':>14}{'trades':>10}\n" + "-" * 101
        )


def run_session(
    quote_fn,
    p: ASParams,
    steps: int = 200,
    rng: np.random.Generator | None = None,
    inventory_limit: float | None = None,
) -> RunResult:
    """Simulate one trading session.

    The mid follows an arithmetic random walk. At each step the dealer
    posts a two-sided quote; a buy order lifting our ask arrives with
    probability lambda_a * dt where lambda_a = A exp(-k * delta_a), and
    symmetrically for the bid. Filled size is one unit, as in the paper.

    Cash is tracked explicitly and P&L is marked at the terminal mid, so
    an unwound inventory is valued honestly rather than assumed away.
    """
    rng = rng or np.random.default_rng()
    dt = p.T / steps
    mid = 100.0
    inventory = 0.0
    cash = 0.0
    trades = 0
    max_abs = 0.0

    for i in range(steps):
        t = i * dt
        bid, ask = quote_fn(mid, inventory, t, p)
        delta_b = max(0.0, mid - bid)
        delta_a = max(0.0, ask - mid)

        lam_b = p.A * math.exp(-p.k * delta_b)
        lam_a = p.A * math.exp(-p.k * delta_a)

        # Probability of at least one arrival in dt, for a Poisson flow.
        hit_bid = rng.random() < 1.0 - math.exp(-lam_b * dt)
        lift_ask = rng.random() < 1.0 - math.exp(-lam_a * dt)

        if hit_bid and (inventory_limit is None or inventory < inventory_limit):
            inventory += 1.0
            cash -= bid
            trades += 1
        if lift_ask and (inventory_limit is None or inventory > -inventory_limit):
            inventory -= 1.0
            cash += ask
            trades += 1

        max_abs = max(max_abs, abs(inventory))
        mid += p.sigma * math.sqrt(dt) * rng.standard_normal()

    return RunResult(
        pnl=cash + inventory * mid,
        final_inventory=inventory,
        n_trades=trades,
        max_abs_inventory=max_abs,
    )


def compare(
    p: ASParams | None = None,
    n_runs: int = 2000,
    steps: int = 200,
    seed: int = 20260913,
) -> list[Summary]:
    """Run both strategies over the same set of random paths.

    The same seed sequence is used for both so the comparison is paired:
    any difference is the strategy, not the draw.
    """
    p = p or ASParams()
    out: list[Summary] = []
    for name, fn in (("inventory-aware (A-S)", as_quotes), ("symmetric benchmark", symmetric_quotes)):
        results = [
            run_session(fn, p, steps=steps, rng=np.random.default_rng(seed + i))
            for i in range(n_runs)
        ]
        pnls = np.array([r.pnl for r in results])
        invs = np.array([r.final_inventory for r in results])
        out.append(
            Summary(
                name=name,
                pnl_mean=float(pnls.mean()),
                pnl_std=float(pnls.std(ddof=1)),
                inv_mean=float(invs.mean()),
                inv_std=float(invs.std(ddof=1)),
                trades_mean=float(np.mean([r.n_trades for r in results])),
                max_inv_mean=float(np.mean([r.max_abs_inventory for r in results])),
                pnl_p05=float(np.percentile(pnls, 5)),
                pnl_p95=float(np.percentile(pnls, 95)),
            )
        )
    return out
