"""A stochastic limit order book simulator.

Why this exists: to test an order book engine and an order-flow estimator
you need order book data, and a hand-written toy sequence proves nothing.
This module generates data from a model of *order flow*, and lets the
price emerge from it. Nothing here assumes any relationship between order
flow and price. That relationship, if it shows up in the regression, is a
consequence of the mechanics, not of the generator.

The model follows the structure of:

    Rama Cont, Sasha Stoikov and Rishi Talreja (2010),
    "A stochastic model for order book dynamics", Operations Research.

Four Poisson flows on a discrete price grid:

    limit buy    arrives i ticks below the best ask, rate lambda(i)
    limit sell   arrives i ticks above the best bid, rate lambda(i)
    market buy   consumes the best ask, rate mu
    market sell  consumes the best bid, rate mu
    cancel       each resting unit cancels at rate theta

with lambda(i) = k / i**alpha, so liquidity thins out away from the
touch, which is what real books look like.

Events are drawn by the standard Gillespie method: the time to the next
event is exponential with the total rate, and the event type is drawn in
proportion to its own rate.

The output can be rendered as OKX-format JSON frames, so the simulated
data flows through exactly the same adapter, book and gap-detection code
as live data. There is no separate "test mode" path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .types import TopOfBook


@dataclass(slots=True)
class SimConfig:
    """Parameters of the order flow model."""

    n_ticks: int = 2400  # price grid size, in ticks
    initial_mid: int = 1200  # grid index of the starting mid
    depth_levels: int = 10  # how far from the touch orders arrive
    lambda_k: float = 10.0  # limit order rate scale, events/sec at i=1
    lambda_alpha: float = 1.2  # decay of arrival rate with distance
    mu: float = 5.0  # market order rate per side, events/sec
    theta: float = 1.5  # per-unit cancellation rate, per sec
    limit_size_max: int = 3  # limit order size ~ U{1..max}
    market_size_max: int = 4  # market order size ~ U{1..max}
    initial_size: float = 6.0  # starting size per level near the touch
    initial_spread: int = 2  # ticks
    seed: int = 20260913

    def lambdas(self) -> np.ndarray:
        i = np.arange(1, self.depth_levels + 1, dtype=float)
        return self.lambda_k / i**self.lambda_alpha


@dataclass(slots=True)
class SimEvent:
    """One state change, with the resulting top of book."""

    t_ns: int
    kind: str  # limit_buy | limit_sell | market_buy | market_sell | cancel_bid | cancel_ask
    price: int
    size_delta: float
    bid_px: int | None
    bid_sz: float
    ask_px: int | None
    ask_sz: float
    changed_bids: list[tuple[int, float]] = field(default_factory=list)
    changed_asks: list[tuple[int, float]] = field(default_factory=list)


class LOBSimulator:
    """Simulates one instrument's order book."""

    def __init__(self, cfg: SimConfig | None = None) -> None:
        self.cfg = cfg or SimConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        n = self.cfg.n_ticks
        self.bid = np.zeros(n)
        self.ask = np.zeros(n)
        self._lam = self.cfg.lambdas()
        self._lam_sum = float(self._lam.sum())
        self._seed_book()

    # ------------------------------------------------------------------ #
    def _seed_book(self) -> None:
        c = self.cfg
        half = c.initial_spread // 2
        best_bid = c.initial_mid - half - 1
        best_ask = c.initial_mid + half + 1
        for i in range(c.depth_levels):
            size = c.initial_size * (1.0 + 0.20 * i)
            b, a = best_bid - i, best_ask + i
            if 0 <= b < c.n_ticks:
                self.bid[b] = round(size)
            if 0 <= a < c.n_ticks:
                self.ask[a] = round(size)

    # ------------------------------------------------------------------ #
    @property
    def best_bid(self) -> int | None:
        nz = np.flatnonzero(self.bid)
        return int(nz[-1]) if nz.size else None

    @property
    def best_ask(self) -> int | None:
        nz = np.flatnonzero(self.ask)
        return int(nz[0]) if nz.size else None

    def top(self, t_ns: int = 0) -> TopOfBook:
        b, a = self.best_bid, self.best_ask
        return TopOfBook(
            bid_px=b,
            bid_sz=float(self.bid[b]) if b is not None else 0.0,
            ask_px=a,
            ask_sz=float(self.ask[a]) if a is not None else 0.0,
            recv_ts_ns=t_ns,
        )

    def average_depth(self, levels: int = 5) -> float:
        b, a = self.best_bid, self.best_ask
        if b is None or a is None:
            return 0.0
        lo = max(0, b - levels + 1)
        hi = min(self.cfg.n_ticks, a + levels)
        return float(self.bid[lo : b + 1].sum() + self.ask[a:hi].sum()) / (2 * levels)

    # ------------------------------------------------------------------ #
    def step(self, t_ns: int) -> tuple[int, SimEvent | None]:
        """Advance one event. Returns (new time in ns, event)."""
        c = self.cfg
        b, a = self.best_bid, self.best_ask
        if b is None or a is None:
            # A side emptied out. Reseed it rather than stalling; in a
            # real market a designated market maker would do the same.
            self._seed_book()
            b, a = self.best_bid, self.best_ask

        # --- build the rate vector -------------------------------------
        lam_buy_idx = np.clip(a - np.arange(1, c.depth_levels + 1), 0, c.n_ticks - 1)
        lam_sell_idx = np.clip(b + np.arange(1, c.depth_levels + 1), 0, c.n_ticks - 1)

        cancel_bid_idx = np.clip(
            b - np.arange(0, c.depth_levels), 0, c.n_ticks - 1
        )
        cancel_ask_idx = np.clip(
            a + np.arange(0, c.depth_levels), 0, c.n_ticks - 1
        )
        cancel_bid_rates = c.theta * self.bid[cancel_bid_idx]
        cancel_ask_rates = c.theta * self.ask[cancel_ask_idx]

        rates = np.concatenate(
            [
                self._lam,  # limit buys
                self._lam,  # limit sells
                [c.mu, c.mu],  # market buy, market sell
                cancel_bid_rates,
                cancel_ask_rates,
            ]
        )
        total = float(rates.sum())
        if total <= 0:
            return t_ns + 1_000_000, None

        dt = self.rng.exponential(1.0 / total)
        t_ns = t_ns + int(dt * 1e9)

        u = self.rng.random() * total
        choice = int(np.searchsorted(np.cumsum(rates), u, side="right"))
        choice = min(choice, rates.size - 1)

        L = c.depth_levels
        if choice < L:
            px = int(lam_buy_idx[choice])
            return t_ns, self._limit(px, is_bid=True, t_ns=t_ns)
        choice -= L
        if choice < L:
            px = int(lam_sell_idx[choice])
            return t_ns, self._limit(px, is_bid=False, t_ns=t_ns)
        choice -= L
        if choice == 0:
            return t_ns, self._market(is_buy=True, t_ns=t_ns)
        if choice == 1:
            return t_ns, self._market(is_buy=False, t_ns=t_ns)
        choice -= 2
        if choice < L:
            px = int(cancel_bid_idx[choice])
            return t_ns, self._cancel(px, is_bid=True, t_ns=t_ns)
        choice -= L
        px = int(cancel_ask_idx[min(choice, L - 1)])
        return t_ns, self._cancel(px, is_bid=False, t_ns=t_ns)

    # ------------------------------------------------------------------ #
    def _limit(self, px: int, is_bid: bool, t_ns: int) -> SimEvent | None:
        c = self.cfg
        book = self.bid if is_bid else self.ask
        other = self.ask if is_bid else self.bid
        if other[px] > 0:
            return None  # would cross; a real venue would match instead
        size = float(self.rng.integers(1, c.limit_size_max + 1))
        book[px] += size
        return self._event(
            "limit_buy" if is_bid else "limit_sell", px, size, is_bid, t_ns
        )

    def _market(self, is_buy: bool, t_ns: int) -> SimEvent | None:
        # A market buy consumes the ask; a market sell consumes the bid.
        book = self.ask if is_buy else self.bid
        px = self.best_ask if is_buy else self.best_bid
        if px is None or book[px] <= 0:
            return None
        size = min(
            float(self.rng.integers(1, self.cfg.market_size_max + 1)), float(book[px])
        )
        book[px] -= size
        return self._event(
            "market_buy" if is_buy else "market_sell", px, -size, not is_buy, t_ns
        )

    def _cancel(self, px: int, is_bid: bool, t_ns: int) -> SimEvent | None:
        book = self.bid if is_bid else self.ask
        if book[px] <= 0:
            return None
        size = min(1.0, float(book[px]))
        book[px] -= size
        return self._event(
            "cancel_bid" if is_bid else "cancel_ask", px, -size, is_bid, t_ns
        )

    def _event(
        self, kind: str, px: int, delta: float, is_bid: bool, t_ns: int
    ) -> SimEvent:
        top = self.top(t_ns)
        level = (px, float(self.bid[px] if is_bid else self.ask[px]))
        return SimEvent(
            t_ns=t_ns,
            kind=kind,
            price=px,
            size_delta=delta,
            bid_px=top.bid_px,
            bid_sz=top.bid_sz,
            ask_px=top.ask_px,
            ask_sz=top.ask_sz,
            changed_bids=[level] if is_bid else [],
            changed_asks=[] if is_bid else [level],
        )

    # ------------------------------------------------------------------ #
    def run(self, n_events: int, start_ns: int = 0):
        """Yield `n_events` non-degenerate events."""
        t = start_ns
        produced = 0
        guard = 0
        while produced < n_events:
            guard += 1
            if guard > n_events * 50:
                raise RuntimeError("simulator is not producing events; check config")
            t, ev = self.step(t)
            if ev is None:
                continue
            produced += 1
            yield ev


# --------------------------------------------------------------------- #
# rendering to a venue wire format
# --------------------------------------------------------------------- #
def to_okx_frames(
    sim: LOBSimulator,
    events,
    inst_id: str = "SIM-USDT",
    price_decimals: int = 1,
    snapshot_levels: int = 25,
    snapshot_every: int = 0,
    snapshot_offset: int = 0,
):
    """Render simulated events as OKX `books` channel JSON frames.

    Yields ``(t_ns, frame)`` pairs. The timestamp is emitted alongside the
    frame rather than left for the caller to zip back on, because the
    number of frames is not the number of events once `snapshot_every` is
    in play.

    Emitting a real venue format rather than a private one means the
    simulated data is parsed by the same adapter, applied by the same
    book, and gap-checked by the same code as live data.

    Parameters
    ----------
    sim
        A simulator positioned at the state *before* the first event.
        Its arrays are advanced as the events are rendered, so that
        periodic snapshots reflect the book at that moment.
    snapshot_every
        Emit a full snapshot every N events, in addition to the deltas.
        0 disables. This mirrors what a client sees after it resubscribes
        following a gap, and is what makes gap *recovery* reproducible in
        an offline replay: without a later snapshot, a stream that has
        lost a message can never be rebuilt.
    snapshot_offset
        Phase of the snapshot within each period, i.e. emit when
        ``i % snapshot_every == snapshot_offset``. Used together with a
        message-dropping harness to place the snapshot a couple of events
        *after* the drop, so the gap is genuinely detected before it is
        recovered. A snapshot landing in the same instant as the drop
        would mask it, and then the recovery path never runs.
    """
    factor = 10**price_decimals

    def px(t: int) -> str:
        return f"{t / factor:.{price_decimals}f}"

    def snapshot_rows(arr, best, descending):
        if best is None:
            return []
        idx = (
            range(best, max(-1, best - snapshot_levels), -1)
            if descending
            else range(best, min(len(arr), best + snapshot_levels))
        )
        return [[px(i), f"{arr[i]:.4f}", "0", "1"] for i in idx if arr[i] > 0]

    def frame(action, bids, asks, ts, seq, prev_seq):
        return json.dumps(
            {
                "arg": {"channel": "books", "instId": inst_id},
                "action": action,
                "data": [
                    {
                        "bids": bids,
                        "asks": asks,
                        "ts": str(ts),
                        "seqId": seq,
                        "prevSeqId": prev_seq,
                    }
                ],
            },
            separators=(",", ":"),
        )

    seq = 1
    yield 0, frame(
        "snapshot",
        snapshot_rows(sim.bid, sim.best_bid, True),
        snapshot_rows(sim.ask, sim.best_ask, False),
        0,
        seq,
        -1,
    )

    for i, ev in enumerate(events, start=1):
        # Track the book so a periodic snapshot is accurate. Each event
        # carries the absolute new size at the level it touched.
        for p, s in ev.changed_bids:
            sim.bid[p] = s
        for p, s in ev.changed_asks:
            sim.ask[p] = s

        prev, seq = seq, seq + 1
        yield ev.t_ns, frame(
            "update",
            [[px(p), f"{s:.4f}", "0", "1"] for p, s in ev.changed_bids],
            [[px(p), f"{s:.4f}", "0", "1"] for p, s in ev.changed_asks],
            ev.t_ns // 1_000_000,
            seq,
            prev,
        )

        if snapshot_every and i % snapshot_every == snapshot_offset:
            prev, seq = seq, seq + 1
            yield ev.t_ns, frame(
                "snapshot",
                snapshot_rows(sim.bid, sim.best_bid, True),
                snapshot_rows(sim.ask, sim.best_ask, False),
                ev.t_ns // 1_000_000,
                seq,
                -1,
            )
