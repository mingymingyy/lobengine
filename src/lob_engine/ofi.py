"""Order Flow Imbalance (OFI).

Implements the estimator from:

    Rama Cont, Arseniy Kukanov and Sasha Stoikov (2011),
    "The price impact of order book events", arXiv:1011.6402v3.

The idea in one line: everything that can move the price on a short
horizon shows up as a change in the size or price of the best bid and
best ask, so you can measure supply/demand pressure from top-of-book
alone, without a trade tape.

For consecutive top-of-book observations n-1 and n the paper defines the
contribution of event n as

    e_n = 1{P_b(n) >= P_b(n-1)} * q_b(n)
        - 1{P_b(n) <= P_b(n-1)} * q_b(n-1)
        - 1{P_a(n) <= P_a(n-1)} * q_a(n)
        + 1{P_a(n) >= P_a(n-1)} * q_a(n-1)

Read it case by case:

    bid price unchanged  -> both indicators fire -> e = q_b(n) - q_b(n-1)
                            (size added to, or taken from, the bid)
    bid price improves   -> e = +q_b(n)      (new, better bid posted)
    bid price falls      -> e = -q_b(n-1)    (the whole bid level went away)
    ask price unchanged  -> e = -(q_a(n) - q_a(n-1))
    ask price improves   -> e = -q_a(n)      (new, better offer posted)
    ask price rises      -> e = +q_a(n-1)    (the whole ask level went away)

OFI over a time bucket is just the sum of e_n inside it. The paper's
central empirical claim is that the mid-price change over the bucket is
close to linear in OFI, with a slope inversely proportional to market
depth, and that this holds up better than the corresponding relationship
using signed trade volume.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import TopOfBook


def event_contribution(prev: TopOfBook, curr: TopOfBook) -> float:
    """`e_n` for one pair of consecutive top-of-book observations.

    One-sided books contribute 0: with no bid or no ask there is no
    well-defined imbalance, and folding them in would fabricate flow.
    """
    if not (prev.is_two_sided and curr.is_two_sided):
        return 0.0

    e = 0.0
    # bid side (demand)
    if curr.bid_px >= prev.bid_px:
        e += curr.bid_sz
    if curr.bid_px <= prev.bid_px:
        e -= prev.bid_sz
    # ask side (supply), signs reversed
    if curr.ask_px <= prev.ask_px:
        e -= curr.ask_sz
    if curr.ask_px >= prev.ask_px:
        e += prev.ask_sz
    return e


class OFIAccumulator:
    """Streaming `e_n` accumulator.

    Feed it every top-of-book you see. It keeps the running sum of
    contributions and lets you drain it at the end of each bucket.
    """

    __slots__ = ("_prev", "_sum", "_events")

    def __init__(self) -> None:
        self._prev: TopOfBook | None = None
        self._sum = 0.0
        self._events = 0

    def update(self, tob: TopOfBook) -> float:
        """Add one observation. Returns this observation's contribution."""
        if self._prev is None:
            self._prev = tob
            return 0.0
        e = event_contribution(self._prev, tob)
        # Only count it as an event if the top of book actually moved.
        if (
            tob.bid_px != self._prev.bid_px
            or tob.ask_px != self._prev.ask_px
            or tob.bid_sz != self._prev.bid_sz
            or tob.ask_sz != self._prev.ask_sz
        ):
            self._events += 1
        self._sum += e
        self._prev = tob
        return e

    @property
    def value(self) -> float:
        return self._sum

    @property
    def events(self) -> int:
        return self._events

    def drain(self) -> tuple[float, int]:
        """Return (OFI, event count) for the bucket and reset the sums.

        The last observation is retained so the next bucket's first
        contribution is measured against it rather than starting blind.
        """
        out = (self._sum, self._events)
        self._sum = 0.0
        self._events = 0
        return out


@dataclass(slots=True)
class Bucket:
    """One time bucket of the regression sample."""

    start_ns: int
    end_ns: int
    ofi: float
    delta_mid: float  # in ticks
    avg_depth: float  # average of top-of-book bid and ask size
    events: int
    mid_start: float
    mid_end: float


def bucket_stream(
    observations: list[TopOfBook],
    interval_ns: int,
    depths: list[float] | None = None,
) -> list[Bucket]:
    """Turn a stream of top-of-book observations into regression buckets.

    Parameters
    ----------
    observations
        Top-of-book snapshots in receive-time order.
    interval_ns
        Bucket width. The paper uses intervals from 1 to 60 seconds.
    depths
        Optional per-observation average depth (e.g. mean of top-5 sizes,
        taken from the full book). Must align with `observations`. When
        omitted, average depth is taken from top-of-book sizes only.

    Notes
    -----
    Buckets that contain no two-sided observation, or where the mid is
    undefined at either end, are dropped rather than zero-filled. Zero
    filling would put a spurious mass of (0, 0) points at the origin and
    inflate R-squared.
    """
    if depths is not None and len(depths) != len(observations):
        raise ValueError("depths must align with observations")
    if interval_ns <= 0:
        raise ValueError("interval_ns must be positive")

    buckets: list[Bucket] = []
    if not observations:
        return buckets

    acc = OFIAccumulator()
    start = observations[0].recv_ts_ns
    bucket_end = start + interval_ns
    mid_start: float | None = observations[0].mid
    depth_samples: list[float] = []

    def close(end_ns: int, mid_end: float | None, b_start: int) -> None:
        nonlocal mid_start, depth_samples
        ofi, n_events = acc.drain()
        if mid_start is not None and mid_end is not None and depth_samples:
            buckets.append(
                Bucket(
                    start_ns=b_start,
                    end_ns=end_ns,
                    ofi=ofi,
                    delta_mid=mid_end - mid_start,
                    avg_depth=float(np.mean(depth_samples)),
                    events=n_events,
                    mid_start=mid_start,
                    mid_end=mid_end,
                )
            )
        mid_start = mid_end
        depth_samples = []

    last_mid: float | None = observations[0].mid
    bucket_start = start

    for i, tob in enumerate(observations):
        while tob.recv_ts_ns >= bucket_end:
            close(bucket_end, last_mid, bucket_start)
            bucket_start = bucket_end
            bucket_end += interval_ns
        acc.update(tob)
        if tob.is_two_sided:
            last_mid = tob.mid
            if depths is not None:
                depth_samples.append(depths[i])
            else:
                depth_samples.append((tob.bid_sz + tob.ask_sz) / 2.0)
            if mid_start is None:
                mid_start = tob.mid

    close(observations[-1].recv_ts_ns, last_mid, bucket_start)
    return buckets
