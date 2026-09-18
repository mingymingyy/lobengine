"""Simulator, strategy and whole-pipeline tests.

The end-to-end test is the one that matters most: it proves the OFI
result is produced by the real parser, the real book and the real
regression code, not by a shortcut.
"""

import numpy as np
import pytest

from lob_engine.feeds import OKXBooksAdapter, replay
from lob_engine.ofi import bucket_stream
from lob_engine.sim import LOBSimulator, SimConfig, to_okx_frames
from lob_engine.stats import ols
from lob_engine.strategy import (
    ASParams,
    as_quotes,
    compare,
    optimal_spread,
    reservation_price,
    run_session,
    symmetric_quotes,
)


# --------------------------------------------------------------------- #
# simulator
# --------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def events():
    return list(LOBSimulator(SimConfig(seed=42)).run(20_000))


@pytest.fixture(scope="module")
def buckets():
    """One shared run of the whole pipeline, reused across assertions."""
    cfg = SimConfig(seed=11)
    events = list(LOBSimulator(cfg).run(60_000))
    frames = list(to_okx_frames(LOBSimulator(cfg), events))
    adapter = OKXBooksAdapter("SIM-USDT", price_decimals=1)
    tops, depths = [], []

    def on_book(b):
        t = b.top()
        if t.is_two_sided:
            tops.append(t)
            depths.append(b.average_depth(5))

    replay(adapter, frames, on_book=on_book)
    return [b for b in bucket_stream(tops, int(1e9), depths=depths) if b.events]


class TestSimulator:
    def test_produces_the_requested_number(self, events):
        assert len(events) == 20_000

    def test_time_is_monotonic(self, events):
        ts = [e.t_ns for e in events]
        assert ts == sorted(ts)
        assert ts[-1] > 0

    def test_book_never_crosses(self, events):
        for e in events:
            if e.bid_px is not None and e.ask_px is not None:
                assert e.bid_px < e.ask_px

    def test_sizes_are_non_negative(self, events):
        assert all(e.bid_sz >= 0 and e.ask_sz >= 0 for e in events)

    def test_all_four_flows_occur(self, events):
        kinds = {e.kind for e in events}
        assert {"limit_buy", "limit_sell", "market_buy", "market_sell"} <= kinds
        assert {"cancel_bid", "cancel_ask"} <= kinds

    def test_price_actually_moves(self, events):
        """A simulator whose price never moves cannot test price impact."""
        mids = [
            (e.bid_px + e.ask_px) / 2
            for e in events
            if e.bid_px is not None and e.ask_px is not None
        ]
        assert max(mids) - min(mids) > 5

    def test_spread_is_usually_tight(self, events):
        spreads = [
            e.ask_px - e.bid_px
            for e in events
            if e.bid_px is not None and e.ask_px is not None
        ]
        assert np.median(spreads) <= 2

    def test_reproducible(self):
        a = list(LOBSimulator(SimConfig(seed=1)).run(500))
        b = list(LOBSimulator(SimConfig(seed=1)).run(500))
        assert [e.t_ns for e in a] == [e.t_ns for e in b]
        assert [e.kind for e in a] == [e.kind for e in b]

    def test_different_seeds_diverge(self):
        a = list(LOBSimulator(SimConfig(seed=1)).run(500))
        b = list(LOBSimulator(SimConfig(seed=2)).run(500))
        assert [e.t_ns for e in a] != [e.t_ns for e in b]

    def test_deeper_flow_gives_a_deeper_book(self):
        thin = LOBSimulator(SimConfig(seed=5, lambda_k=6.0))
        thick = LOBSimulator(SimConfig(seed=5, lambda_k=20.0, theta=1.0))
        for s in (thin, thick):
            list(s.run(8000))
        assert thick.average_depth(5) > thin.average_depth(5)


class TestWireRendering:
    def test_frames_round_trip_through_the_real_adapter(self):
        cfg = SimConfig(seed=3)
        events = list(LOBSimulator(cfg).run(2000))
        frames = list(to_okx_frames(LOBSimulator(cfg), events))
        assert len(frames) == len(events) + 1  # snapshot + one per event

        adapter = OKXBooksAdapter("SIM-USDT", price_decimals=1)
        book, stats = replay(adapter, frames)
        assert stats.parse_errors == 0
        assert stats.resyncs == 0
        assert stats.updates == len(frames)
        assert book.check_integrity()

    def test_replayed_book_matches_the_simulator(self):
        """The book rebuilt from the wire must equal the simulator's own."""
        cfg = SimConfig(seed=8)
        sim = LOBSimulator(cfg)
        events = list(sim.run(5000))
        frames = list(to_okx_frames(LOBSimulator(cfg), events))
        adapter = OKXBooksAdapter("SIM-USDT", price_decimals=1)
        book, _ = replay(adapter, frames)
        t = book.top()
        assert t.bid_px == sim.best_bid
        assert t.ask_px == sim.best_ask
        assert t.bid_sz == pytest.approx(sim.bid[sim.best_bid])
        assert t.ask_sz == pytest.approx(sim.ask[sim.best_ask])

    def test_gap_detection_fires_on_a_dropped_frame(self):
        cfg = SimConfig(seed=4)
        events = list(LOBSimulator(cfg).run(500))
        frames = list(to_okx_frames(LOBSimulator(cfg), events))
        kept = [m for i, m in enumerate(frames) if i == 0 or i % 50]
        adapter = OKXBooksAdapter("SIM-USDT", price_decimals=1)
        _, stats = replay(adapter, kept)
        assert stats.resyncs > 0

    def test_periodic_snapshots_let_the_book_recover(self):
        """Drop messages, then check the book detects every gap and is
        rebuilt correctly by the following snapshot."""
        cfg = SimConfig(seed=12)
        events = list(LOBSimulator(cfg).run(6000))
        frames = list(
            to_okx_frames(
                LOBSimulator(cfg), events, snapshot_every=200, snapshot_offset=2
            )
        )
        kept, update_idx, dropped = [], 0, 0
        for t_ns, frame in frames:
            is_snapshot = '"action":"snapshot"' in frame
            if not is_snapshot:
                update_idx += 1
                if update_idx % 200 == 0:
                    dropped += 1
                    continue
            kept.append((t_ns, frame))

        adapter = OKXBooksAdapter("SIM-USDT", price_decimals=1)
        book, stats = replay(adapter, kept)

        assert dropped > 0
        assert stats.resyncs >= dropped, "every dropped message must be caught"
        # recovery works: the overwhelming majority of updates still land
        assert stats.updates > 0.95 * len(kept)
        assert book.check_integrity()
        assert book.stats.crossed_seen == 0

    def test_periodic_snapshot_matches_the_delta_applied_book(self):
        """A snapshot mid-stream must agree with the book built from
        deltas, or the generator is lying about the state."""
        cfg = SimConfig(seed=13)
        events = list(LOBSimulator(cfg).run(3000))
        frames = list(
            to_okx_frames(LOBSimulator(cfg), events, snapshot_every=500)
        )
        adapter = OKXBooksAdapter("SIM-USDT", price_decimals=1)
        with_snaps, _ = replay(adapter, frames)

        plain = list(to_okx_frames(LOBSimulator(cfg), events))
        adapter2 = OKXBooksAdapter("SIM-USDT", price_decimals=1)
        without_snaps, stats = replay(adapter2, plain)

        assert stats.resyncs == 0
        assert with_snaps.top().bid_px == without_snaps.top().bid_px
        assert with_snaps.top().ask_px == without_snaps.top().ask_px
        assert with_snaps.top().bid_sz == pytest.approx(without_snaps.top().bid_sz)


# --------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------- #
class TestEndToEnd:
    """The headline claim, produced by the production code path."""

    def test_enough_data(self, buckets):
        assert len(buckets) > 100

    def test_ofi_explains_price_changes(self, buckets):
        x = np.array([b.ofi for b in buckets])
        y = np.array([b.delta_mid for b in buckets])
        r = ols(y, x)
        assert r.slope > 0, "buying pressure must push the price up"
        assert r.slope_t > 5, "the relationship must be clearly significant"
        assert r.r_squared > 0.2

    def test_relationship_strengthens_over_longer_intervals(self, buckets):
        """Noise averages out; the signal does not."""
        x = np.array([b.ofi for b in buckets])
        y = np.array([b.delta_mid for b in buckets])
        short = ols(y, x).r_squared

        long_buckets = []
        for i in range(0, len(buckets) - 5, 5):
            chunk = buckets[i : i + 5]
            long_buckets.append(
                (sum(b.ofi for b in chunk), sum(b.delta_mid for b in chunk))
            )
        lx = np.array([a for a, _ in long_buckets])
        ly = np.array([b for _, b in long_buckets])
        assert ols(ly, lx).r_squared > short

    def test_impact_falls_as_depth_rises(self, buckets):
        d = np.array([b.avg_depth for b in buckets])
        x = np.array([b.ofi for b in buckets])
        y = np.array([b.delta_mid for b in buckets])
        thin = d < np.median(d)
        assert ols(y[thin], x[thin]).slope > ols(y[~thin], x[~thin]).slope


# --------------------------------------------------------------------- #
# strategy
# --------------------------------------------------------------------- #
class TestReservationPrice:
    def test_flat_inventory_sits_at_the_mid(self):
        p = ASParams()
        assert reservation_price(100.0, 0.0, 0.0, p) == 100.0

    def test_long_inventory_pushes_it_down(self):
        p = ASParams()
        assert reservation_price(100.0, 10.0, 0.0, p) < 100.0

    def test_short_inventory_pushes_it_up(self):
        p = ASParams()
        assert reservation_price(100.0, -10.0, 0.0, p) > 100.0

    def test_skew_shrinks_as_the_session_ends(self):
        p = ASParams()
        early = abs(reservation_price(100.0, 10.0, 0.0, p) - 100.0)
        late = abs(reservation_price(100.0, 10.0, 0.99, p) - 100.0)
        assert late < early

    def test_skew_is_zero_at_expiry(self):
        p = ASParams()
        assert reservation_price(100.0, 50.0, 1.0, p) == 100.0

    def test_higher_risk_aversion_skews_more(self):
        timid = reservation_price(100.0, 10.0, 0.0, ASParams(gamma=1.0))
        bold = reservation_price(100.0, 10.0, 0.0, ASParams(gamma=0.01))
        assert timid < bold < 100.0


class TestSpread:
    def test_positive(self):
        assert optimal_spread(0.0, ASParams()) > 0

    def test_narrows_toward_the_close(self):
        p = ASParams()
        assert optimal_spread(0.9, p) < optimal_spread(0.0, p)

    def test_wider_when_volatility_is_higher(self):
        assert optimal_spread(0.0, ASParams(sigma=5.0)) > optimal_spread(
            0.0, ASParams(sigma=1.0)
        )

    def test_thinner_book_means_a_wider_quote(self):
        """Lower k means arrivals decay slowly with distance, so it pays
        to quote wider."""
        assert optimal_spread(0.0, ASParams(k=0.5)) > optimal_spread(
            0.0, ASParams(k=5.0)
        )


class TestQuotes:
    def test_flat_quotes_straddle_the_mid(self):
        bid, ask = as_quotes(100.0, 0.0, 0.0, ASParams())
        assert bid < 100.0 < ask
        assert (bid + ask) / 2 == pytest.approx(100.0)

    def test_long_inventory_shifts_both_quotes_down(self):
        p = ASParams()
        flat_b, flat_a = as_quotes(100.0, 0.0, 0.0, p)
        long_b, long_a = as_quotes(100.0, 20.0, 0.0, p)
        assert long_b < flat_b and long_a < flat_a

    def test_symmetric_ignores_inventory(self):
        p = ASParams()
        assert symmetric_quotes(100.0, 0.0, 0.2, p) == symmetric_quotes(
            100.0, 50.0, 0.2, p
        )

    def test_both_quote_the_same_average_width(self):
        """Otherwise the comparison measures width, not skew."""
        p = ASParams()
        b1, a1 = as_quotes(100.0, 0.0, 0.5, p)
        b2, a2 = symmetric_quotes(100.0, 0.0, 0.5, p)
        assert (a1 - b1) == pytest.approx(a2 - b2)


class TestSessions:
    def test_session_trades(self):
        r = run_session(as_quotes, ASParams(), rng=np.random.default_rng(0))
        assert r.n_trades > 0
        assert r.max_abs_inventory >= abs(r.final_inventory)

    def test_inventory_limit_is_respected(self):
        r = run_session(
            symmetric_quotes, ASParams(), rng=np.random.default_rng(1),
            inventory_limit=3,
        )
        assert r.max_abs_inventory <= 3

    def test_inventory_control_reduces_variance(self):
        """The headline result from the paper."""
        inv, sym = compare(n_runs=600, seed=99)
        assert inv.pnl_std < sym.pnl_std * 0.8
        assert inv.inv_std < sym.inv_std * 0.6

    def test_variance_reduction_is_paid_for(self):
        """It is a trade, not a free lunch: mean P&L is lower."""
        inv, sym = compare(n_runs=600, seed=99)
        assert inv.pnl_mean < sym.pnl_mean

    def test_both_strategies_are_profitable_on_average(self):
        inv, sym = compare(n_runs=600, seed=99)
        assert inv.pnl_mean > 0 and sym.pnl_mean > 0

    def test_higher_risk_aversion_holds_less_inventory(self):
        timid, _ = compare(ASParams(gamma=0.5), n_runs=400, seed=5)
        bold, _ = compare(ASParams(gamma=0.02), n_runs=400, seed=5)
        assert timid.inv_std < bold.inv_std
