"""Every case of the Cont-Kukanov-Stoikov event contribution, by hand.

    e_n = 1{Pb_n >= Pb_n-1} qb_n - 1{Pb_n <= Pb_n-1} qb_n-1
        - 1{Pa_n <= Pa_n-1} qa_n + 1{Pa_n >= Pa_n-1} qa_n-1
"""

import numpy as np
import pytest

from lob_engine.ofi import OFIAccumulator, bucket_stream, event_contribution
from lob_engine.types import TopOfBook


def tob(bid_px, bid_sz, ask_px, ask_sz, ts=0):
    return TopOfBook(bid_px=bid_px, bid_sz=bid_sz, ask_px=ask_px,
                     ask_sz=ask_sz, recv_ts_ns=ts)


BASE = tob(100, 10, 102, 8)


class TestBidSide:
    def test_size_added_at_same_price(self):
        """Bid price unchanged: e = qb_n - qb_n-1. More demand, positive."""
        assert event_contribution(BASE, tob(100, 15, 102, 8)) == 5

    def test_size_removed_at_same_price(self):
        """A market sell or a cancel on the bid. Negative."""
        assert event_contribution(BASE, tob(100, 6, 102, 8)) == -4

    def test_price_improving_bid(self):
        """New better bid: the whole new size counts as fresh demand."""
        assert event_contribution(BASE, tob(101, 3, 102, 8)) == 3

    def test_bid_price_falls(self):
        """The old bid level vanished: minus the size that was there."""
        assert event_contribution(BASE, tob(99, 40, 102, 8)) == -10


class TestAskSide:
    def test_size_added_at_same_price(self):
        """More supply is negative imbalance."""
        assert event_contribution(BASE, tob(100, 10, 102, 12)) == -4

    def test_size_removed_at_same_price(self):
        """Ask lifted or cancelled: supply gone, positive."""
        assert event_contribution(BASE, tob(100, 10, 102, 5)) == 3

    def test_price_improving_ask(self):
        """New better offer: minus the whole new size."""
        assert event_contribution(BASE, tob(100, 10, 101, 6)) == -6

    def test_ask_price_rises(self):
        """The old ask level vanished: plus the size that was there."""
        assert event_contribution(BASE, tob(100, 10, 103, 30)) == 8


class TestCombined:
    def test_both_sides_move(self):
        """Bid up by 3 new, ask level gone releasing 8: 3 + 8 = 11."""
        assert event_contribution(BASE, tob(101, 3, 103, 30)) == 11

    def test_no_change_is_zero(self):
        assert event_contribution(BASE, BASE) == 0

    def test_symmetric_add_cancels_out(self):
        """Equal size added to both sides is no net imbalance."""
        assert event_contribution(BASE, tob(100, 15, 102, 13)) == 0

    def test_sign_convention(self):
        """Buying pressure positive, selling pressure negative."""
        assert event_contribution(BASE, tob(101, 20, 102, 8)) > 0
        assert event_contribution(BASE, tob(99, 20, 102, 8)) < 0


class TestOneSided:
    def test_missing_ask_contributes_nothing(self):
        assert event_contribution(BASE, tob(100, 10, None, 0)) == 0

    def test_missing_bid_contributes_nothing(self):
        assert event_contribution(tob(None, 0, 102, 8), BASE) == 0


class TestAccumulator:
    def test_first_observation_is_zero(self):
        a = OFIAccumulator()
        assert a.update(BASE) == 0.0
        assert a.value == 0.0

    def test_running_sum(self):
        a = OFIAccumulator()
        a.update(BASE)
        a.update(tob(100, 15, 102, 8))  # +5
        a.update(tob(100, 15, 102, 11))  # -3
        assert a.value == 2

    def test_drain_resets_but_keeps_the_reference_point(self):
        a = OFIAccumulator()
        a.update(BASE)
        a.update(tob(100, 15, 102, 8))
        ofi, n = a.drain()
        assert ofi == 5 and n == 1
        assert a.value == 0
        # next contribution measured against 15, not against 10
        a.update(tob(100, 18, 102, 8))
        assert a.value == 3

    def test_event_count_ignores_no_ops(self):
        a = OFIAccumulator()
        a.update(BASE)
        a.update(BASE)
        a.update(BASE)
        assert a.events == 0


class TestBucketing:
    def test_splits_on_interval(self):
        obs = [tob(100, 10 + i, 102, 8, ts=i * 300_000_000) for i in range(10)]
        buckets = bucket_stream(obs, interval_ns=1_000_000_000)
        assert len(buckets) >= 2
        assert all(b.end_ns > b.start_ns for b in buckets)

    def test_total_ofi_is_conserved_across_buckets(self):
        obs = [tob(100, 10 + i, 102, 8, ts=i * 300_000_000) for i in range(10)]
        buckets = bucket_stream(obs, interval_ns=1_000_000_000)
        total = sum(b.ofi for b in buckets)
        a = OFIAccumulator()
        for o in obs:
            a.update(o)
        assert total == pytest.approx(a.value)

    def test_delta_mid_is_conserved(self):
        obs = [tob(100 + i, 10, 102 + i, 8, ts=i * 300_000_000) for i in range(10)]
        buckets = bucket_stream(obs, interval_ns=1_000_000_000)
        assert sum(b.delta_mid for b in buckets) == pytest.approx(
            obs[-1].mid - obs[0].mid
        )

    def test_rejects_bad_interval(self):
        with pytest.raises(ValueError):
            bucket_stream([BASE], interval_ns=0)

    def test_rejects_misaligned_depths(self):
        with pytest.raises(ValueError, match="align"):
            bucket_stream([BASE, BASE], interval_ns=1000, depths=[1.0])

    def test_empty_input(self):
        assert bucket_stream([], interval_ns=1000) == []

    def test_uses_supplied_depths(self):
        obs = [tob(100, 10, 102, 8, ts=i * 100_000_000) for i in range(5)]
        buckets = bucket_stream(obs, interval_ns=10_000_000_000, depths=[7.0] * 5)
        assert buckets[0].avg_depth == pytest.approx(7.0)

    def test_one_sided_observations_do_not_create_buckets(self):
        obs = [tob(None, 0, None, 0, ts=i * 100_000_000) for i in range(5)]
        assert bucket_stream(obs, interval_ns=1_000_000_000) == []


def test_ofi_correlates_with_price_in_a_directional_stream():
    """Sanity check the sign end to end: repeated buying pressure should
    produce positive OFI alongside a rising mid."""
    obs = []
    bid, ask = 100, 102
    for i in range(200):
        bid += 1
        ask += 1
        obs.append(tob(bid, 10, ask, 10, ts=i * 10_000_000))
    buckets = bucket_stream(obs, interval_ns=200_000_000)
    ofis = np.array([b.ofi for b in buckets])
    dmids = np.array([b.delta_mid for b in buckets])
    assert (ofis > 0).all()
    assert (dmids > 0).all()
