import pytest

from lob_engine.book import CrossedBook, OrderBook, SequenceGap
from lob_engine.types import BookUpdate, Level, Side


def snap(bids, asks, seq=1, symbol="X"):
    return BookUpdate(
        symbol=symbol,
        bids=tuple(Level(p, s) for p, s in bids),
        asks=tuple(Level(p, s) for p, s in asks),
        is_snapshot=True,
        seq=seq,
        prev_seq=None,
        recv_ts_ns=1_000,
    )


def delta(bids, asks, seq, prev_seq, ts=2_000, symbol="X"):
    return BookUpdate(
        symbol=symbol,
        bids=tuple(Level(p, s) for p, s in bids),
        asks=tuple(Level(p, s) for p, s in asks),
        is_snapshot=False,
        seq=seq,
        prev_seq=prev_seq,
        recv_ts_ns=ts,
    )


@pytest.fixture
def book():
    b = OrderBook("X")
    b.apply(snap([(100, 5), (99, 10), (98, 20)], [(102, 4), (103, 8), (104, 16)]))
    return b


class TestSnapshot:
    def test_applies(self, book):
        t = book.top()
        assert (t.bid_px, t.bid_sz) == (100, 5)
        assert (t.ask_px, t.ask_sz) == (102, 4)
        assert book.ready
        assert book.seq == 1

    def test_second_snapshot_replaces_rather_than_merges(self, book):
        book.apply(snap([(50, 1)], [(60, 1)], seq=99))
        t = book.top()
        assert (t.bid_px, t.ask_px) == (50, 60)
        assert len(book.depth(Side.BUY)) == 1
        assert book.seq == 99

    def test_depth_is_best_first(self, book):
        assert [lv.price for lv in book.depth(Side.BUY)] == [100, 99, 98]
        assert [lv.price for lv in book.depth(Side.SELL)] == [102, 103, 104]

    def test_depth_limit(self, book):
        assert len(book.depth(Side.BUY, 2)) == 2


class TestDeltas:
    def test_size_change(self, book):
        book.apply(delta([(100, 7)], [], seq=2, prev_seq=1))
        assert book.top().bid_sz == 7

    def test_delete_level(self, book):
        book.apply(delta([(100, 0)], [], seq=2, prev_seq=1))
        assert book.top().bid_px == 99

    def test_new_best(self, book):
        book.apply(delta([(101, 3)], [], seq=2, prev_seq=1))
        t = book.top()
        assert (t.bid_px, t.bid_sz) == (101, 3)

    def test_delete_unknown_level_is_counted_not_fatal(self, book):
        book.apply(delta([(1, 0)], [], seq=2, prev_seq=1))
        assert book.stats.no_op_deletes == 1
        assert book.top().bid_px == 100

    def test_both_sides_in_one_message(self, book):
        book.apply(delta([(100, 1)], [(102, 2)], seq=2, prev_seq=1))
        t = book.top()
        assert (t.bid_sz, t.ask_sz) == (1, 2)

    def test_stats_counted(self, book):
        book.apply(delta([(100, 6)], [], seq=2, prev_seq=1))
        book.apply(delta([(100, 7)], [], seq=3, prev_seq=2))
        assert book.stats.snapshots_applied == 1
        assert book.stats.updates_applied == 2


class TestSequencing:
    def test_gap_raises_and_leaves_book_untouched(self, book):
        before = book.top().bid_sz
        with pytest.raises(SequenceGap) as exc:
            book.apply(delta([(100, 999)], [], seq=5, prev_seq=4))
        assert exc.value.expected == 1
        assert exc.value.got == 4
        assert book.top().bid_sz == before
        assert book.stats.gaps_detected == 1

    def test_in_sequence_is_fine(self, book):
        book.apply(delta([(100, 6)], [], seq=2, prev_seq=1))
        book.apply(delta([(100, 7)], [], seq=3, prev_seq=2))
        assert book.seq == 3

    def test_delta_before_snapshot_is_a_gap(self):
        b = OrderBook("X")
        with pytest.raises(SequenceGap):
            b.apply(delta([(100, 1)], [], seq=2, prev_seq=1))

    def test_snapshot_after_gap_recovers(self, book):
        with pytest.raises(SequenceGap):
            book.apply(delta([], [], seq=9, prev_seq=8))
        book.apply(snap([(100, 1)], [(101, 1)], seq=9))
        assert book.ready
        book.apply(delta([(100, 2)], [], seq=10, prev_seq=9))
        assert book.top().bid_sz == 2

    def test_non_strict_mode_ignores_gaps(self):
        b = OrderBook("X", strict_sequencing=False)
        b.apply(snap([(100, 5)], [(102, 5)]))
        b.apply(delta([(100, 6)], [], seq=99, prev_seq=98))
        assert b.top().bid_sz == 6

    def test_missing_sequence_numbers_are_tolerated(self, book):
        """Not every venue publishes them; absence must not be a gap."""
        book.apply(delta([(100, 6)], [], seq=None, prev_seq=None))
        assert book.top().bid_sz == 6


class TestDepthTruncation:
    def test_trims_beyond_max_depth(self):
        b = OrderBook("X", max_depth=2)
        b.apply(snap([(100, 1), (99, 1), (98, 1)], [(102, 1), (103, 1), (104, 1)]))
        assert [lv.price for lv in b.depth(Side.BUY)] == [100, 99]
        assert [lv.price for lv in b.depth(Side.SELL)] == [102, 103]

    def test_trims_the_far_side_not_the_touch(self):
        """A new best bid must push out the worst level, not itself."""
        b = OrderBook("X", max_depth=2)
        b.apply(snap([(100, 1), (99, 1)], [(102, 1), (103, 1)]))
        b.apply(delta([(101, 5)], [], seq=2, prev_seq=1))
        assert [lv.price for lv in b.depth(Side.BUY)] == [101, 100]
        b.apply(delta([], [(101, 0), (104, 1)], seq=3, prev_seq=2))
        assert [lv.price for lv in b.depth(Side.SELL)] == [102, 103]


class TestIntegrity:
    def test_healthy_book_passes(self, book):
        assert book.check_integrity()

    def test_crossed_detected(self, book):
        book.apply(delta([(105, 1)], [], seq=2, prev_seq=1))
        assert not book.check_integrity()
        assert book.stats.crossed_seen == 1

    def test_crossed_can_raise(self, book):
        book.apply(delta([(105, 1)], [], seq=2, prev_seq=1))
        with pytest.raises(CrossedBook):
            book.check_integrity(raise_on_cross=True)

    def test_locked_book_is_crossed(self, book):
        """bid == ask is not tradeable and counts as crossed."""
        book.apply(delta([(102, 1)], [], seq=2, prev_seq=1))
        assert not book.check_integrity()

    def test_empty_side_is_not_crossed(self):
        b = OrderBook("X")
        b.apply(snap([(100, 1)], []))
        assert b.check_integrity()


class TestReads:
    def test_average_depth(self, book):
        # bids 5 + 10 = 15, asks 4 + 8 = 12, mean of the two = 13.5
        assert book.average_depth(2) == pytest.approx(13.5)

    def test_size_at(self, book):
        assert book.size_at(Side.BUY, 99) == 10
        assert book.size_at(Side.SELL, 999) == 0

    def test_reset(self, book):
        book.reset()
        assert not book.ready
        assert book.top().bid_px is None


def test_many_updates_keep_book_sorted():
    """Fuzz: random inserts and deletes must never break ordering."""
    import random

    rng = random.Random(7)
    b = OrderBook("X")
    b.apply(snap([(500, 1)], [(600, 1)]))
    seq = 1
    for _ in range(4000):
        seq += 1
        px = rng.randint(400, 499) if rng.random() < 0.5 else rng.randint(601, 700)
        size = rng.choice([0, 0, 1, 2, 5])
        if px < 500:
            b.apply(delta([(px, size)], [], seq=seq, prev_seq=seq - 1))
        else:
            b.apply(delta([], [(px, size)], seq=seq, prev_seq=seq - 1))
    bids = [lv.price for lv in b.depth(Side.BUY)]
    asks = [lv.price for lv in b.depth(Side.SELL)]
    assert bids == sorted(bids, reverse=True)
    assert asks == sorted(asks)
    assert len(set(bids)) == len(bids)
    assert b.check_integrity()
