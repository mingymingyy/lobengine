import pytest

from lob_engine.fix import ExecType, OrdStatus, OrdType, TimeInForce
from lob_engine.types import Side
from lob_engine.venue import MockVenue


@pytest.fixture
def reports():
    return []


@pytest.fixture
def venue(reports):
    v = MockVenue("X", on_exec=reports.append)
    for px, qty in [(99, 50), (100, 20)]:
        v.add_liquidity(Side.BUY, px, qty)
    for px, qty in [(101, 15), (102, 30)]:
        v.add_liquidity(Side.SELL, px, qty)
    return v


def fills(reports, cl_ord_id=None):
    return [
        r for r in reports
        if r.exec_type is ExecType.TRADE
        and (cl_ord_id is None or r.cl_ord_id == cl_ord_id)
    ]


class TestBook:
    def test_touch(self, venue):
        assert venue.best_bid() == 100
        assert venue.best_ask() == 101
        assert venue.mid() == 100.5

    def test_size_at(self, venue):
        assert venue.size_at(Side.BUY, 100) == 20
        assert venue.size_at(Side.SELL, 999) == 0

    def test_empty_book(self):
        v = MockVenue("X")
        assert v.best_bid() is None and v.mid() is None


class TestResting:
    def test_non_marketable_order_rests(self, venue, reports):
        venue.submit("A", Side.BUY, 10, 99)
        assert reports[0].exec_type is ExecType.NEW
        assert fills(reports) == []
        assert venue.size_at(Side.BUY, 99) == 60

    def test_duplicate_client_id_rejected(self, venue, reports):
        venue.submit("A", Side.BUY, 10, 99)
        reports.clear()
        venue.submit("A", Side.BUY, 10, 99)
        assert reports[0].exec_type is ExecType.REJECTED

    def test_cancel(self, venue, reports):
        venue.submit("A", Side.BUY, 10, 99)
        reports.clear()
        venue.cancel("A")
        assert reports[0].exec_type is ExecType.CANCELED
        assert venue.size_at(Side.BUY, 99) == 50

    def test_cancel_unknown(self, venue, reports):
        venue.cancel("nope")
        assert reports[0].exec_type is ExecType.REJECTED


class TestQueuePosition:
    def test_we_join_the_back_of_the_queue(self, venue):
        venue.submit("A", Side.BUY, 10, 100)
        assert venue.queue_ahead("A") == 20

    def test_a_trade_at_our_price_does_not_fill_us_yet(self, venue, reports):
        """The single most common backtest error."""
        venue.submit("A", Side.BUY, 10, 100)
        reports.clear()
        venue.market_order(Side.SELL, 15)
        assert fills(reports, "A") == []
        assert venue.queue_ahead("A") == 5

    def test_we_fill_once_the_queue_clears(self, venue, reports):
        venue.submit("A", Side.BUY, 10, 100)
        reports.clear()
        venue.market_order(Side.SELL, 25)  # clears 20 ahead, then 5 of ours
        f = fills(reports, "A")
        assert len(f) == 1 and f[0].last_qty == 5 and f[0].last_px == 100

    def test_cancels_ahead_improve_our_position(self, venue):
        venue.submit("A", Side.BUY, 10, 100)
        assert venue.queue_ahead("A") == 20
        venue.cancel_ahead(Side.BUY, 100, 12)
        assert venue.queue_ahead("A") == 8

    def test_cancel_ahead_never_touches_our_own_order(self, venue):
        venue.submit("A", Side.BUY, 10, 100)
        cancelled = venue.cancel_ahead(Side.BUY, 100, 100)
        assert cancelled == 20  # only the anonymous size
        assert venue.size_at(Side.BUY, 100) == 10

    def test_time_priority_between_two_of_our_orders(self, venue, reports):
        venue.submit("FIRST", Side.BUY, 5, 100)
        venue.submit("SECOND", Side.BUY, 5, 100)
        venue.cancel_ahead(Side.BUY, 100, 20)
        reports.clear()
        venue.market_order(Side.SELL, 5)
        assert {f.cl_ord_id for f in fills(reports)} == {"FIRST"}


class TestMatching:
    def test_marketable_limit_fills_immediately(self, venue, reports):
        venue.submit("A", Side.BUY, 10, 101)
        f = fills(reports, "A")
        assert len(f) == 1
        assert f[0].last_qty == 10 and f[0].last_px == 101
        assert f[0].ord_status is OrdStatus.FILLED

    def test_walks_multiple_levels(self, venue, reports):
        venue.submit("A", Side.BUY, 25, 102)
        f = fills(reports, "A")
        assert [(x.last_px, x.last_qty) for x in f] == [(101, 15), (102, 10)]

    def test_price_priority_best_first(self, venue, reports):
        venue.submit("A", Side.SELL, 30, 99)
        f = fills(reports, "A")
        assert [x.last_px for x in f] == [100, 99]

    def test_remainder_rests(self, venue):
        venue.submit("A", Side.BUY, 20, 101)
        assert venue.size_at(Side.SELL, 101) == 0
        assert venue.size_at(Side.BUY, 101) == 5

    def test_does_not_trade_through_the_limit(self, venue, reports):
        venue.submit("A", Side.BUY, 40, 101)
        f = fills(reports, "A")
        assert all(x.last_px <= 101 for x in f)
        assert sum(x.last_qty for x in f) == 15

    def test_passive_side_gets_its_own_report(self, venue, reports):
        venue.submit("PASSIVE", Side.SELL, 10, 103)
        reports.clear()
        # 15 at 101 + 30 at 102 + 10 at 103 = 55 clears the passive order
        venue.submit("AGGRESSIVE", Side.BUY, 55, 103)
        f = fills(reports, "PASSIVE")
        assert len(f) == 1
        assert f[0].last_qty == 10 and f[0].last_px == 103
        assert f[0].ord_status is OrdStatus.FILLED

    def test_passive_side_partial_fill(self, venue, reports):
        venue.submit("PASSIVE", Side.SELL, 10, 103)
        reports.clear()
        # only 5 left by the time the aggressor reaches 103
        venue.submit("AGGRESSIVE", Side.BUY, 50, 103)
        f = fills(reports, "PASSIVE")
        assert len(f) == 1 and f[0].last_qty == 5
        assert f[0].ord_status is OrdStatus.PARTIALLY_FILLED


class TestTimeInForce:
    def test_ioc_cancels_the_remainder(self, venue, reports):
        venue.submit("A", Side.BUY, 100, 101, tif=TimeInForce.IOC)
        assert sum(f.last_qty for f in fills(reports, "A")) == 15
        assert reports[-1].exec_type is ExecType.CANCELED
        assert venue.size_at(Side.BUY, 101) == 0

    def test_fok_rejected_when_not_fully_fillable(self, venue, reports):
        venue.submit("A", Side.BUY, 100, 101, tif=TimeInForce.FOK)
        assert reports[0].exec_type is ExecType.REJECTED
        assert fills(reports) == []
        assert venue.size_at(Side.SELL, 101) == 15  # book untouched

    def test_fok_fills_when_it_can(self, venue, reports):
        venue.submit("A", Side.BUY, 15, 101, tif=TimeInForce.FOK)
        assert sum(f.last_qty for f in fills(reports, "A")) == 15

    def test_market_order_does_not_rest(self, venue):
        venue.submit("A", Side.BUY, 1000, None, ord_type=OrdType.MARKET)
        assert venue.size_at(Side.BUY, 102) == 0

    def test_limit_without_price_rejected(self, venue, reports):
        venue.submit("A", Side.BUY, 10, None)
        assert reports[0].exec_type is ExecType.REJECTED


class TestConservation:
    def test_traded_size_matches_book_reduction(self, venue):
        before = sum(venue.size_at(Side.SELL, p) for p in (101, 102))
        traded = venue.market_order(Side.BUY, 20)
        after = sum(venue.size_at(Side.SELL, p) for p in (101, 102))
        assert traded == 20
        assert before - after == pytest.approx(20)

    def test_market_order_beyond_the_book(self, venue):
        traded = venue.market_order(Side.BUY, 1000)
        assert traded == 45  # 15 + 30, all the offered size
        assert venue.best_ask() is None

    def test_trade_tape_records_everything(self, venue):
        venue.market_order(Side.BUY, 20)
        assert sum(q for _, q, _ in venue.trades) == 20
        assert all(side is Side.BUY for _, _, side in venue.trades)


def test_integration_with_the_oms():
    """Reports from the venue must drive the OMS without any illegal
    transitions."""
    from lob_engine.oms import OMS

    oms = OMS()
    venue = MockVenue("X", on_exec=lambda er: oms.on_execution_report(er))
    venue.add_liquidity(Side.SELL, 101, 30)

    o = oms.create("X", Side.BUY, 25, 101)
    venue.submit(o.cl_ord_id, o.side, o.qty, o.price)

    assert o.status is OrdStatus.FILLED
    assert o.cum_qty == 25
    assert oms.position("X") == 25
    assert oms.rejected_transitions == 0
