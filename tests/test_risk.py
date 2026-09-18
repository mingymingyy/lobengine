import pytest

from lob_engine.fix import ExecType, OrdStatus
from lob_engine.oms import OMS, ExecutionReport
from lob_engine.risk import RiskEngine, RiskLimits
from lob_engine.types import Side


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


@pytest.fixture
def oms():
    return OMS()


@pytest.fixture
def clock():
    return Clock()


def engine(oms, clock=None, **limit_kwargs):
    return RiskEngine(RiskLimits(**limit_kwargs), oms, clock=clock)


def ack(oms, order):
    oms.on_execution_report(
        ExecutionReport(order.cl_ord_id, ExecType.NEW, OrdStatus.NEW)
    )


def fill(oms, order, qty, px):
    status = (
        OrdStatus.FILLED if qty >= order.qty else OrdStatus.PARTIALLY_FILLED
    )
    oms.on_execution_report(
        ExecutionReport(order.cl_ord_id, ExecType.TRADE, status,
                        last_qty=qty, last_px=px)
    )


class TestBasics:
    def test_accepts_a_sane_order(self, oms):
        r = engine(oms, max_order_qty=100)
        assert r.check_new_order("X", Side.BUY, 10, 100, mid=100.0)

    def test_rejects_non_positive_quantity(self, oms):
        r = engine(oms)
        assert not r.check_new_order("X", Side.BUY, 0, 100)
        assert r.check_new_order("X", Side.BUY, -1, 100).check == "qty_sign"

    def test_limit_without_price(self, oms):
        r = engine(oms)
        d = r.check_new_order("X", Side.BUY, 1, None)
        assert not d and d.check == "missing_price"

    def test_market_order_needs_no_price(self, oms):
        r = engine(oms)
        assert r.check_new_order("X", Side.BUY, 1, None, is_market=True)

    def test_market_orders_can_be_disabled(self, oms):
        r = engine(oms, allow_market_orders=False)
        d = r.check_new_order("X", Side.BUY, 1, None, is_market=True)
        assert not d and d.check == "market_disabled"


class TestFatFinger:
    def test_max_order_qty(self, oms):
        r = engine(oms, max_order_qty=100)
        assert r.check_new_order("X", Side.BUY, 100, 10)
        d = r.check_new_order("X", Side.BUY, 101, 10)
        assert not d and d.check == "max_order_qty"

    def test_max_order_value(self, oms):
        r = engine(oms, max_order_value=1000)
        assert r.check_new_order("X", Side.BUY, 10, 100)
        d = r.check_new_order("X", Side.BUY, 11, 100)
        assert not d and d.check == "max_order_value"

    def test_notional_uses_the_price_scale(self, oms):
        """Prices are ticks; limits are quoted in price units."""
        r = RiskEngine(RiskLimits(max_order_value=1000), oms, price_scale_factor=100.0)
        # 10 units at 10000 ticks = 10 * 100.00 = 1000
        assert r.check_new_order("X", Side.BUY, 10, 10000)
        assert not r.check_new_order("X", Side.BUY, 11, 10000)

    def test_price_collar(self, oms):
        r = engine(oms, price_collar_bps=100)  # 1%
        assert r.check_new_order("X", Side.BUY, 1, 10050, mid=10000.0)
        d = r.check_new_order("X", Side.BUY, 1, 10200, mid=10000.0)
        assert not d and d.check == "price_collar"

    def test_collar_applies_in_both_directions(self, oms):
        r = engine(oms, price_collar_bps=100)
        assert not r.check_new_order("X", Side.SELL, 1, 9800, mid=10000.0)

    def test_collar_skipped_without_a_mid(self, oms):
        """No reference price means no collar; do not guess one."""
        r = engine(oms, price_collar_bps=1)
        assert r.check_new_order("X", Side.BUY, 1, 999999, mid=None)


class TestVenueConventions:
    def test_lot_size(self, oms):
        r = engine(oms, lot_size=100)
        assert r.check_new_order("X", Side.BUY, 200, 10)
        d = r.check_new_order("X", Side.BUY, 150, 10)
        assert not d and d.check == "lot_size"

    def test_tick_size(self, oms):
        r = engine(oms, tick_size=5)
        assert r.check_new_order("X", Side.BUY, 1, 100)
        d = r.check_new_order("X", Side.BUY, 1, 102)
        assert not d and d.check == "tick_size"

    def test_fractional_lot_tolerance(self, oms):
        """Float arithmetic must not reject a legitimate multiple."""
        r = engine(oms, lot_size=0.1)
        assert r.check_new_order("X", Side.BUY, 0.3, 10)


class TestPosition:
    def test_current_position_counts(self, oms):
        r = engine(oms, max_position=100)
        o = oms.create("X", Side.BUY, 90, 10)
        ack(oms, o)
        fill(oms, o, 90, 10)
        assert r.check_new_order("X", Side.BUY, 10, 10)
        d = r.check_new_order("X", Side.BUY, 11, 10)
        assert not d and d.check == "max_position"

    def test_working_orders_count_too(self, oms):
        """Checking only the filled position lets resting orders build an
        unbounded position."""
        r = engine(oms, max_position=100)
        o = oms.create("X", Side.BUY, 95, 10)
        ack(oms, o)
        assert oms.position("X") == 0
        d = r.check_new_order("X", Side.BUY, 10, 10)
        assert not d and d.check == "max_position"

    def test_short_side(self, oms):
        r = engine(oms, max_position=50)
        o = oms.create("X", Side.SELL, 45, 10)
        ack(oms, o)
        assert not r.check_new_order("X", Side.SELL, 10, 10)

    def test_reducing_orders_are_allowed(self, oms):
        r = engine(oms, max_position=100)
        o = oms.create("X", Side.BUY, 100, 10)
        ack(oms, o)
        fill(oms, o, 100, 10)
        assert r.check_new_order("X", Side.SELL, 100, 20)

    def test_max_open_orders(self, oms):
        r = engine(oms, max_open_orders=2)
        for _ in range(2):
            ack(oms, oms.create("X", Side.BUY, 1, 10))
        d = r.check_new_order("X", Side.BUY, 1, 10)
        assert not d and d.check == "max_open_orders"


class TestThrottle:
    def test_rate_limit_blocks_then_recovers(self, oms, clock):
        r = engine(oms, clock=clock, max_messages_per_sec=3)
        for _ in range(3):
            assert r.check_new_order("X", Side.BUY, 1, 10)
        d = r.check_new_order("X", Side.BUY, 1, 10)
        assert not d and d.check == "throttle"
        clock.advance(1.1)
        assert r.check_new_order("X", Side.BUY, 1, 10)

    def test_rejected_orders_do_not_consume_the_budget(self, oms, clock):
        r = engine(oms, clock=clock, max_messages_per_sec=2, max_order_qty=10)
        r.check_new_order("X", Side.BUY, 999, 10)  # rejected on size
        assert r.check_new_order("X", Side.BUY, 1, 10)
        assert r.check_new_order("X", Side.BUY, 1, 10)


class TestSelfTrade:
    def test_buy_crossing_our_own_offer(self, oms):
        r = engine(oms)
        ack(oms, oms.create("X", Side.SELL, 10, 100))
        d = r.check_new_order("X", Side.BUY, 5, 100)
        assert not d and d.check == "self_trade"

    def test_sell_crossing_our_own_bid(self, oms):
        r = engine(oms)
        ack(oms, oms.create("X", Side.BUY, 10, 100))
        assert not r.check_new_order("X", Side.SELL, 5, 99)

    def test_non_crossing_quote_is_fine(self, oms):
        r = engine(oms)
        ack(oms, oms.create("X", Side.SELL, 10, 102))
        assert r.check_new_order("X", Side.BUY, 5, 100)

    def test_only_our_own_symbol(self, oms):
        r = engine(oms)
        ack(oms, oms.create("OTHER", Side.SELL, 10, 100))
        assert r.check_new_order("X", Side.BUY, 5, 100)


class TestKillSwitch:
    def test_kill_and_resume(self, oms):
        r = engine(oms)
        assert r.check_new_order("X", Side.BUY, 1, 10)
        r.kill("limit breach on the desk")
        d = r.check_new_order("X", Side.BUY, 1, 10)
        assert not d and d.check == "kill_switch"
        assert "limit breach" in d.reason
        r.resume()
        assert r.check_new_order("X", Side.BUY, 1, 10)

    def test_kill_beats_every_other_check(self, oms):
        r = engine(oms, max_order_qty=1)
        r.kill()
        assert r.check_new_order("X", Side.BUY, 999, 10).check == "kill_switch"


class TestStats:
    def test_counts_by_check(self, oms):
        r = engine(oms, max_order_qty=10, price_collar_bps=10)
        r.check_new_order("X", Side.BUY, 1, 100, mid=100.0)
        r.check_new_order("X", Side.BUY, 99, 100, mid=100.0)
        r.check_new_order("X", Side.BUY, 99, 100, mid=100.0)
        r.check_new_order("X", Side.BUY, 1, 200, mid=100.0)
        assert r.stats.checked == 4
        assert r.stats.accepted == 1
        assert r.stats.rejected == 3
        assert r.stats.by_check["max_order_qty"] == 2
        assert r.stats.by_check["price_collar"] == 1


def test_no_limits_means_everything_passes(oms):
    """An empty RiskLimits must not silently block trading."""
    r = engine(oms)
    assert r.check_new_order("X", Side.BUY, 1e9, 1)
