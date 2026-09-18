import pytest

from lob_engine.fix import ExecType, OrdStatus, OrdType
from lob_engine.oms import OMS, ExecutionReport, OrderStateError, UnknownOrder
from lob_engine.types import Side


@pytest.fixture
def oms():
    return OMS(id_prefix="T")


@pytest.fixture
def order(oms):
    return oms.create("D05", Side.BUY, 100.0, 3450)


def er(cl_ord_id, exec_type, ord_status, **kw):
    return ExecutionReport(
        cl_ord_id=cl_ord_id, exec_type=exec_type, ord_status=ord_status, **kw
    )


class TestCreate:
    def test_defaults(self, order):
        assert order.status is OrdStatus.PENDING_NEW
        assert order.leaves_qty == 100.0
        assert order.cum_qty == 0.0
        assert order.is_working

    def test_ids_are_unique(self, oms):
        ids = {oms.create("X", Side.BUY, 1, 1).cl_ord_id for _ in range(100)}
        assert len(ids) == 100

    def test_duplicate_id_rejected(self, oms):
        oms.create("X", Side.BUY, 1, 1, cl_ord_id="DUP")
        with pytest.raises(ValueError, match="duplicate"):
            oms.create("X", Side.BUY, 1, 1, cl_ord_id="DUP")

    def test_bad_quantity(self, oms):
        with pytest.raises(ValueError):
            oms.create("X", Side.BUY, 0, 1)
        with pytest.raises(ValueError):
            oms.create("X", Side.BUY, -5, 1)

    def test_limit_needs_a_price(self, oms):
        with pytest.raises(ValueError, match="requires a price"):
            oms.create("X", Side.BUY, 1, None, OrdType.LIMIT)

    def test_market_does_not_need_a_price(self, oms):
        o = oms.create("X", Side.BUY, 1, None, OrdType.MARKET)
        assert o.price is None

    def test_unknown_order_lookup(self, oms):
        with pytest.raises(UnknownOrder):
            oms.get("nope")


class TestLifecycle:
    def test_ack(self, oms, order):
        oms.on_execution_report(er(order.cl_ord_id, ExecType.NEW, OrdStatus.NEW,
                                   order_id="V1"))
        assert order.status is OrdStatus.NEW
        assert order.order_id == "V1"

    def test_partial_then_full_fill(self, oms, order):
        cid = order.cl_ord_id
        oms.on_execution_report(er(cid, ExecType.NEW, OrdStatus.NEW))
        oms.on_execution_report(
            er(cid, ExecType.TRADE, OrdStatus.PARTIALLY_FILLED,
               last_qty=40, last_px=3450)
        )
        assert order.cum_qty == 40 and order.leaves_qty == 60
        oms.on_execution_report(
            er(cid, ExecType.TRADE, OrdStatus.FILLED, last_qty=60, last_px=3460)
        )
        assert order.status is OrdStatus.FILLED
        assert order.leaves_qty == 0
        assert not order.is_working
        # weighted average: (40*3450 + 60*3460) / 100
        assert order.avg_px == pytest.approx(3456.0)

    def test_position_tracks_fills(self, oms, order):
        cid = order.cl_ord_id
        oms.on_execution_report(er(cid, ExecType.NEW, OrdStatus.NEW))
        oms.on_execution_report(
            er(cid, ExecType.TRADE, OrdStatus.FILLED, last_qty=100, last_px=3450)
        )
        assert oms.position("D05") == 100

        sell = oms.create("D05", Side.SELL, 30, 3500)
        oms.on_execution_report(er(sell.cl_ord_id, ExecType.NEW, OrdStatus.NEW))
        oms.on_execution_report(
            er(sell.cl_ord_id, ExecType.TRADE, OrdStatus.FILLED,
               last_qty=30, last_px=3500)
        )
        assert oms.position("D05") == 70

    def test_cancel(self, oms, order):
        cid = order.cl_ord_id
        oms.on_execution_report(er(cid, ExecType.NEW, OrdStatus.NEW))
        oms.on_execution_report(er(cid, ExecType.PENDING_CANCEL,
                                   OrdStatus.PENDING_CANCEL))
        oms.on_execution_report(er(cid, ExecType.CANCELED, OrdStatus.CANCELED))
        assert order.status is OrdStatus.CANCELED
        assert order.leaves_qty == 0

    def test_reject_records_the_reason(self, oms, order):
        oms.on_execution_report(
            er(order.cl_ord_id, ExecType.REJECTED, OrdStatus.REJECTED,
               text="price outside limits")
        )
        assert order.status is OrdStatus.REJECTED
        assert order.reject_reason == "price outside limits"

    def test_fill_racing_a_cancel(self, oms, order):
        """A fill arriving while a cancel is pending is legal and common."""
        cid = order.cl_ord_id
        oms.on_execution_report(er(cid, ExecType.NEW, OrdStatus.NEW))
        oms.on_execution_report(er(cid, ExecType.PENDING_CANCEL,
                                   OrdStatus.PENDING_CANCEL))
        oms.on_execution_report(
            er(cid, ExecType.TRADE, OrdStatus.FILLED, last_qty=100, last_px=3450)
        )
        assert order.status is OrdStatus.FILLED
        assert oms.position("D05") == 100

    def test_cancel_reject_returns_to_working(self, oms, order):
        cid = order.cl_ord_id
        oms.on_execution_report(er(cid, ExecType.NEW, OrdStatus.NEW))
        oms.on_execution_report(er(cid, ExecType.PENDING_CANCEL,
                                   OrdStatus.PENDING_CANCEL))
        oms.on_execution_report(er(cid, ExecType.NEW, OrdStatus.NEW))
        assert order.status is OrdStatus.NEW
        assert order.is_working


class TestIllegalTransitions:
    def test_fill_after_terminal_state(self, oms, order):
        cid = order.cl_ord_id
        oms.on_execution_report(er(cid, ExecType.NEW, OrdStatus.NEW))
        oms.on_execution_report(er(cid, ExecType.CANCELED, OrdStatus.CANCELED))
        with pytest.raises(OrderStateError, match="illegal transition"):
            oms.on_execution_report(
                er(cid, ExecType.TRADE, OrdStatus.FILLED, last_qty=1, last_px=1)
            )
        assert oms.rejected_transitions == 1
        assert oms.position("D05") == 0  # nothing was booked

    def test_reject_after_fill(self, oms, order):
        cid = order.cl_ord_id
        oms.on_execution_report(er(cid, ExecType.NEW, OrdStatus.NEW))
        oms.on_execution_report(
            er(cid, ExecType.TRADE, OrdStatus.FILLED, last_qty=100, last_px=3450)
        )
        with pytest.raises(OrderStateError):
            oms.on_execution_report(er(cid, ExecType.REJECTED, OrdStatus.REJECTED))

    def test_overfill_is_refused(self, oms, order):
        """Booking more than we ordered would silently corrupt position."""
        cid = order.cl_ord_id
        oms.on_execution_report(er(cid, ExecType.NEW, OrdStatus.NEW))
        with pytest.raises(OrderStateError, match="overfill"):
            oms.on_execution_report(
                er(cid, ExecType.TRADE, OrdStatus.FILLED, last_qty=101, last_px=3450)
            )
        assert oms.position("D05") == 0

    def test_trade_without_price_or_qty(self, oms, order):
        cid = order.cl_ord_id
        oms.on_execution_report(er(cid, ExecType.NEW, OrdStatus.NEW))
        with pytest.raises(OrderStateError, match="no qty/px"):
            oms.on_execution_report(
                er(cid, ExecType.TRADE, OrdStatus.PARTIALLY_FILLED, last_qty=0)
            )

    def test_cum_qty_mismatch_is_surfaced(self, oms, order):
        cid = order.cl_ord_id
        oms.on_execution_report(er(cid, ExecType.NEW, OrdStatus.NEW))
        with pytest.raises(OrderStateError, match="CumQty mismatch"):
            oms.on_execution_report(
                er(cid, ExecType.PENDING_CANCEL, OrdStatus.PENDING_CANCEL, cum_qty=50)
            )


class TestExposure:
    def test_working_orders_count_toward_the_limit(self, oms):
        o = oms.create("D05", Side.BUY, 100, 3450)
        oms.on_execution_report(er(o.cl_ord_id, ExecType.NEW, OrdStatus.NEW))
        max_long, max_short = oms.exposure("D05")
        assert max_long == 100  # flat now, but 100 long if it fills
        assert max_short == 0

    def test_both_directions(self, oms):
        b = oms.create("D05", Side.BUY, 100, 3450)
        s = oms.create("D05", Side.SELL, 60, 3500)
        for o in (b, s):
            oms.on_execution_report(er(o.cl_ord_id, ExecType.NEW, OrdStatus.NEW))
        assert oms.exposure("D05") == (100, -60)

    def test_terminal_orders_drop_out(self, oms):
        o = oms.create("D05", Side.BUY, 100, 3450)
        oms.on_execution_report(er(o.cl_ord_id, ExecType.NEW, OrdStatus.NEW))
        oms.on_execution_report(er(o.cl_ord_id, ExecType.CANCELED, OrdStatus.CANCELED))
        assert oms.exposure("D05") == (0, 0)
        assert oms.working_orders("D05") == []


class TestPnL:
    def test_round_trip_profit(self, oms):
        b = oms.create("D05", Side.BUY, 10, 100)
        oms.on_execution_report(er(b.cl_ord_id, ExecType.NEW, OrdStatus.NEW))
        oms.on_execution_report(
            er(b.cl_ord_id, ExecType.TRADE, OrdStatus.FILLED, last_qty=10, last_px=100)
        )
        s = oms.create("D05", Side.SELL, 10, 110)
        oms.on_execution_report(er(s.cl_ord_id, ExecType.NEW, OrdStatus.NEW))
        oms.on_execution_report(
            er(s.cl_ord_id, ExecType.TRADE, OrdStatus.FILLED, last_qty=10, last_px=110)
        )
        assert oms.realised_pnl("D05", 105) == pytest.approx(100.0)

    def test_open_position_is_marked_to_market(self, oms):
        b = oms.create("D05", Side.BUY, 10, 100)
        oms.on_execution_report(er(b.cl_ord_id, ExecType.NEW, OrdStatus.NEW))
        oms.on_execution_report(
            er(b.cl_ord_id, ExecType.TRADE, OrdStatus.FILLED, last_qty=10, last_px=100)
        )
        assert oms.realised_pnl("D05", 105) == pytest.approx(50.0)
        assert oms.realised_pnl("D05", 95) == pytest.approx(-50.0)

    def test_other_symbols_are_excluded(self, oms):
        b = oms.create("OTHER", Side.BUY, 10, 100)
        oms.on_execution_report(er(b.cl_ord_id, ExecType.NEW, OrdStatus.NEW))
        oms.on_execution_report(
            er(b.cl_ord_id, ExecType.TRADE, OrdStatus.FILLED, last_qty=10, last_px=100)
        )
        assert oms.realised_pnl("D05", 105) == 0.0
