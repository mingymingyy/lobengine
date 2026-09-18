"""Order management.

An OMS does one job well: it always knows the true state of every order
you have working. That sounds trivial until execution reports arrive out
of order, a cancel races a fill, or a venue sends a fill for an order you
believe is already done.

The state machine below follows the FIX 4.4 order state transition
matrix. Every transition is explicitly allowed or rejected; there is no
"just set the status and hope". An illegal transition raises rather than
silently corrupting position, because a wrong position is worse than a
crash.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

from .fix import ExecType, OrdStatus, OrdType, TimeInForce
from .types import Side


class OrderStateError(Exception):
    """An execution report implied an illegal state transition."""


class UnknownOrder(Exception):
    """An execution report referenced an order we never sent."""


# FIX 4.4 order state transitions. Key: current status. Value: statuses it
# is legal to move to. Self-transitions are allowed where a status can
# repeat (e.g. successive partial fills).
_ALLOWED: dict[OrdStatus, set[OrdStatus]] = {
    OrdStatus.PENDING_NEW: {
        OrdStatus.PENDING_NEW,
        OrdStatus.NEW,
        OrdStatus.REJECTED,
        OrdStatus.PARTIALLY_FILLED,
        OrdStatus.FILLED,
        OrdStatus.CANCELED,
        OrdStatus.EXPIRED,
    },
    OrdStatus.NEW: {
        OrdStatus.NEW,
        OrdStatus.PARTIALLY_FILLED,
        OrdStatus.FILLED,
        OrdStatus.CANCELED,
        OrdStatus.PENDING_CANCEL,
        OrdStatus.PENDING_REPLACE,
        OrdStatus.REPLACED,
        OrdStatus.EXPIRED,
        OrdStatus.REJECTED,
    },
    OrdStatus.PARTIALLY_FILLED: {
        OrdStatus.PARTIALLY_FILLED,
        OrdStatus.FILLED,
        OrdStatus.CANCELED,
        OrdStatus.PENDING_CANCEL,
        OrdStatus.PENDING_REPLACE,
        OrdStatus.REPLACED,
        OrdStatus.EXPIRED,
    },
    OrdStatus.PENDING_CANCEL: {
        OrdStatus.PENDING_CANCEL,
        OrdStatus.CANCELED,
        OrdStatus.PARTIALLY_FILLED,
        OrdStatus.FILLED,
        OrdStatus.NEW,  # cancel rejected, order still working
        OrdStatus.EXPIRED,
    },
    OrdStatus.PENDING_REPLACE: {
        OrdStatus.PENDING_REPLACE,
        OrdStatus.REPLACED,
        OrdStatus.PARTIALLY_FILLED,
        OrdStatus.FILLED,
        OrdStatus.CANCELED,
        OrdStatus.NEW,  # replace rejected
        OrdStatus.EXPIRED,
    },
    OrdStatus.REPLACED: {OrdStatus.REPLACED},
    # terminal
    OrdStatus.FILLED: set(),
    OrdStatus.CANCELED: set(),
    OrdStatus.REJECTED: set(),
    OrdStatus.EXPIRED: set(),
}

TERMINAL = {
    OrdStatus.FILLED,
    OrdStatus.CANCELED,
    OrdStatus.REJECTED,
    OrdStatus.EXPIRED,
    OrdStatus.REPLACED,
}

_EXEC_TO_STATUS = {
    ExecType.NEW: OrdStatus.NEW,
    ExecType.PENDING_NEW: OrdStatus.PENDING_NEW,
    ExecType.REJECTED: OrdStatus.REJECTED,
    ExecType.CANCELED: OrdStatus.CANCELED,
    ExecType.PENDING_CANCEL: OrdStatus.PENDING_CANCEL,
    ExecType.REPLACED: OrdStatus.REPLACED,
    ExecType.PENDING_REPLACE: OrdStatus.PENDING_REPLACE,
    ExecType.EXPIRED: OrdStatus.EXPIRED,
}


@dataclass(slots=True)
class Order:
    """One order as we believe it to be."""

    cl_ord_id: str
    symbol: str
    side: Side
    qty: float
    price: int | None  # ticks; None for market orders
    ord_type: OrdType = OrdType.LIMIT
    tif: TimeInForce = TimeInForce.DAY
    status: OrdStatus = OrdStatus.PENDING_NEW
    order_id: str | None = None  # venue-assigned
    orig_cl_ord_id: str | None = None
    cum_qty: float = 0.0
    avg_px: float = 0.0
    created_ns: int = field(default_factory=time.time_ns)
    updated_ns: int = field(default_factory=time.time_ns)
    reject_reason: str = ""

    @property
    def leaves_qty(self) -> float:
        if self.status in TERMINAL:
            return 0.0
        return max(0.0, self.qty - self.cum_qty)

    @property
    def is_working(self) -> bool:
        return self.status not in TERMINAL

    @property
    def signed_filled(self) -> float:
        return self.cum_qty * self.side.sign


@dataclass(slots=True)
class Fill:
    """A single execution."""

    cl_ord_id: str
    symbol: str
    side: Side
    qty: float
    price: int
    exec_id: str
    ts_ns: int = field(default_factory=time.time_ns)


@dataclass(slots=True)
class ExecutionReport:
    """Venue-neutral execution report, mirroring FIX 35=8."""

    cl_ord_id: str
    exec_type: ExecType
    ord_status: OrdStatus
    order_id: str | None = None
    exec_id: str = ""
    last_qty: float = 0.0
    last_px: int | None = None
    cum_qty: float | None = None
    text: str = ""
    ts_ns: int = field(default_factory=time.time_ns)


class OMS:
    """Tracks orders, positions and fills for one trading session."""

    def __init__(self, id_prefix: str = "ORD") -> None:
        self.orders: dict[str, Order] = {}
        self.fills: list[Fill] = []
        self.positions: dict[str, float] = {}
        self._counter = itertools.count(1)
        self._prefix = id_prefix
        self.rejected_transitions = 0

    # ------------------------------------------------------------------ #
    def next_cl_ord_id(self) -> str:
        return f"{self._prefix}-{next(self._counter):08d}"

    def create(
        self,
        symbol: str,
        side: Side,
        qty: float,
        price: int | None,
        ord_type: OrdType = OrdType.LIMIT,
        tif: TimeInForce = TimeInForce.DAY,
        cl_ord_id: str | None = None,
    ) -> Order:
        """Register a new order in PENDING_NEW. Does not send anything."""
        if qty <= 0:
            raise ValueError("qty must be positive")
        if ord_type is OrdType.LIMIT and price is None:
            raise ValueError("limit order requires a price")
        cid = cl_ord_id or self.next_cl_ord_id()
        if cid in self.orders:
            raise ValueError(f"duplicate ClOrdID {cid}")
        order = Order(
            cl_ord_id=cid,
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            ord_type=ord_type,
            tif=tif,
        )
        self.orders[cid] = order
        return order

    def get(self, cl_ord_id: str) -> Order:
        try:
            return self.orders[cl_ord_id]
        except KeyError as exc:
            raise UnknownOrder(cl_ord_id) from exc

    # ------------------------------------------------------------------ #
    def on_execution_report(self, er: ExecutionReport) -> Order:
        """Apply an execution report. Returns the updated order."""
        order = self.get(er.cl_ord_id)
        new_status = er.ord_status

        if new_status not in _ALLOWED.get(order.status, set()):
            self.rejected_transitions += 1
            raise OrderStateError(
                f"{er.cl_ord_id}: illegal transition "
                f"{order.status.name} -> {new_status.name} "
                f"(ExecType={er.exec_type.name})"
            )

        if er.order_id:
            order.order_id = er.order_id

        if er.exec_type is ExecType.TRADE:
            if er.last_qty <= 0 or er.last_px is None:
                raise OrderStateError(f"{er.cl_ord_id}: trade with no qty/px")
            if er.last_qty > order.leaves_qty + 1e-9:
                raise OrderStateError(
                    f"{er.cl_ord_id}: overfill, last_qty={er.last_qty} "
                    f"leaves={order.leaves_qty}"
                )
            notional = order.avg_px * order.cum_qty + er.last_px * er.last_qty
            order.cum_qty += er.last_qty
            order.avg_px = notional / order.cum_qty
            self.fills.append(
                Fill(
                    cl_ord_id=order.cl_ord_id,
                    symbol=order.symbol,
                    side=order.side,
                    qty=er.last_qty,
                    price=er.last_px,
                    exec_id=er.exec_id,
                    ts_ns=er.ts_ns,
                )
            )
            self.positions[order.symbol] = (
                self.positions.get(order.symbol, 0.0) + er.last_qty * order.side.sign
            )
        elif er.cum_qty is not None and er.cum_qty != order.cum_qty:
            # Some venues restate CumQty on non-trade reports. Trust our
            # own tally of fills instead; a mismatch is worth surfacing.
            raise OrderStateError(
                f"{er.cl_ord_id}: CumQty mismatch, venue={er.cum_qty} "
                f"local={order.cum_qty}"
            )

        if er.exec_type is ExecType.REJECTED:
            order.reject_reason = er.text

        order.status = new_status
        order.updated_ns = er.ts_ns
        return order

    # ------------------------------------------------------------------ #
    def position(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0)

    def working_orders(self, symbol: str | None = None) -> list[Order]:
        return [
            o
            for o in self.orders.values()
            if o.is_working and (symbol is None or o.symbol == symbol)
        ]

    def exposure(self, symbol: str) -> tuple[float, float]:
        """(max long, max short) position if every working order filled.

        Used by the risk layer: checking the current position alone lets
        you build an unbounded position through resting orders.
        """
        pos = self.position(symbol)
        buys = sum(o.leaves_qty for o in self.working_orders(symbol) if o.side is Side.BUY)
        sells = sum(
            o.leaves_qty for o in self.working_orders(symbol) if o.side is Side.SELL
        )
        return pos + buys, pos - sells

    def realised_pnl(self, symbol: str, mark_px: float) -> float:
        """Mark-to-market P&L in ticks * units, from fills only."""
        cash = 0.0
        pos = 0.0
        for f in self.fills:
            if f.symbol != symbol:
                continue
            cash -= f.price * f.qty * f.side.sign
            pos += f.qty * f.side.sign
        return cash + pos * mark_px
