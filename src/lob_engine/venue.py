"""A mock exchange: continuous double auction with price-time priority.

This is the counterparty the OMS talks to in tests and simulations. It is
deliberately simple but it is honest about the one thing most toy
backtests get wrong: **queue position**.

If you post a limit order at the best bid, you do not get filled just
because a trade prints at that price. You get filled when the trades and
cancellations ahead of you in the queue have cleared. A backtest that
assumes otherwise will show a market-making strategy making money that it
would never have made. Here every resting order knows how much size sits
in front of it, and that number only decreases through actual executions
and cancels ahead of it.

Matching rules:
    * price priority first, then time priority within a price level
    * a marketable limit order walks the book and can fill at multiple
      prices
    * IOC cancels the remainder, FOK is all-or-nothing
    * partial fills generate one ExecutionReport each
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from .fix import ExecType, OrdStatus, OrdType, TimeInForce
from .oms import ExecutionReport
from .types import Side


@dataclass(slots=True)
class RestingOrder:
    """An order sitting in the book."""

    order_id: str
    cl_ord_id: str | None  # None for background liquidity
    side: Side
    price: int
    qty: float
    seq: int  # arrival order, for time priority
    is_ours: bool = False

    remaining: float = field(init=False)

    def __post_init__(self) -> None:
        self.remaining = self.qty


class MockVenue:
    """Single-symbol matching engine.

    `on_exec` is called with each ExecutionReport for one of *our* orders.
    Background liquidity does not generate reports.
    """

    def __init__(self, symbol: str, on_exec=None) -> None:
        self.symbol = symbol
        self.on_exec = on_exec or (lambda er: None)
        self.bids: dict[int, list[RestingOrder]] = {}
        self.asks: dict[int, list[RestingOrder]] = {}
        self._order_seq = itertools.count(1)
        self._id_seq = itertools.count(1)
        self._exec_seq = itertools.count(1)
        self.by_id: dict[str, RestingOrder] = {}
        self.by_cl_ord_id: dict[str, RestingOrder] = {}
        self.trades: list[tuple[int, float, Side]] = []  # price, qty, aggressor

    # ------------------------------------------------------------------ #
    # book helpers
    # ------------------------------------------------------------------ #
    def _side_book(self, side: Side) -> dict[int, list[RestingOrder]]:
        return self.bids if side is Side.BUY else self.asks

    def best_bid(self) -> int | None:
        live = [p for p, q in self.bids.items() if q]
        return max(live) if live else None

    def best_ask(self) -> int | None:
        live = [p for p, q in self.asks.items() if q]
        return min(live) if live else None

    def mid(self) -> float | None:
        b, a = self.best_bid(), self.best_ask()
        return (b + a) / 2.0 if b is not None and a is not None else None

    def size_at(self, side: Side, price: int) -> float:
        return sum(o.remaining for o in self._side_book(side).get(price, []))

    def queue_ahead(self, cl_ord_id: str) -> float:
        """Size resting ahead of our order at its price level."""
        o = self.by_cl_ord_id.get(cl_ord_id)
        if o is None:
            return 0.0
        level = self._side_book(o.side).get(o.price, [])
        ahead = 0.0
        for other in level:
            if other is o:
                break
            ahead += other.remaining
        return ahead

    # ------------------------------------------------------------------ #
    # background liquidity
    # ------------------------------------------------------------------ #
    def add_liquidity(self, side: Side, price: int, qty: float) -> RestingOrder:
        """Post anonymous resting size. Used to build a starting book."""
        return self._rest(side, price, qty, cl_ord_id=None, is_ours=False)

    def _rest(
        self, side: Side, price: int, qty: float, cl_ord_id: str | None, is_ours: bool
    ) -> RestingOrder:
        o = RestingOrder(
            order_id=f"V{next(self._id_seq):08d}",
            cl_ord_id=cl_ord_id,
            side=side,
            price=price,
            qty=qty,
            seq=next(self._order_seq),
            is_ours=is_ours,
        )
        self._side_book(side).setdefault(price, []).append(o)
        self.by_id[o.order_id] = o
        if cl_ord_id:
            self.by_cl_ord_id[cl_ord_id] = o
        return o

    # ------------------------------------------------------------------ #
    # order entry
    # ------------------------------------------------------------------ #
    def submit(
        self,
        cl_ord_id: str,
        side: Side,
        qty: float,
        price: int | None,
        ord_type: OrdType = OrdType.LIMIT,
        tif: TimeInForce = TimeInForce.DAY,
    ) -> None:
        """Submit one of our orders. Emits ExecutionReports via `on_exec`."""
        if cl_ord_id in self.by_cl_ord_id:
            self._emit(cl_ord_id, ExecType.REJECTED, OrdStatus.REJECTED,
                       text="duplicate ClOrdID")
            return
        if ord_type is OrdType.LIMIT and price is None:
            self._emit(cl_ord_id, ExecType.REJECTED, OrdStatus.REJECTED,
                       text="limit order without price")
            return

        if tif is TimeInForce.FOK and not self._can_fill_fully(side, qty, price):
            self._emit(cl_ord_id, ExecType.REJECTED, OrdStatus.REJECTED,
                       text="FOK cannot be filled in full")
            return

        order_id = f"V{next(self._id_seq):08d}"
        self._emit(cl_ord_id, ExecType.NEW, OrdStatus.NEW, order_id=order_id)

        remaining, cum, fills = self._match(cl_ord_id, side, qty, price, order_id)

        if remaining <= 1e-12:
            return  # fully filled; _match already emitted the final report

        if ord_type is OrdType.MARKET or tif in (TimeInForce.IOC, TimeInForce.FOK):
            status = OrdStatus.CANCELED
            self._emit(cl_ord_id, ExecType.CANCELED, status,
                       order_id=order_id, cum_qty=cum,
                       text="remainder cancelled")
            return

        resting = self._rest(side, price, remaining, cl_ord_id, is_ours=True)
        resting.order_id = order_id
        self.by_id[order_id] = resting

    def cancel(self, cl_ord_id: str) -> None:
        o = self.by_cl_ord_id.get(cl_ord_id)
        if o is None or o.remaining <= 0:
            self._emit(cl_ord_id, ExecType.REJECTED, OrdStatus.REJECTED,
                       text="unknown or already done")
            return
        self._remove(o)
        self._emit(cl_ord_id, ExecType.CANCELED, OrdStatus.CANCELED,
                   order_id=o.order_id, cum_qty=o.qty - o.remaining)

    # ------------------------------------------------------------------ #
    # matching
    # ------------------------------------------------------------------ #
    def _opposite_levels(self, side: Side, limit_px: int | None) -> list[int]:
        book = self._side_book(side.opposite)
        prices = [p for p, q in book.items() if any(o.remaining > 0 for o in q)]
        if side is Side.BUY:
            prices.sort()
            if limit_px is not None:
                prices = [p for p in prices if p <= limit_px]
        else:
            prices.sort(reverse=True)
            if limit_px is not None:
                prices = [p for p in prices if p >= limit_px]
        return prices

    def _can_fill_fully(self, side: Side, qty: float, price: int | None) -> bool:
        available = sum(
            self.size_at(side.opposite, p) for p in self._opposite_levels(side, price)
        )
        return available >= qty - 1e-12

    def _match(
        self, cl_ord_id: str, side: Side, qty: float, price: int | None, order_id: str
    ) -> tuple[float, float, int]:
        remaining = qty
        cum = 0.0
        n_fills = 0
        book = self._side_book(side.opposite)

        for lvl_px in self._opposite_levels(side, price):
            queue = book.get(lvl_px, [])
            i = 0
            while i < len(queue) and remaining > 1e-12:
                resting = queue[i]
                if resting.remaining <= 0:
                    i += 1
                    continue
                traded = min(remaining, resting.remaining)
                resting.remaining -= traded
                remaining -= traded
                cum += traded
                n_fills += 1
                self.trades.append((lvl_px, traded, side))

                # passive side report, if it was ours
                if resting.is_ours and resting.cl_ord_id:
                    p_status = (
                        OrdStatus.FILLED
                        if resting.remaining <= 1e-12
                        else OrdStatus.PARTIALLY_FILLED
                    )
                    self._emit(
                        resting.cl_ord_id,
                        ExecType.TRADE,
                        p_status,
                        order_id=resting.order_id,
                        last_qty=traded,
                        last_px=lvl_px,
                    )
                # aggressive side report
                a_status = (
                    OrdStatus.FILLED if remaining <= 1e-12 else OrdStatus.PARTIALLY_FILLED
                )
                self._emit(
                    cl_ord_id,
                    ExecType.TRADE,
                    a_status,
                    order_id=order_id,
                    last_qty=traded,
                    last_px=lvl_px,
                )
                if resting.remaining <= 1e-12:
                    i += 1
            self._compact(book, lvl_px)
            if remaining <= 1e-12:
                break
        return remaining, cum, n_fills

    def _compact(self, book: dict[int, list[RestingOrder]], price: int) -> None:
        queue = book.get(price)
        if queue is None:
            return
        keep = []
        for o in queue:
            if o.remaining > 1e-12:
                keep.append(o)
            elif o.cl_ord_id:
                self.by_cl_ord_id.pop(o.cl_ord_id, None)
        if keep:
            book[price] = keep
        else:
            del book[price]

    def _remove(self, o: RestingOrder) -> None:
        book = self._side_book(o.side)
        queue = book.get(o.price, [])
        book[o.price] = [x for x in queue if x is not o]
        if not book[o.price]:
            del book[o.price]
        if o.cl_ord_id:
            self.by_cl_ord_id.pop(o.cl_ord_id, None)
        o.remaining = 0.0

    # ------------------------------------------------------------------ #
    # external flow
    # ------------------------------------------------------------------ #
    def market_order(self, side: Side, qty: float) -> float:
        """Anonymous aggressive flow. Returns the quantity actually traded.

        This is how our resting orders get filled: someone else crosses
        the spread and works through the queue, us included.
        """
        filled, _, _ = self._match("__external__", side, qty, None, "__external__")
        return qty - filled

    def cancel_ahead(self, side: Side, price: int, qty: float) -> float:
        """Cancel `qty` of anonymous size at a level, oldest first.

        Modelled explicitly because cancellations ahead of you are how
        queue position improves without a trade printing.
        """
        queue = self._side_book(side).get(price, [])
        left = qty
        for o in queue:
            if left <= 1e-12:
                break
            if o.is_ours or o.remaining <= 0:
                continue
            take = min(left, o.remaining)
            o.remaining -= take
            left -= take
        self._compact(self._side_book(side), price)
        return qty - left

    # ------------------------------------------------------------------ #
    def _emit(
        self,
        cl_ord_id: str,
        exec_type: ExecType,
        ord_status: OrdStatus,
        order_id: str | None = None,
        last_qty: float = 0.0,
        last_px: int | None = None,
        cum_qty: float | None = None,
        text: str = "",
    ) -> None:
        if cl_ord_id == "__external__":
            return
        self.on_exec(
            ExecutionReport(
                cl_ord_id=cl_ord_id,
                exec_type=exec_type,
                ord_status=ord_status,
                order_id=order_id,
                exec_id=f"E{next(self._exec_seq):08d}",
                last_qty=last_qty,
                last_px=last_px,
                cum_qty=cum_qty,
                text=text,
            )
        )
