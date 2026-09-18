"""Pre-trade risk controls.

Every order goes through here before it reaches the wire. This is not
optional decoration: MAS Notice SFA 04-N16 and the SGX member rules on
algorithmic trading both require a member firm to have automated
pre-trade controls and the ability to stop its own algos. A quant dev at
a broker spends real time on this layer.

The checks implemented, roughly in the order a desk cares about them:

    kill switch        one flag that stops everything
    tick / lot         the order is well formed for the venue
    max order qty      fat finger on size
    max order value    fat finger on notional
    price collar       fat finger on price, measured against the mid
    max position       projected position if all working orders fill
    max open orders    runaway quoting loop
    message throttle   sliding-window rate limit
    self-trade         would cross our own resting order

Each check returns a reason string rather than a bare bool so rejects can
be logged and counted, which is what you actually want at 3pm when the
desk asks why an order did not go.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .oms import OMS, Order
from .types import Side


@dataclass(slots=True)
class RiskLimits:
    """Per-symbol limits. All optional; None disables that check."""

    max_order_qty: float | None = None
    max_order_value: float | None = None  # qty * price, in price units
    max_position: float | None = None  # absolute, both directions
    max_open_orders: int | None = None
    price_collar_bps: float | None = None  # vs mid
    max_messages_per_sec: float | None = None
    lot_size: float | None = None
    tick_size: int | None = None  # in ticks; 1 means every tick is valid
    allow_market_orders: bool = True


@dataclass(slots=True)
class RiskDecision:
    accepted: bool
    reason: str = ""
    check: str = ""

    def __bool__(self) -> bool:
        return self.accepted


ACCEPT = RiskDecision(True)


@dataclass(slots=True)
class RiskStats:
    checked: int = 0
    accepted: int = 0
    rejected: int = 0
    by_check: dict[str, int] = field(default_factory=dict)

    def record(self, decision: RiskDecision) -> None:
        self.checked += 1
        if decision.accepted:
            self.accepted += 1
        else:
            self.rejected += 1
            self.by_check[decision.check] = self.by_check.get(decision.check, 0) + 1


class RiskEngine:
    """Stateful pre-trade gate.

    `price_scale_factor` converts ticks to price units for notional
    checks (10**decimals). Notional limits are quoted in price units
    because that is how a desk sets them.
    """

    def __init__(
        self,
        limits: RiskLimits,
        oms: OMS,
        price_scale_factor: float = 1.0,
        clock=None,
    ) -> None:
        import time as _time

        self.limits = limits
        self.oms = oms
        self.scale = float(price_scale_factor)
        self._clock = clock or _time.monotonic
        self._msg_times: deque[float] = deque()
        self.killed = False
        self.kill_reason = ""
        self.stats = RiskStats()

    # ------------------------------------------------------------------ #
    def kill(self, reason: str = "manual") -> None:
        """Stop all new orders. The one control a desk must always have."""
        self.killed = True
        self.kill_reason = reason

    def resume(self) -> None:
        self.killed = False
        self.kill_reason = ""

    # ------------------------------------------------------------------ #
    def check_new_order(
        self,
        symbol: str,
        side: Side,
        qty: float,
        price: int | None,
        mid: float | None = None,
        is_market: bool = False,
    ) -> RiskDecision:
        d = self._run_checks(symbol, side, qty, price, mid, is_market)
        self.stats.record(d)
        if d.accepted:
            self._msg_times.append(self._clock())
        return d

    def _run_checks(
        self,
        symbol: str,
        side: Side,
        qty: float,
        price: int | None,
        mid: float | None,
        is_market: bool,
    ) -> RiskDecision:
        L = self.limits

        if self.killed:
            return RiskDecision(False, f"kill switch active: {self.kill_reason}", "kill_switch")

        if qty <= 0:
            return RiskDecision(False, "quantity must be positive", "qty_sign")

        if is_market and not L.allow_market_orders:
            return RiskDecision(False, "market orders disabled", "market_disabled")

        if not is_market and price is None:
            return RiskDecision(False, "limit order without price", "missing_price")

        if L.lot_size and abs(qty / L.lot_size - round(qty / L.lot_size)) > 1e-9:
            return RiskDecision(
                False, f"qty {qty} is not a multiple of lot size {L.lot_size}", "lot_size"
            )

        if L.tick_size and price is not None and price % L.tick_size != 0:
            return RiskDecision(
                False, f"price {price} is not a multiple of tick size {L.tick_size}", "tick_size"
            )

        if L.max_order_qty is not None and qty > L.max_order_qty:
            return RiskDecision(
                False, f"qty {qty} exceeds max order qty {L.max_order_qty}", "max_order_qty"
            )

        if L.max_order_value is not None and price is not None:
            value = qty * price / self.scale
            if value > L.max_order_value:
                return RiskDecision(
                    False,
                    f"order value {value:.2f} exceeds max {L.max_order_value:.2f}",
                    "max_order_value",
                )

        if (
            L.price_collar_bps is not None
            and price is not None
            and mid is not None
            and mid > 0
        ):
            deviation_bps = abs(price - mid) / mid * 10_000.0
            if deviation_bps > L.price_collar_bps:
                return RiskDecision(
                    False,
                    f"price {price} is {deviation_bps:.1f} bps from mid "
                    f"(limit {L.price_collar_bps:.1f})",
                    "price_collar",
                )

        if L.max_position is not None:
            max_long, max_short = self.oms.exposure(symbol)
            projected = max_long + qty if side is Side.BUY else max_short - qty
            if abs(projected) > L.max_position:
                return RiskDecision(
                    False,
                    f"projected position {projected:+.4f} exceeds "
                    f"max position {L.max_position}",
                    "max_position",
                )

        if (
            L.max_open_orders is not None
            and len(self.oms.working_orders(symbol)) >= L.max_open_orders
        ):
            return RiskDecision(
                False,
                f"already at max open orders ({L.max_open_orders})",
                "max_open_orders",
            )

        if L.max_messages_per_sec is not None:
            now = self._clock()
            while self._msg_times and now - self._msg_times[0] > 1.0:
                self._msg_times.popleft()
            if len(self._msg_times) >= L.max_messages_per_sec:
                return RiskDecision(
                    False,
                    f"message rate {len(self._msg_times)}/s at limit "
                    f"{L.max_messages_per_sec}/s",
                    "throttle",
                )

        if price is not None:
            crossed = self._self_trade(symbol, side, price)
            if crossed is not None:
                return RiskDecision(
                    False,
                    f"would cross own resting order {crossed.cl_ord_id} "
                    f"at {crossed.price}",
                    "self_trade",
                )

        return ACCEPT

    def _self_trade(self, symbol: str, side: Side, price: int) -> Order | None:
        """Find a resting order on the other side that this would hit."""
        for o in self.oms.working_orders(symbol):
            if o.side is side or o.price is None:
                continue
            if side is Side.BUY and price >= o.price:
                return o
            if side is Side.SELL and price <= o.price:
                return o
        return None
