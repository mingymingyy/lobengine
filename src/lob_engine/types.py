"""Shared value types.

Prices are carried as integers ("ticks") everywhere inside the engine.
Feeds hand us decimal strings; we scale them exactly with `decimal.Decimal`
and never let a float touch a dictionary key. Float keys silently break
book maintenance the moment 0.1 + 0.2 shows up.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from enum import Enum


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY

    @property
    def sign(self) -> int:
        """+1 for buy, -1 for sell. Handy for position arithmetic."""
        return 1 if self is Side.BUY else -1


class PriceScale:
    """Converts between decimal price strings and integer ticks.

    A tick is the smallest price increment the venue quotes. Storing
    6812345 rather than 68123.45 makes every comparison and dict lookup
    exact.
    """

    __slots__ = ("decimals", "_factor", "_q")

    def __init__(self, decimals: int) -> None:
        if decimals < 0:
            raise ValueError("decimals must be >= 0")
        self.decimals = decimals
        self._factor = 10**decimals
        self._q = Decimal(1).scaleb(-decimals)

    def to_ticks(self, price: str | float | int | Decimal) -> int:
        # Fast path. Venues send prices as plain decimal strings, and
        # building a Decimal for each one is the single most expensive
        # thing in the parse stage. String arithmetic on the digits is
        # exact, so this is a speed change and not an accuracy change;
        # anything unusual falls through to the Decimal path below.
        if type(price) is str:
            neg = price.startswith("-")
            body = price[1:] if neg else price
            int_part, sep, frac = body.partition(".")
            if int_part.isdigit() and (not sep or frac.isdigit()):
                if len(frac) > self.decimals:
                    if frac[self.decimals :].strip("0"):
                        raise ValueError(
                            f"price {price!r} has finer precision than "
                            f"{self.decimals} dp"
                        )
                    frac = frac[: self.decimals]
                ticks = int(int_part + frac.ljust(self.decimals, "0"))
                return -ticks if neg else ticks

        try:
            d = Decimal(str(price))
        except InvalidOperation as exc:
            # Decimal raises ArithmeticError, not ValueError. Normalising
            # here means every caller only has to handle one exception
            # type for "that is not a price".
            raise ValueError(f"not a valid price: {price!r}") from exc
        if not d.is_finite():
            raise ValueError(f"not a finite price: {price!r}")
        scaled = d.scaleb(self.decimals)
        rounded = scaled.to_integral_value(rounding=ROUND_HALF_EVEN)
        if rounded != scaled:
            # Venue sent more precision than we were configured for. Refuse
            # rather than silently round: it means the config is wrong.
            raise ValueError(
                f"price {price!r} has finer precision than {self.decimals} dp"
            )
        return int(rounded)

    def to_price(self, ticks: int) -> Decimal:
        return (Decimal(ticks) / self._factor).quantize(self._q)

    def to_float(self, ticks: int) -> float:
        return ticks / self._factor

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"PriceScale(decimals={self.decimals})"


@dataclass(slots=True)
class Level:
    """One price level of a Level-2 book.

    Not frozen, deliberately. Nothing in the engine mutates a Level, but
    `frozen=True` routes every field assignment through `object.__setattr__`,
    which costs about 0.16 microseconds per instance. On a snapshot with 25
    levels a side that is real time in the parse stage, so the immutability
    here is a convention rather than something enforced by the runtime.
    """

    price: int  # ticks
    size: float


@dataclass(slots=True)
class TopOfBook:
    """Best bid and offer at a point in time.

    `bid_px`/`ask_px` are None when that side of the book is empty.
    """

    bid_px: int | None
    bid_sz: float
    ask_px: int | None
    ask_sz: float
    exchange_ts_ns: int | None = None
    recv_ts_ns: int = 0

    @property
    def is_two_sided(self) -> bool:
        return self.bid_px is not None and self.ask_px is not None

    @property
    def mid(self) -> float | None:
        if not self.is_two_sided:
            return None
        return (self.bid_px + self.ask_px) / 2.0

    @property
    def spread(self) -> int | None:
        if not self.is_two_sided:
            return None
        return self.ask_px - self.bid_px

    @property
    def microprice(self) -> float | None:
        """Size-weighted mid.

        Standard convention: the weights are swapped, so a heavy bid and
        a thin ask pull the microprice up toward the ask. The reading is
        that a big bid queue is buying pressure, and empirically the mid
        tends to follow. Note this is the opposite of the queue-depletion
        intuition, where a big queue is one that is hard to clear.
        """
        if not self.is_two_sided:
            return None
        total = self.bid_sz + self.ask_sz
        if total <= 0:
            return self.mid
        return (self.bid_px * self.ask_sz + self.ask_px * self.bid_sz) / total


@dataclass(slots=True)
class BookUpdate:
    """A venue-neutral book message.

    Every feed adapter normalises into this shape, so the book, the
    recorder and the replay path never learn anything venue-specific.

    seq / prev_seq
        Venue sequence numbers. `prev_seq` is the sequence the venue says
        this message follows. When a venue only publishes `seq`, adapters
        set `prev_seq` to the previous `seq` they saw on the wire, so gap
        detection still works.
    """

    symbol: str
    bids: tuple[Level, ...] = ()
    asks: tuple[Level, ...] = ()
    is_snapshot: bool = False
    seq: int | None = None
    prev_seq: int | None = None
    exchange_ts_ns: int | None = None
    recv_ts_ns: int = field(default_factory=time.time_ns)
    venue: str = ""
