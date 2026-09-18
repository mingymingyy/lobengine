"""Level-2 order book reconstruction.

A venue sends you one snapshot and then a stream of deltas. Your job is to
hold the resulting book in memory and never let it drift from the venue's.
Three things go wrong in practice and all three are handled here:

1. Messages arrive out of order or one goes missing. Detected via sequence
   numbers; the book raises `SequenceGap` and refuses to apply the update.
   The caller is expected to resnapshot rather than carry on with a book
   it can no longer trust.
2. The book ends up crossed (bid >= ask) because a delta was mis-applied.
   Detected by `check_integrity`.
3. A delta removes a level you never had. Tracked as a counter rather than
   an exception, because on a depth-truncated feed (top-N levels only) it
   is expected and harmless.
"""

from __future__ import annotations

from bisect import bisect_left, insort
from dataclasses import dataclass

from .types import BookUpdate, Level, Side, TopOfBook


class SequenceGap(Exception):
    """Raised when an update does not follow the last one applied."""

    def __init__(self, expected: int | None, got: int | None) -> None:
        super().__init__(f"sequence gap: expected prev_seq={expected}, got {got}")
        self.expected = expected
        self.got = got


class CrossedBook(Exception):
    """Raised when the best bid is at or above the best ask."""


class _BookSide:
    """One side of the book: price(ticks) -> size.

    Prices are kept in a single ascending list so `bisect` can be used for
    both sides. Insert is O(n) worst case but it is a memmove on a list of
    a few hundred ints, which is far faster in CPython than a tree for
    realistic depths. Lookup of the best price is O(1).
    """

    __slots__ = ("_prices", "_sizes", "_is_bid")

    def __init__(self, is_bid: bool) -> None:
        self._is_bid = is_bid
        self._prices: list[int] = []  # always ascending
        self._sizes: dict[int, float] = {}

    def clear(self) -> None:
        self._prices.clear()
        self._sizes.clear()

    def set(self, price: int, size: float) -> bool:
        """Set the size at `price`. size <= 0 deletes the level.

        Returns True if the update changed the book, False if it was a
        no-op (a delete for a level we do not hold).
        """
        if size > 0:
            if price not in self._sizes:
                insort(self._prices, price)
            elif self._sizes[price] == size:
                return False
            self._sizes[price] = size
            return True
        # deletion
        if price in self._sizes:
            del self._sizes[price]
            i = bisect_left(self._prices, price)
            del self._prices[i]
            return True
        return False

    def best(self) -> tuple[int, float] | None:
        if not self._prices:
            return None
        p = self._prices[-1] if self._is_bid else self._prices[0]
        return p, self._sizes[p]

    def levels(self, depth: int | None = None) -> tuple[Level, ...]:
        """Top `depth` levels, best first."""
        prices = reversed(self._prices) if self._is_bid else iter(self._prices)
        out = []
        for i, p in enumerate(prices):
            if depth is not None and i >= depth:
                break
            out.append(Level(p, self._sizes[p]))
        return tuple(out)

    def total_size(self, depth: int | None = None) -> float:
        return sum(lv.size for lv in self.levels(depth))

    def truncate(self, max_depth: int) -> None:
        """Drop levels beyond `max_depth` from the best price.

        Needed when following a top-N feed: without this, a level that
        falls out of the venue's published window would sit in our book
        forever, because the venue never sends a delete for it.
        """
        excess = len(self._prices) - max_depth
        if excess <= 0:
            return
        if self._is_bid:
            dropped, self._prices = self._prices[:excess], self._prices[excess:]
        else:
            self._prices, dropped = self._prices[:max_depth], self._prices[max_depth:]
        for p in dropped:
            del self._sizes[p]

    def __len__(self) -> int:
        return len(self._prices)


@dataclass(slots=True)
class BookStats:
    """Counters worth watching in production."""

    updates_applied: int = 0
    snapshots_applied: int = 0
    gaps_detected: int = 0
    no_op_deletes: int = 0
    crossed_seen: int = 0


class OrderBook:
    """A single-symbol L2 book.

    Parameters
    ----------
    symbol
        Instrument identifier, used only for error messages.
    max_depth
        If the feed publishes only the top N levels, set this to N so
        stale far-touch levels get trimmed. Leave None for a full-depth
        feed.
    strict_sequencing
        If True (default) an out-of-sequence update raises `SequenceGap`
        and is not applied.
    """

    __slots__ = ("symbol", "max_depth", "strict_sequencing", "bids", "asks",
                 "seq", "last_update_ns", "last_exchange_ts_ns", "stats", "_ready")

    def __init__(
        self,
        symbol: str,
        max_depth: int | None = None,
        strict_sequencing: bool = True,
    ) -> None:
        self.symbol = symbol
        self.max_depth = max_depth
        self.strict_sequencing = strict_sequencing
        self.bids = _BookSide(is_bid=True)
        self.asks = _BookSide(is_bid=False)
        self.seq: int | None = None
        self.last_update_ns: int = 0
        self.last_exchange_ts_ns: int | None = None
        self.stats = BookStats()
        self._ready = False

    # ------------------------------------------------------------------ #
    # state
    # ------------------------------------------------------------------ #
    @property
    def ready(self) -> bool:
        """True once a snapshot has been applied."""
        return self._ready

    def reset(self) -> None:
        """Forget everything. Call before applying a fresh snapshot."""
        self.bids.clear()
        self.asks.clear()
        self.seq = None
        self._ready = False

    # ------------------------------------------------------------------ #
    # ingestion
    # ------------------------------------------------------------------ #
    def apply(self, update: BookUpdate) -> None:
        """Apply one update.

        Raises
        ------
        SequenceGap
            If `strict_sequencing` and the update does not follow the last
            one applied. The book is left untouched, so the caller can
            resnapshot cleanly.
        """
        if update.is_snapshot:
            self.reset()
        else:
            if not self._ready:
                # A delta before any snapshot is meaningless. Treat it the
                # same as a gap so the caller goes and fetches a snapshot.
                self.stats.gaps_detected += 1
                raise SequenceGap(None, update.seq)
            self._check_sequence(update)

        for lv in update.bids:
            if not self.bids.set(lv.price, lv.size) and lv.size <= 0:
                self.stats.no_op_deletes += 1
        for lv in update.asks:
            if not self.asks.set(lv.price, lv.size) and lv.size <= 0:
                self.stats.no_op_deletes += 1

        if self.max_depth is not None:
            self.bids.truncate(self.max_depth)
            self.asks.truncate(self.max_depth)

        if update.seq is not None:
            self.seq = update.seq
        self.last_update_ns = update.recv_ts_ns
        self.last_exchange_ts_ns = update.exchange_ts_ns

        if update.is_snapshot:
            self._ready = True
            self.stats.snapshots_applied += 1
        else:
            self.stats.updates_applied += 1

    def _check_sequence(self, update: BookUpdate) -> None:
        if not self.strict_sequencing:
            return
        if update.prev_seq is None or self.seq is None:
            return
        if update.prev_seq != self.seq:
            self.stats.gaps_detected += 1
            raise SequenceGap(self.seq, update.prev_seq)

    # ------------------------------------------------------------------ #
    # reads
    # ------------------------------------------------------------------ #
    def top(self) -> TopOfBook:
        b = self.bids.best()
        a = self.asks.best()
        return TopOfBook(
            bid_px=b[0] if b else None,
            bid_sz=b[1] if b else 0.0,
            ask_px=a[0] if a else None,
            ask_sz=a[1] if a else 0.0,
            exchange_ts_ns=self.last_exchange_ts_ns,
            recv_ts_ns=self.last_update_ns,
        )

    def depth(self, side: Side, n: int | None = None) -> tuple[Level, ...]:
        return (self.bids if side is Side.BUY else self.asks).levels(n)

    def size_at(self, side: Side, price: int) -> float:
        book = self.bids if side is Side.BUY else self.asks
        return book._sizes.get(price, 0.0)

    def average_depth(self, n: int = 1) -> float:
        """Mean of top-n bid size and top-n ask size.

        Cont, Kukanov & Stoikov use average depth as the scaling variable
        for price impact, so it is computed here rather than at call sites.
        """
        return (self.bids.total_size(n) + self.asks.total_size(n)) / 2.0

    # ------------------------------------------------------------------ #
    # integrity
    # ------------------------------------------------------------------ #
    def check_integrity(self, raise_on_cross: bool = False) -> bool:
        """Return True if the book looks sane.

        A crossed book almost always means a delta was applied wrongly or
        a message was silently dropped.
        """
        b = self.bids.best()
        a = self.asks.best()
        if b is None or a is None:
            return True
        if b[0] >= a[0]:
            self.stats.crossed_seen += 1
            if raise_on_cross:
                raise CrossedBook(
                    f"{self.symbol}: bid {b[0]} >= ask {a[0]} at seq {self.seq}"
                )
            return False
        return True

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        t = self.top()
        return (
            f"<OrderBook {self.symbol} seq={self.seq} "
            f"{t.bid_sz}@{t.bid_px} / {t.ask_sz}@{t.ask_px}>"
        )
