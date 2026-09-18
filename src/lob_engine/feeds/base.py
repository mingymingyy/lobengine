"""Feed adapter contract.

An adapter does exactly two things:

    parse(raw_json, recv_ts_ns) -> list[BookUpdate]
    subscribe_payloads() -> list[str]     what to send after connecting

Everything venue-specific lives behind that boundary, so the book, the
recorder, the replay path and every test stay venue-agnostic. Parsing is
a *pure function* of the raw payload, which is why the whole pipeline can
be tested offline against recorded messages with no network at all.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import BookUpdate, PriceScale


class FeedAdapter(ABC):
    """Base class for venue adapters."""

    #: Human-readable venue name, stamped into captures.
    venue: str = "unknown"

    #: Max depth the channel publishes, or None for full depth. The book
    #: uses this to trim levels the venue will never send a delete for.
    max_depth: int | None = None

    def __init__(self, symbol: str, price_decimals: int) -> None:
        self.symbol = symbol
        self.scale = PriceScale(price_decimals)

    @abstractmethod
    def subscribe_payloads(self) -> list[str]:
        """Raw strings to send immediately after the socket opens."""

    @abstractmethod
    def parse(self, raw: str, recv_ts_ns: int) -> list[BookUpdate]:
        """Turn one raw websocket frame into zero or more BookUpdates.

        Must not raise on control frames (subscription acks, pongs); it
        should return an empty list for anything that is not book data.
        """

    def ping_payload(self) -> str | None:
        """Application-level keepalive, if the venue wants one."""
        return None

    def url(self) -> str:
        raise NotImplementedError
