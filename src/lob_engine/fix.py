"""A minimal FIX 4.4 codec.

FIX is a tag=value protocol. Fields are separated by SOH (ASCII 0x01) and
every message is wrapped in three framing fields that have to be exactly
right or the counterparty drops the session:

    8=FIX.4.4<SOH>      BeginString, always first
    9=<n><SOH>          BodyLength, always second
    ...body...
    10=<ccc><SOH>       CheckSum, always last

BodyLength counts the bytes *after* the SOH that terminates tag 9, up to
and including the SOH that terminates the last body field. It excludes
the BeginString and BodyLength fields themselves, and excludes the
CheckSum field.

CheckSum is the sum of every byte from the first byte of "8=" up to and
including the SOH before "10=", taken modulo 256 and rendered as exactly
three digits with leading zeros.

Both of those definitions are implemented literally below and are checked
in the test suite against an independently written reference
implementation, because "looks about right" is how FIX sessions end up
disconnecting at 09:00.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

SOH = "\x01"
BEGIN_STRING = "FIX.4.4"


# --------------------------------------------------------------------- #
# tags
# --------------------------------------------------------------------- #
class Tag:
    BeginString = 8
    BodyLength = 9
    CheckSum = 10
    MsgType = 35
    SenderCompID = 49
    TargetCompID = 56
    MsgSeqNum = 34
    SendingTime = 52
    PossDupFlag = 43
    OrigSendingTime = 122

    # session
    HeartBtInt = 108
    EncryptMethod = 98
    TestReqID = 112
    BeginSeqNo = 7
    EndSeqNo = 16
    NewSeqNo = 36
    GapFillFlag = 123
    Text = 58
    RefSeqNum = 45

    # order
    ClOrdID = 11
    OrigClOrdID = 41
    OrderID = 37
    ExecID = 17
    ExecType = 150
    OrdStatus = 39
    Symbol = 55
    Side = 54
    OrderQty = 38
    OrdType = 40
    Price = 44
    TimeInForce = 59
    TransactTime = 60
    LastQty = 32
    LastPx = 31
    LeavesQty = 151
    CumQty = 14
    AvgPx = 6
    OrdRejReason = 103
    CxlRejReason = 102
    CxlRejResponseTo = 434
    ExecInst = 18


class MsgType(str, Enum):
    HEARTBEAT = "0"
    TEST_REQUEST = "1"
    RESEND_REQUEST = "2"
    REJECT = "3"
    SEQUENCE_RESET = "4"
    LOGOUT = "5"
    EXECUTION_REPORT = "8"
    ORDER_CANCEL_REJECT = "9"
    LOGON = "A"
    NEW_ORDER_SINGLE = "D"
    ORDER_CANCEL_REQUEST = "F"
    ORDER_CANCEL_REPLACE_REQUEST = "G"


class ExecType(str, Enum):
    NEW = "0"
    TRADE = "F"
    CANCELED = "4"
    REPLACED = "5"
    REJECTED = "8"
    PENDING_CANCEL = "6"
    PENDING_NEW = "A"
    PENDING_REPLACE = "E"
    EXPIRED = "C"


class OrdStatus(str, Enum):
    NEW = "0"
    PARTIALLY_FILLED = "1"
    FILLED = "2"
    CANCELED = "4"
    PENDING_CANCEL = "6"
    REJECTED = "8"
    PENDING_NEW = "A"
    EXPIRED = "C"
    PENDING_REPLACE = "E"
    REPLACED = "5"


class OrdType(str, Enum):
    MARKET = "1"
    LIMIT = "2"


class TimeInForce(str, Enum):
    DAY = "0"
    GTC = "1"
    IOC = "3"
    FOK = "4"


FIX_SIDE = {"BUY": "1", "SELL": "2"}
SIDE_FIX = {v: k for k, v in FIX_SIDE.items()}


class FixError(Exception):
    """Malformed or unverifiable FIX message."""


# --------------------------------------------------------------------- #
# framing
# --------------------------------------------------------------------- #
def checksum(payload: str) -> str:
    """Three-digit FIX checksum of everything up to (not including) 10=."""
    total = sum(payload.encode("ascii")) % 256
    return f"{total:03d}"


def _wire(value) -> str:
    """Render a field value for the wire.

    Enums are written as their value, never their name. `str(SomeEnum.X)`
    on a str-mixin enum gives "SomeEnum.X" on modern Python, which would
    put `35=MsgType.NEW_ORDER_SINGLE` on the wire and get the session
    dropped. This is the kind of bug that only shows up against a real
    counterparty, so it is handled once, here.
    """
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def encode(msg_type, body: list[tuple[int, str]]) -> str:
    """Frame a message.

    `body` must already contain the session fields (49, 56, 34, 52) in
    the order they should appear. MsgType is prepended here because tag
    35 must be the third field.
    """
    fields = [(Tag.MsgType, _wire(msg_type))] + [(t, _wire(v)) for t, v in body]
    body_str = "".join(f"{t}={v}{SOH}" for t, v in fields)
    head = f"{Tag.BeginString}={BEGIN_STRING}{SOH}{Tag.BodyLength}={len(body_str)}{SOH}"
    prefix = head + body_str
    return f"{prefix}{Tag.CheckSum}={checksum(prefix)}{SOH}"


@dataclass(slots=True)
class FixMessage:
    """A decoded message. `fields` preserves wire order and duplicates."""

    fields: list[tuple[int, str]] = field(default_factory=list)

    def get(self, tag: int, default: str | None = None) -> str | None:
        for t, v in self.fields:
            if t == tag:
                return v
        return default

    def require(self, tag: int) -> str:
        v = self.get(tag)
        if v is None:
            raise FixError(f"missing required tag {tag}")
        return v

    def get_int(self, tag: int, default: int | None = None) -> int | None:
        v = self.get(tag)
        return int(v) if v is not None else default

    def get_float(self, tag: int, default: float | None = None) -> float | None:
        v = self.get(tag)
        return float(v) if v is not None else default

    @property
    def msg_type(self) -> str:
        return self.require(Tag.MsgType)

    @property
    def seq_num(self) -> int:
        return int(self.require(Tag.MsgSeqNum))

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        inner = " ".join(f"{t}={v}" for t, v in self.fields)
        return f"<FIX {inner}>"


def decode(raw: str, verify: bool = True) -> FixMessage:
    """Parse one complete FIX message.

    Raises `FixError` on bad framing, a bad BodyLength or a bad CheckSum.
    """
    if not raw.endswith(SOH):
        raise FixError("message does not end with SOH")
    parts = raw.split(SOH)[:-1]
    if len(parts) < 3:
        raise FixError("too few fields")

    fields: list[tuple[int, str]] = []
    for p in parts:
        tag_str, sep, value = p.partition("=")
        if not sep:
            raise FixError(f"field without '=': {p!r}")
        try:
            tag = int(tag_str)
        except ValueError as exc:
            raise FixError(f"non-numeric tag: {tag_str!r}") from exc
        fields.append((tag, value))

    if fields[0][0] != Tag.BeginString:
        raise FixError("first field must be BeginString (8)")
    if fields[1][0] != Tag.BodyLength:
        raise FixError("second field must be BodyLength (9)")
    if fields[-1][0] != Tag.CheckSum:
        raise FixError("last field must be CheckSum (10)")

    if verify:
        idx = raw.rfind(f"{SOH}{Tag.CheckSum}=")
        if idx == -1:
            raise FixError("cannot locate CheckSum field")
        prefix = raw[: idx + 1]  # include the SOH before 10=
        expected = checksum(prefix)
        if fields[-1][1] != expected:
            raise FixError(
                f"checksum mismatch: wire={fields[-1][1]} computed={expected}"
            )

        head_len = len(
            f"{Tag.BeginString}={fields[0][1]}{SOH}"
            f"{Tag.BodyLength}={fields[1][1]}{SOH}"
        )
        body_len = len(prefix) - head_len
        try:
            declared = int(fields[1][1])
        except ValueError as exc:
            raise FixError("non-numeric BodyLength") from exc
        if declared != body_len:
            raise FixError(
                f"body length mismatch: wire={declared} computed={body_len}"
            )

    return FixMessage(fields)


def split_stream(buffer: str) -> tuple[list[str], str]:
    """Split a TCP read buffer into complete messages plus a remainder.

    FIX has no framing character, so the only reliable way to find a
    message boundary is to read BodyLength and count bytes. Scanning for
    "10=" would break on any body field whose value happens to contain
    that sequence.
    """
    messages: list[str] = []
    begin = f"{Tag.BeginString}={BEGIN_STRING}{SOH}"
    bl_marker = f"{Tag.BodyLength}="

    # `consumed` is the end of the last *complete* message; `pos` is the
    # scan cursor. They are separate on purpose: the remainder returned to
    # the caller must start at `consumed`, never at the scan cursor, or a
    # message that was only partially received loses its leading bytes on
    # the next read.
    consumed = 0
    pos = 0

    while True:
        start = buffer.find(begin, pos)
        if start == -1:
            break
        bl_start = start + len(begin)
        if not buffer.startswith(bl_marker, bl_start):
            if len(buffer) - bl_start < len(bl_marker):
                break  # truncated mid-header, wait for more bytes
            pos = start + 1  # a literal "8=FIX.4.4" inside a value
            continue
        bl_end = buffer.find(SOH, bl_start)
        if bl_end == -1:
            break
        try:
            body_len = int(buffer[bl_start + len(bl_marker) : bl_end])
        except ValueError:
            pos = start + 1
            continue
        if body_len < 0:
            pos = start + 1
            continue
        cs_start = bl_end + 1 + body_len
        if cs_start > len(buffer):
            break  # body not fully received
        cs_end = buffer.find(SOH, cs_start)
        if cs_end == -1:
            break  # checksum field not fully received
        end = cs_end + 1
        messages.append(buffer[start:end])
        consumed = end
        pos = end

    return messages, buffer[consumed:]


def utc_timestamp(dt: datetime | None = None) -> str:
    """FIX UTCTimestamp with milliseconds: YYYYMMDD-HH:MM:SS.sss"""
    dt = dt or datetime.now(timezone.utc)
    return dt.strftime("%Y%m%d-%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"
