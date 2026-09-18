"""FIX session layer.

The session layer is the part that keeps the connection alive and keeps
both sides agreeing on what has been sent. It is deliberately separate
from the application layer (orders) because the failure modes are
different: session problems disconnect you, application problems reject
an order.

What is implemented here:

* Logon / Logout with a negotiated heartbeat interval.
* Outbound heartbeats when the link has been quiet.
* TestRequest when the counterparty has been quiet for too long, and
  disconnect if it stays quiet.
* Strictly increasing inbound sequence numbers, with a ResendRequest on a
  gap and a Logout on a sequence number that is too low.
* SequenceReset-GapFill handling.
* PossDupFlag handling, so a replayed message is not treated as a gap.

This is a clean-room implementation for a simulator. A production desk
would use QuickFIX; the point of writing it out is that the sequence
logic is where FIX integrations actually break, and it is worth
understanding rather than importing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto

from . import fix
from .fix import SOH, FixMessage, MsgType, Tag


class SessionState(Enum):
    DISCONNECTED = auto()
    LOGON_SENT = auto()
    ACTIVE = auto()
    LOGOUT_SENT = auto()


@dataclass(slots=True)
class SessionConfig:
    sender_comp_id: str
    target_comp_id: str
    heartbeat_interval: float = 30.0
    # A counterparty is considered dead after this multiple of the
    # heartbeat interval with no inbound traffic. FIX recommends 1.2x for
    # the TestRequest and disconnecting shortly after.
    test_request_multiplier: float = 1.2
    disconnect_multiplier: float = 2.4
    reset_seq_on_logon: bool = True


@dataclass(slots=True)
class SessionStats:
    sent: int = 0
    received: int = 0
    heartbeats_sent: int = 0
    test_requests_sent: int = 0
    resend_requests_sent: int = 0
    gaps_detected: int = 0
    rejects_sent: int = 0


class FixSession:
    """Pure state machine. It never touches a socket.

    Call `poll()` on a timer and `on_message()` for each inbound message.
    Both return a list of raw FIX strings for the caller to write out.
    That keeps the session testable without any I/O.
    """

    def __init__(self, config: SessionConfig, clock=time.time) -> None:
        self.cfg = config
        self._clock = clock
        self.state = SessionState.DISCONNECTED
        self.out_seq = 1
        self.in_seq = 1
        self.last_sent = 0.0
        self.last_received = 0.0
        self.stats = SessionStats()
        self.sent_messages: dict[int, str] = {}  # for resend
        self._pending_test_req: str | None = None

    # ------------------------------------------------------------------ #
    # outbound
    # ------------------------------------------------------------------ #
    def _header(self) -> list[tuple[int, str]]:
        return [
            (Tag.SenderCompID, self.cfg.sender_comp_id),
            (Tag.TargetCompID, self.cfg.target_comp_id),
            (Tag.MsgSeqNum, str(self.out_seq)),
            (Tag.SendingTime, fix.utc_timestamp()),
        ]

    def send(self, msg_type: str, body: list[tuple[int, str]] | None = None) -> str:
        """Build and record an outbound message, advancing the sequence."""
        raw = fix.encode(msg_type, self._header() + list(body or []))
        self.sent_messages[self.out_seq] = raw
        self.out_seq += 1
        self.last_sent = self._clock()
        self.stats.sent += 1
        return raw

    def logon(self) -> str:
        if self.cfg.reset_seq_on_logon:
            self.out_seq = 1
            self.in_seq = 1
            self.sent_messages.clear()
        raw = self.send(
            MsgType.LOGON,
            [
                (Tag.EncryptMethod, "0"),
                (Tag.HeartBtInt, str(int(self.cfg.heartbeat_interval))),
            ],
        )
        self.state = SessionState.LOGON_SENT
        self.last_received = self._clock()
        return raw

    def logout(self, text: str = "") -> str:
        body = [(Tag.Text, text)] if text else []
        raw = self.send(MsgType.LOGOUT, body)
        self.state = SessionState.LOGOUT_SENT
        return raw

    # ------------------------------------------------------------------ #
    # timers
    # ------------------------------------------------------------------ #
    def poll(self) -> list[str]:
        """Emit heartbeats / test requests. Call this regularly."""
        out: list[str] = []
        if self.state not in (SessionState.ACTIVE, SessionState.LOGON_SENT):
            return out
        now = self._clock()
        hb = self.cfg.heartbeat_interval

        if now - self.last_sent >= hb:
            out.append(self.send(MsgType.HEARTBEAT))
            self.stats.heartbeats_sent += 1

        quiet = now - self.last_received
        if quiet >= hb * self.cfg.disconnect_multiplier:
            out.append(self.logout("no inbound traffic"))
            self.state = SessionState.DISCONNECTED
        elif quiet >= hb * self.cfg.test_request_multiplier and not self._pending_test_req:
            req_id = f"TR{self.out_seq}"
            self._pending_test_req = req_id
            out.append(self.send(MsgType.TEST_REQUEST, [(Tag.TestReqID, req_id)]))
            self.stats.test_requests_sent += 1
        return out

    # ------------------------------------------------------------------ #
    # inbound
    # ------------------------------------------------------------------ #
    def on_message(self, msg: FixMessage) -> tuple[list[str], bool]:
        """Process one inbound message.

        Returns (messages_to_send, deliver_to_application).
        `deliver_to_application` is False for pure session traffic and for
        anything rejected at the session level.
        """
        self.last_received = self._clock()
        self.stats.received += 1
        mtype = msg.msg_type
        seq = msg.get_int(Tag.MsgSeqNum)
        poss_dup = msg.get(Tag.PossDupFlag) == "Y"

        # SequenceReset-Reset is processed regardless of sequence number.
        if mtype == MsgType.SEQUENCE_RESET:
            return self._on_sequence_reset(msg)

        if seq is None:
            return [self._reject(None, "missing MsgSeqNum")], False

        if seq > self.in_seq:
            if poss_dup:
                return [], False
            self.stats.gaps_detected += 1
            self.stats.resend_requests_sent += 1
            return [
                self.send(
                    MsgType.RESEND_REQUEST,
                    [(Tag.BeginSeqNo, str(self.in_seq)), (Tag.EndSeqNo, "0")],
                )
            ], False

        if seq < self.in_seq:
            if poss_dup:
                return [], False  # already processed
            # Too low and not a duplicate is unrecoverable per the spec.
            return [self.logout(f"MsgSeqNum too low, expecting {self.in_seq}")], False

        self.in_seq += 1

        if mtype == MsgType.LOGON:
            self.state = SessionState.ACTIVE
            hb = msg.get_int(Tag.HeartBtInt)
            if hb:
                self.cfg.heartbeat_interval = float(hb)
            return [], False
        if mtype == MsgType.HEARTBEAT:
            if self._pending_test_req and msg.get(Tag.TestReqID) == self._pending_test_req:
                self._pending_test_req = None
            return [], False
        if mtype == MsgType.TEST_REQUEST:
            req = msg.get(Tag.TestReqID, "")
            return [self.send(MsgType.HEARTBEAT, [(Tag.TestReqID, req)])], False
        if mtype == MsgType.RESEND_REQUEST:
            return self._on_resend_request(msg), False
        if mtype == MsgType.LOGOUT:
            out = []
            if self.state != SessionState.LOGOUT_SENT:
                out.append(self.send(MsgType.LOGOUT, [(Tag.Text, "logout ack")]))
            self.state = SessionState.DISCONNECTED
            return out, False

        if self.state == SessionState.LOGON_SENT:
            self.state = SessionState.ACTIVE
        return [], True

    def _on_sequence_reset(self, msg: FixMessage) -> tuple[list[str], bool]:
        new_seq = msg.get_int(Tag.NewSeqNo)
        if new_seq is None:
            return [self._reject(msg.get_int(Tag.MsgSeqNum), "missing NewSeqNo")], False
        if new_seq < self.in_seq:
            return [
                self._reject(msg.get_int(Tag.MsgSeqNum), "NewSeqNo below expected")
            ], False
        self.in_seq = new_seq
        return [], False

    def _on_resend_request(self, msg: FixMessage) -> list[str]:
        begin = msg.get_int(Tag.BeginSeqNo) or 1
        end = msg.get_int(Tag.EndSeqNo) or 0
        last = self.out_seq - 1
        if end == 0 or end > last:
            end = last

        out: list[str] = []
        gap_from: int | None = None

        def flush_gap(upto: int) -> None:
            nonlocal gap_from
            if gap_from is None:
                return
            out.append(
                self._admin_gap_fill(gap_from, upto)
            )
            gap_from = None

        for s in range(begin, end + 1):
            raw = self.sent_messages.get(s)
            if raw is None or _is_admin(raw):
                # Administrative messages are never resent; they are
                # replaced by a SequenceReset-GapFill.
                if gap_from is None:
                    gap_from = s
                continue
            flush_gap(s)
            out.append(_mark_poss_dup(raw))
        flush_gap(end + 1)
        return out

    def _admin_gap_fill(self, from_seq: int, new_seq: int) -> str:
        raw = fix.encode(
            MsgType.SEQUENCE_RESET,
            [
                (Tag.SenderCompID, self.cfg.sender_comp_id),
                (Tag.TargetCompID, self.cfg.target_comp_id),
                (Tag.MsgSeqNum, str(from_seq)),
                (Tag.SendingTime, fix.utc_timestamp()),
                (Tag.PossDupFlag, "Y"),
                (Tag.GapFillFlag, "Y"),
                (Tag.NewSeqNo, str(new_seq)),
            ],
        )
        self.last_sent = self._clock()
        return raw

    def _reject(self, ref_seq: int | None, text: str) -> str:
        self.stats.rejects_sent += 1
        body: list[tuple[int, str]] = []
        if ref_seq is not None:
            body.append((Tag.RefSeqNum, str(ref_seq)))
        body.append((Tag.Text, text))
        return self.send(MsgType.REJECT, body)


_ADMIN_TYPES = {
    MsgType.HEARTBEAT,
    MsgType.TEST_REQUEST,
    MsgType.RESEND_REQUEST,
    MsgType.SEQUENCE_RESET,
    MsgType.LOGON,
    MsgType.LOGOUT,
    MsgType.REJECT,
}


def _is_admin(raw: str) -> bool:
    marker = f"{SOH}{Tag.MsgType}="
    i = raw.find(marker)
    if i == -1:
        return False
    j = raw.find(SOH, i + len(marker))
    return raw[i + len(marker) : j] in {t.value for t in _ADMIN_TYPES}


def _mark_poss_dup(raw: str) -> str:
    """Re-frame a stored message with PossDupFlag=Y and OrigSendingTime.

    BodyLength and CheckSum both change, so the message has to be rebuilt
    rather than string-patched.
    """
    msg = fix.decode(raw, verify=False)
    orig_sending = msg.get(Tag.SendingTime, "")
    body: list[tuple[int, str]] = []
    for tag, value in msg.fields:
        if tag in (Tag.BeginString, Tag.BodyLength, Tag.CheckSum, Tag.MsgType):
            continue
        if tag in (Tag.PossDupFlag, Tag.OrigSendingTime):
            continue
        body.append((tag, value))
        if tag == Tag.SendingTime:
            body.append((Tag.PossDupFlag, "Y"))
            body.append((Tag.OrigSendingTime, orig_sending))
    return fix.encode(msg.msg_type, body)
