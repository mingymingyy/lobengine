import pytest

from lob_engine import fix
from lob_engine.fix import Tag
from lob_engine.session import FixSession, SessionConfig, SessionState


class Clock:
    """Controllable clock so timer behaviour is deterministic."""

    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def session(clock):
    return FixSession(
        SessionConfig(sender_comp_id="US", target_comp_id="THEM",
                      heartbeat_interval=30.0),
        clock=clock,
    )


def inbound(msg_type, seq, extra=None):
    body = [
        (Tag.SenderCompID, "THEM"),
        (Tag.TargetCompID, "US"),
        (Tag.MsgSeqNum, str(seq)),
        (Tag.SendingTime, "20260913-09:30:00.000"),
    ] + list(extra or [])
    return fix.decode(fix.encode(msg_type, body))


class TestLogon:
    def test_logon_message(self, session):
        raw = session.logon()
        msg = fix.decode(raw)
        assert msg.msg_type == "A"
        assert msg.get(Tag.HeartBtInt) == "30"
        assert msg.seq_num == 1
        assert session.state is SessionState.LOGON_SENT

    def test_logon_ack_activates(self, session):
        session.logon()
        session.on_message(inbound("A", 1, [(Tag.HeartBtInt, "30")]))
        assert session.state is SessionState.ACTIVE
        assert session.in_seq == 2

    def test_counterparty_heartbeat_interval_is_adopted(self, session):
        session.logon()
        session.on_message(inbound("A", 1, [(Tag.HeartBtInt, "5")]))
        assert session.cfg.heartbeat_interval == 5.0


class TestSequencing:
    def test_in_order_delivers_to_application(self, session):
        session.logon()
        session.on_message(inbound("A", 1))
        out, deliver = session.on_message(inbound("8", 2, [(Tag.ClOrdID, "X")]))
        assert deliver and out == []
        assert session.in_seq == 3

    def test_gap_triggers_resend_request(self, session):
        session.logon()
        session.on_message(inbound("A", 1))
        out, deliver = session.on_message(inbound("8", 7))
        assert not deliver
        msg = fix.decode(out[0])
        assert msg.msg_type == "2"
        assert msg.get(Tag.BeginSeqNo) == "2"
        assert msg.get(Tag.EndSeqNo) == "0"
        assert session.in_seq == 2  # unchanged; we are still waiting
        assert session.stats.gaps_detected == 1

    def test_too_low_without_poss_dup_logs_out(self, session):
        session.logon()
        session.on_message(inbound("A", 1))
        session.on_message(inbound("8", 2))
        out, deliver = session.on_message(inbound("8", 2))
        assert not deliver
        assert fix.decode(out[0]).msg_type == "5"

    def test_poss_dup_replay_is_ignored_quietly(self, session):
        session.logon()
        session.on_message(inbound("A", 1))
        session.on_message(inbound("8", 2))
        out, deliver = session.on_message(
            inbound("8", 2, [(Tag.PossDupFlag, "Y")])
        )
        assert out == [] and not deliver

    def test_missing_seq_num_is_rejected(self, session):
        session.logon()
        raw = fix.encode(
            "8",
            [(Tag.SenderCompID, "THEM"), (Tag.TargetCompID, "US"),
             (Tag.SendingTime, "20260913-09:30:00.000")],
        )
        out, deliver = session.on_message(fix.decode(raw))
        assert not deliver
        assert fix.decode(out[0]).msg_type == "3"


class TestSequenceReset:
    def test_gap_fill_advances_expected_sequence(self, session):
        session.logon()
        session.on_message(inbound("A", 1))
        session.on_message(inbound("4", 99, [(Tag.GapFillFlag, "Y"),
                                             (Tag.NewSeqNo, "10")]))
        assert session.in_seq == 10

    def test_reset_below_expected_is_rejected(self, session):
        session.logon()
        session.on_message(inbound("A", 1))
        session.on_message(inbound("8", 2))
        out, _ = session.on_message(inbound("4", 3, [(Tag.NewSeqNo, "1")]))
        assert fix.decode(out[0]).msg_type == "3"
        assert session.in_seq == 3


class TestTimers:
    def test_heartbeat_after_quiet_period(self, session, clock):
        session.logon()
        session.on_message(inbound("A", 1))
        assert session.poll() == []
        clock.advance(31)
        out = session.poll()
        assert any(fix.decode(m).msg_type == "0" for m in out)
        assert session.stats.heartbeats_sent == 1

    def test_test_request_when_counterparty_is_quiet(self, session, clock):
        session.logon()
        session.on_message(inbound("A", 1))
        clock.advance(37)  # past 1.2 * 30
        out = session.poll()
        assert any(fix.decode(m).msg_type == "1" for m in out)

    def test_disconnect_when_counterparty_stays_quiet(self, session, clock):
        session.logon()
        session.on_message(inbound("A", 1))
        clock.advance(80)  # past 2.4 * 30
        out = session.poll()
        assert any(fix.decode(m).msg_type == "5" for m in out)
        assert session.state is SessionState.DISCONNECTED

    def test_test_request_is_answered_with_matching_id(self, session):
        session.logon()
        session.on_message(inbound("A", 1))
        out, _ = session.on_message(inbound("1", 2, [(Tag.TestReqID, "ABC")]))
        msg = fix.decode(out[0])
        assert msg.msg_type == "0"
        assert msg.get(Tag.TestReqID) == "ABC"


class TestResend:
    def test_application_messages_are_resent_with_poss_dup(self, session):
        session.logon()
        session.on_message(inbound("A", 1))
        session.send("D", [(Tag.ClOrdID, "ONE")])
        session.send("D", [(Tag.ClOrdID, "TWO")])

        out, _ = session.on_message(
            inbound("2", 2, [(Tag.BeginSeqNo, "2"), (Tag.EndSeqNo, "0")])
        )
        resent = [fix.decode(m) for m in out]
        app = [m for m in resent if m.msg_type == "D"]
        assert [m.get(Tag.ClOrdID) for m in app] == ["ONE", "TWO"]
        assert all(m.get(Tag.PossDupFlag) == "Y" for m in app)
        assert all(m.get(Tag.OrigSendingTime) for m in app)

    def test_resent_messages_are_still_valid_fix(self, session):
        """Re-framing must recompute BodyLength and CheckSum."""
        session.logon()
        session.on_message(inbound("A", 1))
        session.send("D", [(Tag.ClOrdID, "ONE"), (Tag.Symbol, "D05")])
        out, _ = session.on_message(
            inbound("2", 2, [(Tag.BeginSeqNo, "1"), (Tag.EndSeqNo, "0")])
        )
        for raw in out:
            fix.decode(raw)  # raises on bad framing

    def test_admin_messages_become_a_gap_fill(self, session, clock):
        session.logon()
        session.on_message(inbound("A", 1))
        clock.advance(31)
        session.poll()  # sends a heartbeat at seq 2
        session.send("D", [(Tag.ClOrdID, "ONE")])  # seq 3

        out, _ = session.on_message(
            inbound("2", 2, [(Tag.BeginSeqNo, "1"), (Tag.EndSeqNo, "0")])
        )
        types = [fix.decode(m).msg_type for m in out]
        assert "4" in types  # SequenceReset-GapFill covers the admin range
        assert "D" in types
        gap = next(fix.decode(m) for m in out if fix.decode(m).msg_type == "4")
        assert gap.get(Tag.GapFillFlag) == "Y"


class TestLogout:
    def test_logout_is_acknowledged(self, session):
        session.logon()
        session.on_message(inbound("A", 1))
        out, deliver = session.on_message(inbound("5", 2))
        assert not deliver
        assert fix.decode(out[0]).msg_type == "5"
        assert session.state is SessionState.DISCONNECTED

    def test_our_own_logout_is_not_double_acked(self, session):
        session.logon()
        session.on_message(inbound("A", 1))
        session.logout("done")
        out, _ = session.on_message(inbound("5", 2))
        assert out == []
        assert session.state is SessionState.DISCONNECTED
