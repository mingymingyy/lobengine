"""FIX codec tests.

The framing fields are checked against a *separately written* reference
implementation below, not against the encoder itself. A round-trip test
alone would pass happily with a wrong checksum algorithm, because the
same wrong algorithm would be used to verify it.
"""

import pytest

from lob_engine import fix
from lob_engine.fix import SOH, FixError, MsgType, OrdType, Tag, TimeInForce


# --------------------------------------------------------------------- #
# independent reference
# --------------------------------------------------------------------- #
def reference_frame(msg_type: str, body_fields: list[tuple[int, str]]) -> str:
    """Build a FIX message from the spec definitions, written out longhand.

    BodyLength: number of characters after the SOH ending tag 9, up to and
    including the SOH ending the last field before tag 10.
    CheckSum: sum of all character codes up to and including that SOH,
    modulo 256, three digits.
    """
    parts = [f"35={msg_type}"] + [f"{t}={v}" for t, v in body_fields]
    body = ""
    for part in parts:
        body = body + part + chr(1)

    body_length = len(body)
    prefix = "8=FIX.4.4" + chr(1) + "9=" + str(body_length) + chr(1) + body

    total = 0
    for ch in prefix:
        total = total + ord(ch)
    cs = total % 256

    return prefix + "10=" + str(cs).rjust(3, "0") + chr(1)


BODY = [
    (Tag.SenderCompID, "LOBENG"),
    (Tag.TargetCompID, "VENUE"),
    (Tag.MsgSeqNum, "12"),
    (Tag.SendingTime, "20260913-09:30:00.123"),
    (Tag.ClOrdID, "ORD-00000001"),
    (Tag.Symbol, "D05"),
    (Tag.Side, "1"),
    (Tag.OrderQty, "100"),
    (Tag.OrdType, "2"),
    (Tag.Price, "34.52"),
    (Tag.TimeInForce, "0"),
    (Tag.TransactTime, "20260913-09:30:00.123"),
]


class TestFraming:
    def test_matches_independent_reference(self):
        assert fix.encode(MsgType.NEW_ORDER_SINGLE, BODY) == reference_frame("D", BODY)

    @pytest.mark.parametrize(
        "mtype,body",
        [
            ("0", []),
            ("A", [(Tag.EncryptMethod, "0"), (Tag.HeartBtInt, "30")]),
            ("8", [(Tag.ClOrdID, "X"), (Tag.ExecType, "F"), (Tag.LastPx, "1.0")]),
            ("D", BODY),
        ],
    )
    def test_reference_agreement_across_message_types(self, mtype, body):
        assert fix.encode(mtype, body) == reference_frame(mtype, body)

    def test_body_length_counted_by_hand(self):
        raw = fix.encode("0", [(Tag.SenderCompID, "A"), (Tag.TargetCompID, "B")])
        # body is "35=0|49=A|56=B|" = 5 + 5 + 5 = 15
        assert f"{SOH}9=15{SOH}" in raw

    def test_checksum_is_three_digits(self):
        for i in range(200):
            raw = fix.encode("0", [(Tag.Text, "x" * i)])
            cs = raw.rsplit("10=", 1)[1][:-1]
            assert len(cs) == 3 and cs.isdigit()

    def test_checksum_wraps_at_256(self):
        assert fix.checksum("") == "000"
        assert fix.checksum(chr(1) * 256) == "000"
        assert fix.checksum("A") == "065"

    def test_field_order_is_preserved(self):
        raw = fix.encode("D", BODY)
        assert raw.startswith(f"8=FIX.4.4{SOH}9=")
        assert f"{SOH}35=D{SOH}49=LOBENG{SOH}" in raw
        assert raw.endswith(SOH)


class TestEnumSerialisation:
    def test_msg_type_enum_writes_its_value(self):
        """`35=MsgType.NEW_ORDER_SINGLE` on the wire drops the session."""
        raw = fix.encode(MsgType.NEW_ORDER_SINGLE, [])
        assert f"{SOH}35=D{SOH}" in raw
        assert "MsgType" not in raw

    def test_field_value_enums_write_their_value(self):
        raw = fix.encode(
            MsgType.NEW_ORDER_SINGLE,
            [(Tag.OrdType, OrdType.LIMIT), (Tag.TimeInForce, TimeInForce.IOC)],
        )
        assert f"{SOH}40=2{SOH}" in raw
        assert f"{SOH}59=3{SOH}" in raw

    def test_numbers_are_stringified(self):
        raw = fix.encode("D", [(Tag.OrderQty, 100), (Tag.Price, 34.5)])
        assert f"{SOH}38=100{SOH}" in raw
        assert f"{SOH}44=34.5{SOH}" in raw


class TestDecode:
    def test_round_trip(self):
        raw = fix.encode(MsgType.NEW_ORDER_SINGLE, BODY)
        msg = fix.decode(raw)
        assert msg.msg_type == "D"
        assert msg.get(Tag.ClOrdID) == "ORD-00000001"
        assert msg.get_int(Tag.MsgSeqNum) == 12
        assert msg.get_float(Tag.Price) == 34.52
        assert msg.seq_num == 12

    def test_missing_field_returns_default(self):
        msg = fix.decode(fix.encode("0", []))
        assert msg.get(Tag.Price) is None
        assert msg.get(Tag.Price, "x") == "x"
        assert msg.get_int(Tag.Price, 7) == 7

    def test_require_raises(self):
        msg = fix.decode(fix.encode("0", []))
        with pytest.raises(FixError, match="missing required tag"):
            msg.require(Tag.Price)

    def test_value_may_contain_equals(self):
        raw = fix.encode("0", [(Tag.Text, "a=b=c")])
        assert fix.decode(raw).get(Tag.Text) == "a=b=c"

    def test_empty_value_allowed(self):
        raw = fix.encode("0", [(Tag.Text, "")])
        assert fix.decode(raw).get(Tag.Text) == ""


class TestDecodeRejectsCorruption:
    def test_bad_checksum(self):
        raw = fix.encode("D", BODY)
        bad = raw[:-4] + "999" + SOH
        with pytest.raises(FixError, match="checksum mismatch"):
            fix.decode(bad)

    def test_flipped_byte_in_the_body(self):
        raw = fix.encode("D", BODY)
        bad = raw.replace("D05", "D06")
        with pytest.raises(FixError, match="checksum mismatch"):
            fix.decode(bad)

    def test_bad_body_length(self):
        raw = fix.encode("D", BODY)
        n = int(raw.split(f"{SOH}9=")[1].split(SOH)[0])
        tampered = raw.replace(f"{SOH}9={n}{SOH}", f"{SOH}9={n + 1}{SOH}", 1)
        with pytest.raises(FixError):
            fix.decode(tampered)

    def test_missing_trailing_soh(self):
        with pytest.raises(FixError, match="end with SOH"):
            fix.decode(fix.encode("0", [])[:-1])

    def test_field_without_equals(self):
        with pytest.raises(FixError, match="without"):
            fix.decode(f"8=FIX.4.4{SOH}9=5{SOH}garbage{SOH}10=000{SOH}")

    def test_non_numeric_tag(self):
        with pytest.raises(FixError, match="non-numeric tag"):
            fix.decode(f"8=FIX.4.4{SOH}9=5{SOH}xx=1{SOH}10=000{SOH}")

    def test_wrong_first_field(self):
        with pytest.raises(FixError, match="BeginString"):
            fix.decode(f"35=D{SOH}8=FIX.4.4{SOH}10=000{SOH}")

    def test_verify_can_be_disabled(self):
        raw = fix.encode("D", BODY)
        bad = raw[:-4] + "999" + SOH
        assert fix.decode(bad, verify=False).msg_type == "D"


class TestSplitStream:
    def test_splits_two_messages(self):
        a = fix.encode("0", [(Tag.MsgSeqNum, "1")])
        b = fix.encode("0", [(Tag.MsgSeqNum, "2")])
        msgs, rest = fix.split_stream(a + b)
        assert msgs == [a, b]
        assert rest == ""

    def test_holds_back_a_partial_message(self):
        a = fix.encode("0", [(Tag.MsgSeqNum, "1")])
        b = fix.encode("0", [(Tag.MsgSeqNum, "2")])
        msgs, rest = fix.split_stream(a + b[:10])
        assert msgs == [a]
        assert rest == b[:10]
        # the rest completes on the next read
        msgs2, rest2 = fix.split_stream(rest + b[10:])
        assert msgs2 == [b] and rest2 == ""

    def test_body_containing_the_checksum_marker(self):
        """A value containing '10=' must not be mistaken for the trailer.

        This is why the splitter counts BodyLength bytes rather than
        searching for the checksum tag.
        """
        tricky = fix.encode("0", [(Tag.Text, "note 10=999 is not the trailer")])
        msgs, rest = fix.split_stream(tricky)
        assert msgs == [tricky]
        assert rest == ""
        assert fix.decode(msgs[0]).get(Tag.Text) == "note 10=999 is not the trailer"

    def test_body_containing_a_begin_string(self):
        """A literal BeginString inside a value must not start a message."""
        tricky = fix.encode("0", [(Tag.Text, f"8=FIX.4.4{'x'}")])
        msgs, rest = fix.split_stream(tricky)
        assert msgs == [tricky]
        assert rest == ""

    def test_leading_garbage_is_skipped(self):
        a = fix.encode("0", [(Tag.MsgSeqNum, "1")])
        msgs, _ = fix.split_stream("junk" + a)
        assert msgs == [a]

    def test_every_split_point(self):
        """Chunk the stream at each byte; the result must not change."""
        a = fix.encode("0", [(Tag.MsgSeqNum, "1")])
        b = fix.encode("D", BODY)
        stream = a + b
        for cut in range(1, len(stream)):
            first, rest = fix.split_stream(stream[:cut])
            more, leftover = fix.split_stream(rest + stream[cut:])
            assert first + more == [a, b]
            assert leftover == ""


def test_utc_timestamp_format():
    from datetime import datetime, timezone

    ts = fix.utc_timestamp(datetime(2026, 9, 13, 9, 30, 0, 123456, timezone.utc))
    assert ts == "20260913-09:30:00.123"
