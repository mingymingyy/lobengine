import random
from decimal import Decimal

import pytest

from lob_engine.types import BookUpdate, Level, PriceScale, Side, TopOfBook


class TestPriceScale:
    def test_round_trip(self):
        s = PriceScale(2)
        assert s.to_ticks("68123.45") == 6812345
        assert s.to_price(6812345) == Decimal("68123.45")

    def test_no_float_error(self):
        """The whole reason ticks exist: 0.1 + 0.2 must not bite."""
        s = PriceScale(1)
        assert s.to_ticks("0.1") + s.to_ticks("0.2") == s.to_ticks("0.3")

    def test_zero_decimals(self):
        s = PriceScale(0)
        assert s.to_ticks("42") == 42
        assert s.to_price(42) == Decimal("42")

    def test_trailing_zeros_ok(self):
        assert PriceScale(2).to_ticks("100.00") == 10000
        assert PriceScale(2).to_ticks("100") == 10000

    def test_rejects_excess_precision(self):
        """Silently rounding away venue precision corrupts the book."""
        with pytest.raises(ValueError, match="finer precision"):
            PriceScale(2).to_ticks("1.005")

    def test_negative_decimals_rejected(self):
        with pytest.raises(ValueError):
            PriceScale(-1)

    def test_accepts_float_and_decimal(self):
        s = PriceScale(2)
        assert s.to_ticks(1.5) == 150
        assert s.to_ticks(Decimal("1.5")) == 150
        assert s.to_ticks(3) == 300


class TestSide:
    def test_opposite_and_sign(self):
        assert Side.BUY.opposite is Side.SELL
        assert Side.SELL.opposite is Side.BUY
        assert Side.BUY.sign == 1
        assert Side.SELL.sign == -1


class TestTopOfBook:
    def test_mid_and_spread(self):
        t = TopOfBook(bid_px=100, bid_sz=5, ask_px=102, ask_sz=5)
        assert t.mid == 101.0
        assert t.spread == 2
        assert t.is_two_sided

    def test_one_sided(self):
        t = TopOfBook(bid_px=100, bid_sz=5, ask_px=None, ask_sz=0)
        assert not t.is_two_sided
        assert t.mid is None
        assert t.spread is None
        assert t.microprice is None

    def test_microprice_pulled_toward_the_thin_side(self):
        """Heavy bid, thin ask: the microprice sits above the mid.

        Weights are swapped by construction, so the microprice moves
        toward the price on the side with less size resting.
        """
        t = TopOfBook(bid_px=100, bid_sz=90, ask_px=102, ask_sz=10)
        assert t.mid == 101.0
        assert t.microprice > t.mid
        # (100*10 + 102*90) / 100 = 101.8
        assert t.microprice == pytest.approx(101.8)

    def test_microprice_symmetric_case(self):
        t = TopOfBook(bid_px=100, bid_sz=10, ask_px=102, ask_sz=90)
        assert t.microprice < t.mid

    def test_microprice_balanced_equals_mid(self):
        t = TopOfBook(bid_px=100, bid_sz=10, ask_px=102, ask_sz=10)
        assert t.microprice == t.mid

    def test_microprice_zero_size_falls_back_to_mid(self):
        t = TopOfBook(bid_px=100, bid_sz=0, ask_px=102, ask_sz=0)
        assert t.microprice == t.mid


def test_book_update_defaults():
    u = BookUpdate(symbol="X", bids=(Level(1, 2.0),))
    assert u.recv_ts_ns > 0
    assert u.asks == ()
    assert not u.is_snapshot


class TestPriceScaleErrors:
    def test_non_numeric_raises_value_error(self):
        """Callers should only have to catch ValueError."""
        with pytest.raises(ValueError, match="not a valid price"):
            PriceScale(2).to_ticks("abc")

    def test_nan_and_inf_rejected(self):
        for bad in ("NaN", "Infinity", float("inf"), float("nan")):
            with pytest.raises(ValueError):
                PriceScale(2).to_ticks(bad)


class TestPriceScaleFastPath:
    """The string fast path must be exactly equivalent to the Decimal path.

    It exists only for speed, so any disagreement is a correctness bug.
    This is a differential test: both paths are run on the same inputs and
    compared.
    """

    @staticmethod
    def _decimal_path(scale, text):
        d = Decimal(text).scaleb(scale.decimals)
        assert d == d.to_integral_value(), "test input is not representable"
        return int(d)

    @pytest.mark.parametrize("decimals", [0, 1, 2, 4, 8])
    def test_agrees_with_decimal_on_random_inputs(self, decimals):
        rng = random.Random(decimals)
        scale = PriceScale(decimals)
        for _ in range(2000):
            whole = rng.randint(0, 10 ** rng.randint(1, 7))
            n_frac = rng.randint(0, decimals)
            frac = "".join(str(rng.randint(0, 9)) for _ in range(n_frac))
            text = f"{whole}.{frac}" if frac else str(whole)
            for candidate in (text, "-" + text):
                assert scale.to_ticks(candidate) == self._decimal_path(
                    scale, candidate
                )

    @pytest.mark.parametrize(
        "text",
        ["0", "0.0", "00.00", "000123.4500", "1.", "999999999999.99"],
    )
    def test_edge_shapes(self, text):
        scale = PriceScale(2)
        expected = self._decimal_path(scale, text)
        assert scale.to_ticks(text) == expected

    def test_trailing_zeros_beyond_precision_are_accepted(self):
        assert PriceScale(2).to_ticks("1.5000") == 150

    def test_significant_digits_beyond_precision_still_rejected(self):
        """The fast path must not become a silent rounding shortcut."""
        with pytest.raises(ValueError, match="finer precision"):
            PriceScale(2).to_ticks("1.005")

    @pytest.mark.parametrize(
        "text", [".5", "1e5", "1_000", " 1.5", "1.5 ", "+1.5", "nan", ""]
    )
    def test_unusual_forms_fall_through_without_wrong_answers(self, text):
        """Either the Decimal path handles it, or it raises. Never a
        silently wrong number."""
        scale = PriceScale(2)
        try:
            got = scale.to_ticks(text)
        except ValueError:
            return
        assert got == self._decimal_path(scale, text)
