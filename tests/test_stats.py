import numpy as np
import pytest

from lob_engine.stats import ols, percentiles


class TestOLS:
    def test_exact_fit(self):
        x = np.arange(50.0)
        y = 3.0 + 2.0 * x
        r = ols(y, x)
        assert r.intercept == pytest.approx(3.0)
        assert r.slope == pytest.approx(2.0)
        assert r.r_squared == pytest.approx(1.0)

    def test_recovers_known_coefficients_with_noise(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=5000)
        y = 1.5 + 0.8 * x + rng.normal(scale=0.5, size=5000)
        r = ols(y, x)
        assert r.slope == pytest.approx(0.8, abs=0.02)
        assert r.intercept == pytest.approx(1.5, abs=0.02)
        assert 0.5 < r.r_squared < 0.8

    def test_matches_the_normal_equations(self):
        """Independent check: beta = (X'X)^-1 X'y computed separately."""
        rng = np.random.default_rng(3)
        x = rng.normal(size=200)
        y = rng.normal(size=200)
        X = np.column_stack([np.ones(200), x])
        expected = np.linalg.solve(X.T @ X, X.T @ y)
        assert ols(y, x).params == pytest.approx(expected)

    def test_multivariate(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(2000, 3))
        y = 0.5 + X @ np.array([1.0, -2.0, 0.5]) + rng.normal(scale=0.1, size=2000)
        r = ols(y, X)
        assert r.n_params == 4
        assert r.params[1:] == pytest.approx([1.0, -2.0, 0.5], abs=0.02)

    def test_no_constant(self):
        x = np.arange(1.0, 21.0)
        r = ols(2.0 * x, x, add_constant=False)
        assert r.n_params == 1
        assert r.params[0] == pytest.approx(2.0)

    def test_zero_slope_has_a_small_t_stat(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=1000)
        y = rng.normal(size=1000)
        assert abs(ols(y, x).slope_t) < 3.0

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="same number of rows"):
            ols(np.arange(10.0), np.arange(9.0))

    def test_too_few_observations(self):
        with pytest.raises(ValueError, match="not enough observations"):
            ols(np.array([1.0]), np.array([1.0]))

    def test_adjusted_r_squared_penalises_regressors(self):
        rng = np.random.default_rng(2)
        y = rng.normal(size=60)
        X = rng.normal(size=(60, 20))
        r = ols(y, X)
        assert r.adj_r_squared < r.r_squared


class TestHAC:
    def test_plain_errors_match_the_textbook_formula(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=500)
        y = 1.0 + 2.0 * x + rng.normal(scale=0.3, size=500)
        r = ols(y, x, hac_lags=0)
        X = np.column_stack([np.ones(500), x])
        resid = y - X @ r.params
        s2 = resid @ resid / (500 - 2)
        expected = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
        # hac_lags=0 still applies White's heteroskedasticity correction,
        # so the two agree closely under homoskedasticity rather than
        # exactly.
        assert r.stderr == pytest.approx(expected, rel=0.15)

    def test_serial_correlation_inflates_standard_errors(self):
        """The reason HAC exists: naive errors overstate significance when
        residuals are autocorrelated."""
        rng = np.random.default_rng(6)
        n = 2000
        x = np.cumsum(rng.normal(size=n))  # persistent regressor
        e = np.zeros(n)
        for i in range(1, n):
            e[i] = 0.9 * e[i - 1] + rng.normal()
        y = 0.0 * x + e
        naive = ols(y, x, hac_lags=0)
        robust = ols(y, x)
        assert robust.hac_lags > 0
        assert robust.stderr[1] > naive.stderr[1]

    def test_default_bandwidth_rule(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=1000)
        # floor(4 * (1000/100)^(2/9)) = floor(4 * 1.6681...) = 6
        assert ols(x, x, hac_lags=None).hac_lags == 6

    def test_lags_are_clamped(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=10)
        assert ols(x, x, hac_lags=999).hac_lags == 9
        assert ols(x, x, hac_lags=-5).hac_lags == 0

    def test_standard_errors_are_positive(self):
        rng = np.random.default_rng(10)
        x = rng.normal(size=300)
        y = rng.normal(size=300)
        assert (ols(y, x).stderr > 0).all()


class TestSummary:
    def test_renders(self):
        x = np.arange(30.0)
        text = ols(3.0 + 2.0 * x + np.sin(x), x).summary(["const", "OFI"])
        assert "OFI" in text and "R^2" in text


class TestPercentiles:
    def test_basic(self):
        p = percentiles(range(101), qs=(50,))
        assert p[50] == pytest.approx(50.0)

    def test_empty_gives_nan(self):
        p = percentiles([], qs=(50, 99))
        assert all(np.isnan(v) for v in p.values())

    def test_multiple_quantiles(self):
        p = percentiles(range(1001))
        assert p[50] < p[90] < p[99] < p[99.9]
