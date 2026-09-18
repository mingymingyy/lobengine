"""Small OLS helper with heteroskedasticity- and autocorrelation-consistent
(Newey-West) standard errors.

Written out rather than pulled from statsmodels for two reasons: it keeps
the dependency list to numpy, and the price-impact regression is one of
the places where using plain OLS standard errors would overstate
significance. Bucketed order-flow data is both heteroskedastic (impact
scales with depth) and serially correlated at short horizons.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RegressionResult:
    """Result of a univariate or multivariate OLS fit."""

    params: np.ndarray  # [intercept, slope, ...]
    stderr: np.ndarray
    tstat: np.ndarray
    r_squared: float
    adj_r_squared: float
    n_obs: int
    n_params: int
    resid_std: float
    hac_lags: int

    @property
    def intercept(self) -> float:
        return float(self.params[0])

    @property
    def slope(self) -> float:
        """First non-intercept coefficient."""
        return float(self.params[1])

    @property
    def slope_t(self) -> float:
        return float(self.tstat[1])

    def summary(self, names: list[str] | None = None) -> str:
        names = names or ["const"] + [f"x{i}" for i in range(1, self.n_params)]
        lines = [
            f"n = {self.n_obs}   R^2 = {self.r_squared:.4f}   "
            f"adj R^2 = {self.adj_r_squared:.4f}   "
            f"HAC lags = {self.hac_lags}",
            f"{'term':<14}{'coef':>14}{'std err':>12}{'t':>10}",
            "-" * 50,
        ]
        for i, nm in enumerate(names):
            lines.append(
                f"{nm:<14}{self.params[i]:>14.6g}"
                f"{self.stderr[i]:>12.4g}{self.tstat[i]:>10.2f}"
            )
        return "\n".join(lines)


def _newey_west_lags(n: int) -> int:
    """Newey & West's rule-of-thumb bandwidth: floor(4*(n/100)^(2/9))."""
    if n <= 1:
        return 0
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def ols(
    y: np.ndarray,
    X: np.ndarray,
    add_constant: bool = True,
    hac_lags: int | None = None,
) -> RegressionResult:
    """Fit y = X b + e.

    Parameters
    ----------
    y : (n,) array
    X : (n,) or (n, k) array of regressors, excluding the constant
    add_constant : prepend a column of ones
    hac_lags : Newey-West bandwidth. None uses the rule of thumb; 0 gives
        plain (non-robust) OLS standard errors.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] != y.shape[0]:
        raise ValueError("y and X must have the same number of rows")
    if add_constant:
        X = np.column_stack([np.ones(len(y)), X])

    n, k = X.shape
    if n <= k:
        raise ValueError(f"not enough observations: n={n}, k={k}")

    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    ss_res = float(resid @ resid)
    y_dm = y - y.mean() if add_constant else y
    ss_tot = float(y_dm @ y_dm)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    adj_r2 = (
        1.0 - (1.0 - r2) * (n - 1) / (n - k) if np.isfinite(r2) and n > k else float("nan")
    )

    lags = _newey_west_lags(n) if hac_lags is None else int(hac_lags)
    lags = max(0, min(lags, n - 1))

    # Meat of the sandwich: S = sum_j w_j * (Gamma_j + Gamma_j')
    u = X * resid[:, None]
    S = u.T @ u
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)  # Bartlett kernel
        G = u[lag:].T @ u[:-lag]
        S += w * (G + G.T)

    # Small-sample correction, matching statsmodels' default for HAC.
    scale = n / (n - k)
    cov = XtX_inv @ (S * scale) @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)

    return RegressionResult(
        params=beta,
        stderr=se,
        tstat=t,
        r_squared=r2,
        adj_r_squared=adj_r2,
        n_obs=n,
        n_params=k,
        resid_std=float(np.sqrt(ss_res / (n - k))),
        hac_lags=lags,
    )


def percentiles(samples, qs=(50, 90, 99, 99.9)) -> dict[float, float]:
    """Percentiles of a sample, keyed by q. Empty input gives NaNs."""
    arr = np.asarray(list(samples), dtype=float)
    if arr.size == 0:
        return {q: float("nan") for q in qs}
    vals = np.percentile(arr, qs)
    return {q: float(v) for q, v in zip(qs, np.atleast_1d(vals), strict=False)}
