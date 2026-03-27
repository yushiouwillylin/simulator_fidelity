"""
DUQ / SimFidelity Quantile Utilities (gamma-only API)

Conventions
-----------
- gamma in (0,1) is the *coverage probability* of the confidence set C:
      P( p_true ∈ C(hat_p; gamma) ) >= gamma.
- delta := 1 - gamma is the miscoverage probability used inside log(·/delta).
- k is the simulator budget per scenario used to form qhat from model samples.

This module supports:
- Confidence sets for: bounded scalar mean, Bernoulli mean (KL-ball), multinomial (KL-ball)
- Upper pseudo-discrepancy: Δ^+_j = sup_{u in C_j} L(u, qhat_j)
- Lower pseudo-discrepancy: Δ^-_j = inf_{u in C_j} L(u, qhat_j)
- Pairwise pseudo-gap: δ_j = sup_{u in C_j} [L(u,q1hat)-L(u,q2hat)] for bounded scalar / Bernoulli

"""

from __future__ import annotations

import math
import re
import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

# ----------------------------
# Helpers
# ----------------------------

def _ensure_rng(rng: Optional[Union[int, np.random.Generator]] = None) -> np.random.Generator:
    """
    Normalize an RNG input.

    Parameters
    ----------
    rng : None, int, or np.random.Generator
        - None: use fresh default_rng()
        - int: treated as seed
        - Generator: used directly

    Returns
    -------
    np.random.Generator
    """
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)

def _delta_from_gamma(gamma: float, eps: float = 1e-12) -> float:
    """
    Convert coverage gamma to miscoverage delta = 1 - gamma (clipped for stability).
    """
    if not np.isfinite(gamma):
        raise ValueError("gamma must be finite.")
    if gamma <= 0.0 or gamma >= 1.0:
        raise ValueError("gamma must be in (0,1).")
    return float(np.clip(1.0 - gamma, eps, 1.0 - eps))

def _as_dict_series(obj: Union[Dict[Any, Any], pd.DataFrame]) -> Dict[str, pd.Series]:
    """
    Convert a dict-of-arrays or DataFrame into a dict[str -> pd.Series], coercing to numeric.

    Parameters
    ----------
    obj : dict or DataFrame
        - dict: keys are scenario ids; values are array-like
        - DataFrame: columns are scenario ids

    Returns
    -------
    dict[str, pd.Series]
    """
    if isinstance(obj, dict):
        return {str(k): pd.to_numeric(pd.Series(v, dtype="float64"), errors="coerce") for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return {str(c): pd.to_numeric(obj[c], errors="coerce") for c in obj.columns}
    raise TypeError("Expect dict[qid->Series/array] or DataFrame")

def _dropna_np(x: Any) -> np.ndarray:
    """
    Convert array-like to float numpy array with NaNs removed.
    """
    return pd.Series(x).dropna().to_numpy(dtype=float)

def _to_1d_numeric(x: Any) -> np.ndarray:
    """
    Convert array-like / Series / 1-col DataFrame to a 1D float array (may include NaNs).
    """
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("Expected a 1-column DataFrame for scalar/bernoulli outcomes.")
        x = x.iloc[:, 0]
    if isinstance(x, pd.Series):
        x = pd.to_numeric(x, errors="coerce").values
    return np.asarray(x, dtype=float).ravel()

def _subsample_human_1d(
    y_human_full: Any,
    n_target: Optional[int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, int, int]:
    """
    Drop missing and subsample up to n_target without replacement.

    Returns
    -------
    y_sub   : np.ndarray, float, length n_used
    n_used  : int
    n_avail : int
    """
    arr = _to_1d_numeric(y_human_full)
    arr = arr[np.isfinite(arr)]
    n_avail = int(arr.size)

    if n_target is None:
        return arr, n_avail, n_avail

    n_target = int(n_target)
    if n_target <= 0 or n_avail == 0:
        return np.empty((0,), dtype=float), 0, n_avail

    n_used = min(n_target, n_avail)
    idx = rng.choice(n_avail, size=n_used, replace=False)
    return arr[idx].astype(float, copy=False), n_used, n_avail

def _human_to_multinomial_counts(
    y_human_full: Any,
    n_target: Optional[int],
    rng: np.random.Generator,
    d: Optional[int] = None,
) -> Tuple[np.ndarray, int, int, int]:
    """
    Convert human data to a counts vector for multinomial CI.

    Accepts:
      (i)  counts vector (already aggregated; no subsampling possible)
      (ii) 1D integer labels (individual responses): subsample individuals -> bincount
      (iii)2D rows of probs/one-hot: subsample rows -> sum rows (expected counts)

    Returns
    -------
    counts  : np.ndarray float, shape (d_used,)
    n_used  : int (effective individuals used OR sum(counts) if counts input)
    n_avail : int (available individuals OR sum(counts) if counts input)
    d_used  : int
    """
    y = getattr(y_human_full, "values", y_human_full)
    arr = np.asarray(y)

    # Case (i): counts vector
    if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
        v = np.asarray(arr, dtype=float)
        if np.any(~np.isfinite(v)):
            v = v[np.isfinite(v)]
        if v.size == 0 or np.any(v < 0):
            return np.zeros((0,), dtype=float), 0, 0, 0
        tot = int(round(float(v.sum())))
        if tot <= 0:
            return np.zeros((0,), dtype=float), 0, 0, 0
        if d is not None and int(d) != int(v.size):
            raise ValueError("Provided d does not match counts length.")
        return v, tot, tot, int(v.size)

    # Case (iii): 2D rows
    if arr.ndim == 2:
        n_avail = int(arr.shape[0])
        if n_avail == 0:
            return np.zeros((0,), dtype=float), 0, 0, 0
        d_used = int(arr.shape[1] if d is None else d)
        n_used = n_avail if n_target is None else min(int(n_target), n_avail)
        idx = rng.choice(n_avail, size=n_used, replace=False)

        rows = np.asarray(arr[idx], dtype=float)[:, :d_used]
        rows = np.clip(rows, 0.0, 1.0)
        rows = rows / np.maximum(rows.sum(axis=1, keepdims=True), 1e-12)

        counts = rows.sum(axis=0)
        return counts.astype(float), int(n_used), int(n_avail), int(d_used)

    # Case (ii): 1D labels
    labels = np.asarray(arr).ravel()
    if np.issubdtype(labels.dtype, np.floating):
        labels = labels[np.isfinite(labels)]
    labels = labels.astype(int, copy=False)
    labels = labels[labels >= 0]

    n_avail = int(labels.size)
    if n_avail == 0:
        return np.zeros((0,), dtype=float), 0, 0, 0

    n_used = n_avail if n_target is None else min(int(n_target), n_avail)
    idx = rng.choice(n_avail, size=n_used, replace=False)
    lab = labels[idx]

    d_used = int((lab.max() + 1) if d is None else d)
    counts = np.bincount(lab, minlength=d_used).astype(float)
    return counts, int(n_used), int(n_avail), int(d_used)

def empirical_quantile_curve(values: Sequence[float], alpha_grid: np.ndarray) -> np.ndarray:
    """
    Left-continuous empirical quantile curve with a monotone envelope.

    This is intended to behave like a CDF-inverse with left-continuity,
    using order statistics with index:
        idx(alpha) = ceil(m * alpha) - 1   (clipped to [0, m-1])

    Parameters
    ----------
    values : sequence of float
        Sample values (e.g., {Δ_j}).
    alpha_grid : np.ndarray
        Array of alpha in [0,1].

    Returns
    -------
    np.ndarray
        Quantiles evaluated on alpha_grid, with monotone nondecreasing enforcement.
    """
    s = np.sort(np.asarray(values, dtype=float))
    m = len(s)
    if m == 0:
        return np.full_like(alpha_grid, np.nan, dtype=float)

    alpha_grid = np.asarray(alpha_grid, dtype=float)
    idx = np.ceil(m * alpha_grid) - 1.0
    idx = np.clip(idx.astype(int), 0, m - 1)

    q = s[idx]
    # enforce monotonicity in alpha_grid order
    return np.maximum.accumulate(q)


# ----------------------------
# Localized conformal utilities
# ----------------------------

def _lc_pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances for row vectors."""
    X = np.asarray(X, dtype=float)
    g = np.sum(X * X, axis=1, keepdims=True)
    d2 = g + g.T - 2.0 * (X @ X.T)
    return np.maximum(d2, 0.0)


def _lc_build_raw_localizer_matrix(
    X_aug: np.ndarray,
    *,
    localizer_kind: str,
    localizer_kwargs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Build unnormalized localization matrix H on augmented points.

    H[i, j] is the localizer value centered at row i, evaluated at row j.
    This is the quantity used directly by the deterministic Guan Algorithm 1 scan.
    """
    localizer_kwargs = localizer_kwargs or {}
    kind = str(localizer_kind).lower()

    X_aug = np.asarray(X_aug, dtype=float)
    n = int(X_aug.shape[0])
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    if kind == "global":
        return np.ones((n, n), dtype=float)

    gamma = float(localizer_kwargs.get("gamma", 1.0))
    min_weight = float(localizer_kwargs.get("min_weight", 0.0))
    d2 = _lc_pairwise_sq_dists(X_aug)

    if kind == "rbf":
        H = np.exp(-gamma * d2)
        if min_weight > 0.0:
            H = np.maximum(H, min_weight)
        return H

    if kind in ("gaussian", "gaussian_rbf"):
        # Gaussian kernel: exp(-||x-x'||^2 / (2*sigma^2)).
        sigma = localizer_kwargs.get("sigma", None)
        if sigma is None:
            sigma = localizer_kwargs.get("h", None)
        if sigma is not None:
            sigma = float(sigma)
            if sigma <= 0.0:
                raise ValueError("gaussian localizer requires sigma > 0.")
            gamma_eff = 1.0 / (2.0 * sigma * sigma)
        else:
            # Backward-compatible parameterization with gamma.
            gamma_eff = float(gamma)
        H = np.exp(-gamma_eff * d2)
        if min_weight > 0.0:
            H = np.maximum(H, min_weight)
        return H

    if kind == "gaussian_median":
        # Median-distance bandwidth: sigma = median(||x_i-x_j||) over nonzero pairs.
        d = np.sqrt(np.maximum(d2, 0.0))
        upper = d[np.triu_indices(n, k=1)]
        upper = upper[np.isfinite(upper)]
        upper = upper[upper > 0.0]
        sigma = float(np.median(upper)) if upper.size else 1.0
        sigma = max(sigma, 1e-8)
        gamma_eff = 1.0 / (2.0 * sigma * sigma)
        H = np.exp(-gamma_eff * d2)
        if min_weight > 0.0:
            H = np.maximum(H, min_weight)
        return H

    if kind == "knn_rbf":
        k_nn = int(localizer_kwargs.get("k_nn", 60))
        k = min(max(1, k_nn), n)
        H = np.zeros((n, n), dtype=float)
        for i in range(n):
            idx = np.argpartition(d2[i], k - 1)[:k]
            vals = np.exp(-gamma * d2[i, idx])
            if min_weight > 0.0:
                vals = np.maximum(vals, min_weight)
            H[i, idx] = vals
        return H

    raise ValueError(f"Unknown localizer kind: {localizer_kind}")


class GuanLocalizedConformalUpper:
    """
    Deterministic sample-splitting localized conformal upper bound (Guan 2023).

    This implements the finite-support, single-scan deterministic procedure
    (Lemma 2 / Algorithm 1 style) for one-sided scores:
        V(x, y) = y - g(x)
    with API miscoverage `alpha_api`, so the target coverage is
        coverage_target = 1 - alpha_api.

    Notes
    -----
    - This class does not refit the base model; it only consumes calibration
      features and calibration scores.
    - Optional randomized exact-level variant (Theorem 2) is not implemented.
    """

    def __init__(
        self,
        alpha_api: float = 0.1,
        *,
        alpha: Optional[float] = None,
        localizer_kind: str = "rbf",
        localizer_kwargs: Optional[Dict[str, Any]] = None,
        **_: Any,
    ):
        if alpha is not None:
            alpha_api = float(alpha)
        self.alpha_api = float(alpha_api)
        if not (0.0 < self.alpha_api < 1.0):
            raise ValueError("alpha_api must be in (0,1).")
        self.coverage_target = float(1.0 - self.alpha_api)
        self.localizer_kind = str(localizer_kind)
        self.localizer_kwargs = dict(localizer_kwargs or {})

    def fit(self, X_cal: np.ndarray, scores_cal: np.ndarray):
        """Store calibration features/scores (no retraining)."""
        self.X_cal = np.asarray(X_cal, dtype=float)
        self.scores = np.asarray(scores_cal, dtype=float).ravel()
        if self.X_cal.ndim != 2:
            raise ValueError("X_cal must be 2D.")
        if self.X_cal.shape[0] != self.scores.shape[0]:
            raise ValueError("X_cal and scores_cal must have the same number of rows.")
        self.n_cal_ = int(self.scores.shape[0])
        return self

    @staticmethod
    def _standard_split_upper_correction(scores: np.ndarray, alpha_api: float) -> float:
        """
        Standard one-sided split conformal correction from calibration scores.
        """
        s = np.sort(np.asarray(scores, dtype=float))
        n = int(s.size)
        if n == 0:
            return float("nan")
        coverage_target = float(1.0 - alpha_api)
        rank = int(np.ceil((n + 1) * coverage_target))
        if rank <= 0:
            return float("-inf")
        if rank >= n + 1:
            return float("inf")
        return float(s[rank - 1])

    @staticmethod
    def _prefix_at(cum: np.ndarray, l_val: int) -> float:
        """Return cumulative sum up to l_val (1-based); 0 when l_val=0."""
        if l_val <= 0:
            return 0.0
        return float(cum[int(l_val) - 1])

    def _single_correction(self, x_star: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        n = int(self.n_cal_)
        if n == 0:
            return float("nan"), {
                "k_star": 0,
                "v_star": np.nan,
                "n_cal": 0,
                "coverage_target": float(self.coverage_target),
                "is_inf_v_star": False,
                "c1_at_k_star": 0,
                "c2_at_k_star": 0,
                "c3_at_k_star": 0,
            }

        # Stable sort so tie handling is deterministic.
        perm = np.argsort(self.scores, kind="mergesort")
        v_sorted = self.scores[perm]
        X_sorted = self.X_cal[perm]

        # Global localizer must reduce to ordinary split conformal exactly.
        if self.localizer_kind.lower() == "global":
            v_star = self._standard_split_upper_correction(v_sorted, self.alpha_api)
            k_star = n + 1 if np.isinf(v_star) else int(np.searchsorted(v_sorted, v_star, side="left") + 1)
            return float(v_star), {
                "k_star": int(k_star),
                "v_star": float(v_star),
                "n_cal": int(n),
                "coverage_target": float(self.coverage_target),
                "is_inf_v_star": bool(np.isinf(v_star)),
                "c1_at_k_star": np.nan,
                "c2_at_k_star": np.nan,
                "c3_at_k_star": np.nan,
            }

        X_aug = np.vstack([X_sorted, np.asarray(x_star, dtype=float).reshape(1, -1)])
        H = _lc_build_raw_localizer_matrix(
            X_aug,
            localizer_kind=self.localizer_kind,
            localizer_kwargs=self.localizer_kwargs,
        )

        H_cc = H[:n, :n]
        H_ct = H[:n, n]
        H_tc = H[n, :n]
        H_tt = float(H[n, n])

        # Q_{i,k} = sum_{j<=k} H_{i,j} in sorted-score order.
        Q_cal = np.cumsum(H_cc, axis=1)
        Q_test = np.cumsum(H_tc)

        row_sum_cal = Q_cal[:, -1] + H_ct
        row_sum_test = float(Q_test[-1] + H_tt)

        # l(i) = max{j in {1..n}: V_bar[j] < V_bar[i]}, max(empty)=0.
        l_cal = np.searchsorted(v_sorted, v_sorted, side="left").astype(int)
        l_all = np.concatenate([l_cal, np.array([n], dtype=int)], axis=0)

        # tilde_theta_i for i=1..n+1
        tilde_theta = np.zeros(n + 1, dtype=float)
        denom_test = row_sum_test if row_sum_test > 0.0 else 1.0
        for i in range(n + 1):
            num = self._prefix_at(Q_test, int(l_all[i]))
            tilde_theta[i] = float(num / denom_test)

        theta = np.zeros(n, dtype=float)
        theta_plus = np.zeros(n, dtype=float)
        for i in range(n):
            denom_i = float(row_sum_cal[i]) if row_sum_cal[i] > 0.0 else 1.0
            q_il = self._prefix_at(Q_cal[i], int(l_cal[i]))
            theta[i] = float(q_il / denom_i)
            theta_plus[i] = float((q_il + H_ct[i]) / denom_i)

        # Partition A1, A2, A3 (calibration indices only).
        a1_vals = []
        a2_vals = []
        a3_lvals = []
        for i in range(n):
            t_tilde = tilde_theta[i]
            if theta_plus[i] < t_tilde:
                a1_vals.append(float(theta_plus[i]))
            elif theta[i] >= t_tilde:
                a2_vals.append(float(theta[i]))
            else:
                a3_lvals.append(int(l_cal[i]))

        a1_vals = np.sort(np.asarray(a1_vals, dtype=float))
        a2_vals = np.sort(np.asarray(a2_vals, dtype=float))
        a3_lvals = np.sort(np.asarray(a3_lvals, dtype=int))

        # Single-scan Algorithm 1 over k=1,...,n+1.
        c1 = c2 = c3 = 0
        s_vals = np.zeros(n + 1, dtype=float)
        c_hist = np.zeros((n + 1, 3), dtype=int)
        for k in range(n + 1):
            tk = float(tilde_theta[k])
            lk = int(l_all[k])
            while c1 < a1_vals.size and a1_vals[c1] < tk:
                c1 += 1
            while c2 < a2_vals.size and a2_vals[c2] < tk:
                c2 += 1
            while c3 < a3_lvals.size and a3_lvals[c3] < lk:
                c3 += 1
            s_vals[k] = float((c1 + c2 + c3) / float(n + 1))
            c_hist[k, 0] = c1
            c_hist[k, 1] = c2
            c_hist[k, 2] = c3

        admissible = np.where(s_vals < float(self.coverage_target))[0]
        if admissible.size == 0:
            k_star_1b = 0
            v_star = float("-inf")
            c1s = c2s = c3s = 0
        else:
            k_star_1b = int(admissible[-1] + 1)
            if k_star_1b <= n:
                v_star = float(v_sorted[k_star_1b - 1])
            else:
                v_star = float("inf")
            c1s, c2s, c3s = map(int, c_hist[k_star_1b - 1].tolist())

        return float(v_star), {
            "k_star": int(k_star_1b),
            "v_star": float(v_star),
            "n_cal": int(n),
            "coverage_target": float(self.coverage_target),
            "is_inf_v_star": bool(np.isinf(v_star)),
            "c1_at_k_star": int(c1s),
            "c2_at_k_star": int(c2s),
            "c3_at_k_star": int(c3s),
        }

    def correction(self, X_test: np.ndarray) -> Tuple[np.ndarray, list[Dict[str, Any]]]:
        """Return per-test correction values and Algorithm 1 metadata."""
        X_test = np.asarray(X_test, dtype=float)
        if X_test.ndim != 2:
            raise ValueError("X_test must be 2D.")
        if X_test.shape[1] != self.X_cal.shape[1]:
            raise ValueError("X_test feature dimension must match X_cal.")

        corr = np.zeros(X_test.shape[0], dtype=float)
        meta_rows: list[Dict[str, Any]] = []
        for i in range(X_test.shape[0]):
            v_star, meta = self._single_correction(X_test[i])
            corr[i] = float(v_star)
            meta_rows.append(meta)
        return corr, meta_rows

# ----------------------------
# Losses
# ----------------------------

def loss_abs_mean_gap(p: float, q: float) -> float:
    """L(p,q) = |p - q| for scalar means."""
    return abs(float(p) - float(q))

def loss_sq_mean_gap(p: float, q: float) -> float:
    """L(p,q) = (p - q)^2 for scalar means."""
    d = float(p) - float(q)
    return d * d

def loss_tv_bern(p: float, q: float) -> float:
    """
    Bernoulli total variation distance equals |p - q|.
    This is a special case of the tv, but we keep it separate for clarity and potential future extensions (e.g., to other distributions).
    """
    return abs(float(p) - float(q))

def loss_kl_bern(p: float, q: float, eps: float = 1e-12) -> float:
    """
    Bernoulli KL divergence KL(p || q).

    Parameters
    ----------
    p, q : float in [0,1]
    eps : float
        Numerical clipping to avoid log(0).

    Returns
    -------
    float
        KL(p||q)
    """
    p = float(np.clip(p, eps, 1.0 - eps))
    q = float(np.clip(q, eps, 1.0 - eps))
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))

def get_loss(loss_kind: str) -> Callable:
    """
    Return a scenario-wise loss function.

    Supported loss_kind
    -------------------
    - 'abs' : |u - q|
    - 'sq'  : (u - q)^2
    - 'tv'  : Bernoulli TV = |p - q|
    - 'kl'  : Bernoulli KL(p||q)

    Notes
    -----
    - For Bernoulli means, 'tv' is numerically identical to 'abs'.
    """
    if loss_kind == "abs":
        return loss_abs_mean_gap
    if loss_kind in ("sq", "l2"):
        return loss_sq_mean_gap
    if loss_kind == "tv":
        return loss_tv_bern
    if loss_kind == "kl":
        return loss_kl_bern
    raise ValueError(f"Unknown loss_kind={loss_kind}")

# ----------------------------
# Confidence sets
# ----------------------------

# --- Bounded Scalar Mean CI

def ci_bounded_mean(
    y: Any,
    gamma: float,
    bounds: Tuple[float, float] = (-1.0, 1.0),
    method: str = "hoeffding",
) -> Tuple[float, Tuple[float, float], Dict[str, Any]]:
    """
    Confidence interval for a bounded scalar mean.

    Model
    -----
    Observations Y_i are bounded in [a,b] (default [-1,1]).
    Let m = E[Y]. Construct C(hat m; gamma) = [L,U] such that
        P( m ∈ [L,U] ) >= gamma.

    Implementation
    --------------
    Uses Hoeffding radius with delta = 1-gamma:
      rad = (b-a) * sqrt( log(2/delta) / (2n) ).

    Parameters
    ----------
    y : array-like
        Human samples.
    gamma : float
        Coverage probability in (0,1).
    bounds : (a,b)
        Known bounds of the variable.
    method : {'hoeffding'}
        Kept for API compatibility; non-Hoeffding values are ignored.

    Returns
    -------
    mhat : float
    (L,U) : tuple
        CI endpoints clipped to [a,b].
    stats : dict
        Bookkeeping (n, rad, gamma, delta, method, sample variance).
    """
    a0, b0 = float(bounds[0]), float(bounds[1])
    if not (a0 < b0):
        raise ValueError("bounds must satisfy a < b.")

    delta = _delta_from_gamma(gamma) # From coverage gamma to miscoverage delta = 1 - gamma
    yy = np.asarray(y, dtype=float)
    yy = yy[np.isfinite(yy)]
    n = int(yy.size)

    if n == 0:
        return np.nan, (np.nan, np.nan), {"n": 0, "rad": np.nan, "gamma": gamma, "delta": delta, "method": "hoeffding"}

    phat = float(np.mean(yy))
    s2 = float(np.var(yy, ddof=1)) if n > 1 else 0.0

    width = (b0 - a0)
    rad = width * math.sqrt(math.log(2.0 / delta) / (2.0 * n))
    used = "hoeffding"

    L = max(a0, phat - rad)
    U = min(b0, phat + rad)

    return phat, (L, U), {"n": n, "rad": rad, "gamma": gamma, "delta": delta, "method": used, "s2": s2, "bounds": bounds}


# ---- Bernoulli KL-ball

def _D_kl_ph_to_p(ph: float, p: float) -> float:
    """
    Create a function to calculate KL( ph || p ) for Bernoulli parameters.
    Used to invert KL-ball endpoints.

    Returns +inf if p is outside (0,1) in a way that makes KL undefined.
    """
    ph = float(ph)
    p = float(p)

    if ph <= 0.0:
        if p >= 1.0:
            return float("inf")
        return -math.log(max(1.0 - p, 1e-300))
    if ph >= 1.0:
        if p <= 0.0:
            return float("inf")
        return -math.log(max(p, 1e-300))

    if p <= 0.0 or p >= 1.0:
        return float("inf")

    return ph * math.log(ph / p) + (1.0 - ph) * math.log((1.0 - ph) / (1.0 - p))

def _bisect_target(fun: Callable[[float], float], a: float, b: float, target: float, tol: float = 1e-10, it: int = 100) -> float:
    """
    Generic bisection to find x in [a,b] such that fun(x) ~= target, and the function in this case would be KL defined above.
    Assumes fun is (weakly) monotone over [a,b].
    Note that his function will be separately adopted at right and left hand side.
    """
    fa, fb = fun(a) - target, fun(b) - target
    if fa * fb > 0:
        # Not bracketed: return the endpoint with closer value, ie. make sure Intermediate Value Theorem holds. 
        return a if abs(fa) < abs(fb) else b

    lo, hi = a, b
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        fm = fun(mid) - target
        if fa * fm <= 0:
            hi, fb = mid, fm
        else:
            lo, fa = mid, fm
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)

def ci_bernoulli_kl(
    phat: float,
    n: int,
    gamma: float,
) -> Tuple[float, Tuple[float, float], Dict[str, Any]]:
    """
    KL-ball confidence set for a Bernoulli mean.

    Set
    ---
    C = { p in [0,1] : KL(phat || p) <= r },  with r = log(2/delta)/n, delta = 1-gamma.

    Parameters
    ----------
    phat : float
        Empirical mean in [0,1].
    n : int
        Sample size.
    gamma : float
        Coverage probability in (0,1).

    Returns
    -------
    phat : float
    (pL,pU) : tuple
        Endpoints of the KL-ball intersection with [0,1].
    stats : dict
        Contains n, r, gamma, delta.
    """
    delta = _delta_from_gamma(gamma)
    if n <= 0 or not np.isfinite(phat):
        return np.nan, (np.nan, np.nan), {"n": int(n), "r": np.nan, "gamma": gamma, "delta": delta}

    ph = float(np.clip(phat, 0.0, 1.0))
    r = math.log(2.0 / delta) / max(1, int(n))

    if r == 0.0:
        return ph, (ph, ph), {"n": int(n), "r": float(r), "gamma": gamma, "delta": delta}

    if ph <= 0.0:
        pL, pU = 0.0, 1.0 - math.exp(-r)
    elif ph >= 1.0:
        pL, pU = math.exp(-r), 1.0
    else:
        pL = _bisect_target(lambda p: _D_kl_ph_to_p(ph, p), 0.0, ph, r)
        pU = _bisect_target(lambda p: _D_kl_ph_to_p(ph, p), ph, 1.0, r)

    return ph, (float(pL), float(pU)), {"n": int(n), "r": float(r), "gamma": gamma, "delta": delta}

# ---- Multinomial KL-ball

def ci_multinomial_klball(
    phat_vec_or_counts: Any,
    gamma: float,
    n: Optional[int] = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Multinomial KL-ball confidence set around empirical probabilities, and return a radius.

    Set
    ---
    C = { u in simplex : KL(phat || u) <= r },
    with r = (d-1)/n * log( 2(d-1)/delta ), delta = 1-gamma.

    Parameters
    ----------
    phat_vec_or_counts : array-like
        Either normalized probabilities (sum to 1) or raw counts.
    gamma : float
        Coverage probability in (0,1).
    n : int, optional
        Total sample size. If None, uses sum(counts).
    eps : float
        Numerical floor for probabilities.

    Returns
    -------
    phat : np.ndarray
        Normalized empirical distribution in the simplex.
    r : float
        KL radius.
    stats : dict
        Bookkeeping fields (n, d, gamma, delta, method).
    """
    delta = _delta_from_gamma(gamma)
    v = np.asarray(phat_vec_or_counts, dtype=float)

    total = float(np.sum(v)) # Handling the case where sum of multinomial might not = 1.
    if n is None:
        if np.all(v >= 0) and np.all(v <= 1) and abs(total - 1.0) < 1e-6:
            raise ValueError("ci_multinomial_klball: n must be provided when phat is already normalized probabilities.")
        n_eff = int(round(total))
    else:
        n_eff = int(n)

    if total <= 0.0 or n_eff <= 0:
        return None, np.nan, {"n": n_eff, "r": np.nan, "d": np.nan, "gamma": gamma, "delta": delta, "method": "multinomial"}

    phat = v / total # Convert count to probability
    phat = np.clip(phat, eps, 1.0) # Entry-wise
    phat = phat / phat.sum()

    d = int(phat.size) # Dimensionality of the multinomial vector
    if d <= 1:
        return phat, 0.0, {"n": n_eff, "r": 0.0, "d": d, "gamma": gamma, "delta": delta, "method": "multinomial"}

    r = (d - 1.0) / n_eff * math.log(2.0 * (d - 1.0) / delta)
    return phat, float(r), {"n": n_eff, "r": float(r), "d": d, "gamma": gamma, "delta": delta, "method": "multinomial"}

def _kl_div_vec(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p||q) for discrete distributions with clipping. Numerically stabelized by clipping out 0."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, eps, 1.0); p = p / p.sum()
    q = np.clip(q, eps, 1.0); q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))

def _multinomial_kl_inner_max(phat: np.ndarray, qhat: np.ndarray, r: float, tol: float = 1e-9, max_iter: int = 200) -> Tuple[np.ndarray, float]:
    """
    Solve: max_{u: KL(phat||u) <= r} KL(u||qhat) via a 1D dual search.

    Returns
    -------
    u_star : np.ndarray
    value : float
        KL(u_star || qhat)

    Notes
    -----
    This is a specialized routine; keep it isolated so you can swap the solver later. Specifically, it is not clear how far this approximation is from the actual answer. Can replace with grid search if necessary.
    """
    phat = np.asarray(phat, dtype=float)
    qhat = np.asarray(qhat, dtype=float)

    phat = np.clip(phat, 1e-12, 1.0); phat /= phat.sum()
    qhat = np.clip(qhat, 1e-12, 1.0); qhat /= qhat.sum()

    def u_lambda(lam: float) -> np.ndarray:
        """
        Solve the lagrangian with \lambda to get this u_lambda representation.
        """
        # u_i ∝ phat_i^{1/(1+lam)} * qhat_i^{-lam/(1+lam)}
        expo1 = 1.0 / (1.0 + lam)
        expo2 = -lam / (1.0 + lam)
        u = (phat ** expo1) * (qhat ** expo2)
        u = u / u.sum()
        return u

    # Check feasibility at lambda = 0
    u0 = u_lambda(0.0)
    if _kl_div_vec(phat, u0) > r:
        # Fallback: if numerical issues, return phat (most conservative)
        return phat, _kl_div_vec(phat, qhat)

    # Set lambda range to be large enough to bisect such that KL on both side multiplies to negative.
    lam_lo, lam_hi = 0.0, 100.0
    for _ in range(max_iter):
        u = u_lambda(lam_hi)
        if _kl_div_vec(phat, u) > r:
            break
        lam_hi *= 2.0

    # Bisection to enforce KL(phat||u)=r
    u_star = u0
    for _ in range(max_iter):
        lam = 0.5 * (lam_lo + lam_hi)
        u = u_lambda(lam)
        kl_val = _kl_div_vec(phat, u)
        u_star = u

        if abs(kl_val - r) < tol:
            break
        if kl_val > r:
            lam_hi = lam
        else:
            lam_lo = lam

    return u_star, _kl_div_vec(u_star, qhat)

def _multinomial_kl_inner_min_cvxpy(
    phat: np.ndarray,
    qhat: np.ndarray,
    r: float,
    eps: float = 1e-12,
    solver: str = "ECOS",
    verbose: bool = False,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Exact inner minimization using CVXPY:

        minimize_u   KL(u || qhat)
        subject to   KL(phat || u) <= r,
                    u >= 0, sum(u) = 1.

    Notes
    -----
    - Uses rel_entr for DCP compliance.
    - Clips/renormalizes phat and qhat to avoid zeros.
    - If qhat is feasible (KL(phat||qhat) <= r), optimum is u=qhat and value 0.
    """
    try:
        import cvxpy as cp
    except ImportError as e:
        raise ImportError("cvxpy is required for _multinomial_kl_inner_min_cvxpy.") from e

    info: Dict[str, Any] = {"method": "cvxpy_inner_min_kl_over_klball", "solver": solver}

    p = np.asarray(phat, dtype=float).copy()
    q = np.asarray(qhat, dtype=float).copy()
    if p.ndim != 1 or q.ndim != 1 or p.size != q.size:
        raise ValueError("phat and qhat must be 1D arrays of the same length.")

    d = int(p.size)
    if d <= 1:
        u = np.array([1.0], dtype=float)
        return u, 0.0, {**info, "d": d, "status": "degenerate"}

    if not np.isfinite(r) or r < 0:
        raise ValueError("r must be finite and nonnegative.")

    # Stabilize inputs (CVXPY can handle zeros in rel_entr in some cases, but q=0 causes infinities in KL(u||q))
    p = np.clip(p, eps, 1.0); p /= p.sum()
    q = np.clip(q, eps, 1.0); q /= q.sum()

    # Variables
    u = cp.Variable(d, nonneg=True)

    # KL(u||q) = sum_i u_i log(u_i / q_i)
    obj = cp.Minimize(cp.sum(cp.rel_entr(u, q)))

    # KL(p||u) = sum_i p_i log(p_i / u_i) = sum rel_entr(p, u)
    # p is constant here
    constraints = [
        cp.sum(u) == 1,
        cp.sum(cp.rel_entr(p, u)) <= r,
    ]

    prob = cp.Problem(obj, constraints)
    prob.solve(solver=solver, verbose=verbose)

    info["status"] = prob.status
    if prob.status not in ("optimal", "optimal_inaccurate"):
        # Fall back: return feasible point p (always satisfies KL(p||p)=0 <= r)
        u_fallback = p.copy()
        val = float(_kl_div_vec(u_fallback, q, eps=eps))
        return u_fallback, val, {**info, "note": "solver_failed_fallback_phat", "value": val}

    u_star = np.asarray(u.value, dtype=float).ravel()
    u_star = np.clip(u_star, eps, 1.0)
    u_star = u_star / u_star.sum()

    val = float(_kl_div_vec(u_star, q, eps=eps))
    # Diagnostics
    info["value"] = float(prob.value)  # should match val up to eps effects
    info["kl_p_u"] = float(_kl_div_vec(p, u_star, eps=eps))
    info["kl_u_q"] = float(val)
    info["d"] = d

    return u_star, val, info

# ---- Simulator Side Helpers 

def _model_mean_from_samples(y_model: Any, k: int, rng: np.random.Generator) -> Tuple[float, int]:
    """
    Compute qhat as the mean of k subsampled finite model outputs.

    Parameters
    ----------
    y_model : array-like
        Model outputs for a scenario.
    k : int
        Subsample budget.
    rng : np.random.Generator

    Returns
    -------
    qhat : float
    k_used : int
    """
    ym = np.asarray(getattr(y_model, "values", y_model), dtype=float)
    ym = ym[np.isfinite(ym)]
    if ym.size == 0:
        return np.nan, 0

    k_used = int(min(int(k), int(ym.size)))
    qhat = float(rng.choice(ym, size=k_used, replace=False).mean())
    return qhat, k_used

def _model_prob_from_samples(y_model_samples: Any, k: int, d: int, rng: np.random.Generator) -> Tuple[np.ndarray, int]:
    """
    Compute qhat (categorical probabilities) from model outputs.

    Accepts either:
    - 2D array of probabilities / one-hot rows: shape (N, d)
    - 1D array of integer category labels (label mode)

    Parameters
    ----------
    y_model_samples : array-like
    k : int
    d : int
        Number of categories (target dimension).
    rng : np.random.Generator

    Returns
    -------
    qhat : np.ndarray shape (d,)
    k_used : int
    """
    ys = np.asarray(getattr(y_model_samples, "values", y_model_samples))
    if ys.ndim == 2:
        nrows = ys.shape[0]
        k_used = int(min(int(k), int(nrows)))
        idx = rng.choice(nrows, size=k_used, replace=False)
        qhat = np.asarray(ys[idx].mean(axis=0), dtype=float)
        qhat = qhat[:d]
        qhat = np.clip(qhat, 1e-12, 1.0)
        qhat = qhat / qhat.sum()
        return qhat, k_used

    # label mode
    ys = ys.astype(int, copy=False)
    n = ys.size
    k_used = int(min(int(k), int(n)))
    idx = rng.choice(n, size=k_used, replace=False)
    counts = np.bincount(ys[idx], minlength=d).astype(float)
    qhat = counts / max(1, k_used)
    qhat = np.clip(qhat, 1e-12, 1.0)
    qhat = qhat / qhat.sum()
    return qhat, k_used

# ----------------------------
# Upper pseudo-discrepancies Δ+
# ----------------------------

def pseudo_delta_scalar_bounded(
    y_human: Any,
    y_model: Any,
    k: int,
    gamma: float,
    loss_kind: str,
    ci_method: str = "hoeffding",
    bounds: Tuple[float, float] = (-1.0, 1.0),
    n_target: Optional[int] = None,   # NEW
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[float, Dict[str, Any]]:
    rng = _ensure_rng(rng)

    yh, n_used, n_avail = _subsample_human_1d(y_human, n_target=n_target, rng=rng)
    if n_used == 0:
        return np.nan, {"reason": "empty_human", "n_target": n_target, "n_used": 0, "n_avail": n_avail}

    m_h, (L, U), stats = ci_bounded_mean(yh, gamma=gamma, bounds=bounds, method=ci_method)

    qhat, k_used = _model_mean_from_samples(y_model, k=k, rng=rng)
    if not np.isfinite(qhat) or k_used <= 0:
        return np.nan, {"reason": "empty_model"}

    if loss_kind == "abs":
        d_plus = max(abs(L - qhat), abs(U - qhat))
    elif loss_kind in ("sq", "l2"):
        d_plus = max((L - qhat) ** 2, (U - qhat) ** 2)
    else:
        raise ValueError("bounded scalar: loss_kind must be in {'abs','sq'}")

    info = {
        "family": "bounded",
        "loss_kind": loss_kind,
        "gamma": gamma,
        "delta": stats.get("delta"),
        "n_h": stats.get("n"),
        "n_target": n_target,
        "n_used": n_used,
        "n_avail": n_avail,
        "rad": stats.get("rad"),
        "ci_method": stats.get("method"),
        "ci_interval": (L, U),
        "m_h": m_h,
        "qhat": qhat,
        "k_used": k_used,
        "bounds": bounds,
    }
    return float(d_plus), info

def pseudo_delta_bernoulli(
    y_human: Any,
    y_model: Any,
    k: int,
    gamma: float,
    loss_kind: str,
    n_target: Optional[int] = None, 
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[float, Dict[str, Any]]:
    rng = _ensure_rng(rng)

    yh, n_used, n_avail = _subsample_human_1d(y_human, n_target=n_target, rng=rng)
    if n_used == 0:
        return np.nan, {"reason": "empty_human", "n_target": n_target, "n_used": 0, "n_avail": n_avail}

    phat = float(np.mean(yh))
    _, (pL, pU), ci_info = ci_bernoulli_kl(phat, n=int(yh.size), gamma=gamma)

    qhat, k_used = _model_mean_from_samples(y_model, k=k, rng=rng)
    if not np.isfinite(qhat) or k_used <= 0:
        return np.nan, {"reason": "empty_model"}

    if loss_kind == "tv":
        Lfun = loss_tv_bern
    elif loss_kind in ("sq", "l2"):
        Lfun = loss_sq_mean_gap
    elif loss_kind == "kl":
        Lfun = loss_kl_bern
    else:
        raise ValueError("bernoulli: loss_kind must be in {'tv','sq','kl'}")

    d_plus = max(Lfun(pL, qhat), Lfun(pU, qhat))

    info = {
        "family": "bernoulli",
        "loss_kind": loss_kind,
        "gamma": gamma,
        "delta": ci_info.get("delta"),
        "n_h": ci_info.get("n"),
        "n_target": n_target,
        "n_used": n_used,
        "n_avail": n_avail,
        "r": ci_info.get("r"),
        "phat": phat,
        "pL": pL,
        "pU": pU,
        "qhat": qhat,
        "k_used": k_used,
    }
    return float(d_plus), info

def pseudo_delta_multinomial(
    y_human_counts: Any,           # can be full human labels/rows/counts
    y_model_samples: Any,
    k: int,
    gamma: float,
    loss_kind: str,
    n_target: Optional[int] = None,   # NEW
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[float, Dict[str, Any]]:
    rng = _ensure_rng(rng)

    counts, n_used, n_avail, d_used = _human_to_multinomial_counts(
        y_human_full=y_human_counts, n_target=n_target, rng=rng, d=None
    )
    if counts.size == 0 or n_used == 0:
        return np.nan, {"reason": "empty_human", "n_target": n_target, "n_used": n_used, "n_avail": n_avail}

    phat, r, ci_info = ci_multinomial_klball(counts, gamma=gamma, n=None)
    if phat is None or not np.isfinite(r):
        return np.nan, {"reason": "empty_human"}

    d = int(phat.size)
    qhat, k_used_model = _model_prob_from_samples(y_model_samples, k=k, d=d, rng=rng)
    if k_used_model <= 0:
        return np.nan, {"reason": "empty_model"}

    if loss_kind == "tv":
        tv_ph_q = 0.5 * float(np.sum(np.abs(phat - qhat)))
        d_plus = tv_ph_q + math.sqrt(0.5 * r)
        details = {"tv(phat,qhat)": tv_ph_q, "bound": "tv + sqrt(r/2)"}
    elif loss_kind == "kl":
        u_star, kl_val = _multinomial_kl_inner_max(phat, qhat, r)
        d_plus = float(kl_val)
        details = {"KL(u*,qhat)": float(kl_val), "solver": "dual-bisect"}
    else:
        raise ValueError("multinomial: loss_kind must be in {'tv','kl'}")

    info = {
        "family": "multinomial",
        "loss_kind": loss_kind,
        "gamma": gamma,
        "delta": ci_info.get("delta"),
        "n_h": ci_info.get("n"),
        "n_target": n_target,
        "n_used": n_used,
        "n_avail": n_avail,
        "d_used": d_used,
        "d": ci_info.get("d"),
        "r": ci_info.get("r"),
        "k_used": k_used_model,
        **details,
    }
    return float(d_plus), info

def compute_pseudo_delta(
    y_human: Any,
    y_model: Any,
    k: int = 200,
    gamma: float = 0.5,
    ci_family: str = "bounded",
    loss_kind: str = "abs",
    ci_kwargs: Optional[Dict[str, Any]] = None,
    n_target: Optional[int] = None,   # NEW
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Unified entry point for Δ^+.

    Parameters
    ----------
    y_human : array-like
    y_model : array-like
    k : int
        Model subsample budget.
    gamma : float
        Confidence set coverage probability in (0,1).
    ci_family : {'bounded','bernoulli','multinomial'}
    loss_kind : str
        Must be compatible with ci_family:
        - bounded:   {'abs','sq'}
        - bernoulli: {'tv','sq','kl'}
        - multinomial: {'tv','kl'}
    ci_kwargs : dict
        Family-specific parameters, e.g.:
        - bounded: {'method': 'hoeffding', 'bounds': (-1,1)}
    rng : seed or Generator

    Returns
    -------
    (d_plus, info)
    """
    ci_kwargs = ci_kwargs or {}
    fam = str(ci_family).lower()

    if fam == "bounded":
        return pseudo_delta_scalar_bounded(
            y_human=y_human, y_model=y_model, k=k, gamma=gamma, loss_kind=loss_kind,
            ci_method=ci_kwargs.get("method", "hoeffding"),
            bounds=ci_kwargs.get("bounds", (-1.0, 1.0)),
            n_target=n_target,
            rng=rng,
        )
    if fam == "bernoulli":
        return pseudo_delta_bernoulli(
            y_human=y_human, y_model=y_model, k=k, gamma=gamma, loss_kind=loss_kind,
            n_target=n_target,
            rng=rng,
        )
    if fam == "multinomial":
        return pseudo_delta_multinomial(
            y_human_counts=y_human, y_model_samples=y_model, k=k, gamma=gamma, loss_kind=loss_kind,
            n_target=n_target,
            rng=rng,
        )
    raise ValueError(f"Unknown ci_family={ci_family}")

# ----------------------------
# Lower pseudo-discrepancies Δ-
# ----------------------------

def lower_delta_bounded(
    y_human: Any,
    y_model: Any,
    k: int,
    gamma: float,
    loss_kind: str = "abs",
    ci_method: str = "hoeffding",
    bounds: Tuple[float, float] = (-1.0, 1.0),
    n_target: Optional[int] = None, 
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[float, Dict[str, Any]]:
    rng = _ensure_rng(rng)

    yh, n_used, n_avail = _subsample_human_1d(y_human, n_target=n_target, rng=rng)
    if n_used == 0:
        return np.nan, {"reason": "empty_human", "n_target": n_target, "n_used": 0, "n_avail": n_avail}

    m_h, (L, U), stats = ci_bounded_mean(yh, gamma=gamma, bounds=bounds, method=ci_method)
    qhat, k_used = _model_mean_from_samples(y_model, k=k, rng=rng)
    if not np.isfinite(qhat) or k_used <= 0:
        return np.nan, {"reason": "empty_model"}

    if L <= qhat <= U:
        d_minus = 0.0
    else:
        if loss_kind == "abs":
            d_minus = min(abs(L - qhat), abs(U - qhat))
        elif loss_kind in ("sq", "l2"):
            d_minus = min((L - qhat) ** 2, (U - qhat) ** 2)
        else:
            raise ValueError("bounded: loss_kind must be in {'abs','sq'}")

    info = {
        "family": "bounded",
        "loss_kind": loss_kind,
        "gamma": gamma,
        "delta": stats.get("delta"),
        "n_h": stats.get("n"),
        "n_target": n_target,
        "n_used": n_used,
        "n_avail": n_avail,
        "ci_interval": (L, U),
        "qhat": qhat,
        "k_used": k_used,
        "m_h": m_h,
    }
    return float(d_minus), info

def lower_delta_bernoulli(
    y_human: Any,
    y_model: Any,
    k: int,
    gamma: float,
    loss_kind: str = "kl",
    n_target: Optional[int] = None,   # NEW
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[float, Dict[str, Any]]:
    rng = _ensure_rng(rng)

    yh, n_used, n_avail = _subsample_human_1d(y_human, n_target=n_target, rng=rng)
    if n_used == 0:
        return np.nan, {"reason": "empty_human", "n_target": n_target, "n_used": 0, "n_avail": n_avail}

    phat = float(np.mean(yh))
    _, (pL, pU), ci_info = ci_bernoulli_kl(phat, n=int(yh.size), gamma=gamma)

    qhat, k_used = _model_mean_from_samples(y_model, k=k, rng=rng)
    if not np.isfinite(qhat) or k_used <= 0:
        return np.nan, {"reason": "empty_model"}

    if loss_kind == "tv":
        Lfun = loss_tv_bern
        d_minus = 0.0 if (pL <= qhat <= pU) else min(Lfun(pL, qhat), Lfun(pU, qhat))
    elif loss_kind in ("sq", "l2"):
        Lfun = loss_sq_mean_gap
        d_minus = 0.0 if (pL <= qhat <= pU) else min(Lfun(pL, qhat), Lfun(pU, qhat))
    elif loss_kind == "kl":
        Lfun = loss_kl_bern
        d_minus = 0.0 if (pL <= qhat <= pU) else min(Lfun(pL, qhat), Lfun(pU, qhat))
    else:
        raise ValueError("bernoulli: loss_kind must be in {'tv','sq','kl'}")

    info = {
        "family": "bernoulli",
        "loss_kind": loss_kind,
        "gamma": gamma,
        "delta": ci_info.get("delta"),
        "n_h": ci_info.get("n"),
        "n_target": n_target,
        "n_used": n_used,
        "n_avail": n_avail,
        "r": ci_info.get("r"),
        "phat": phat,
        "pL": pL,
        "pU": pU,
        "qhat": qhat,
        "k_used": k_used,
    }
    return float(d_minus), info

def lower_delta_multinomial(
    y_human_counts: Any,
    y_model_samples: Any,
    k: int,
    gamma: float,
    loss_kind: str = "tv",
    n_target: Optional[int] = None,   # NEW
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[float, Dict[str, Any]]:
    rng = _ensure_rng(rng)

    counts, n_used, n_avail, d_used = _human_to_multinomial_counts(
        y_human_full=y_human_counts, n_target=n_target, rng=rng, d=None
    )
    if counts.size == 0 or n_used == 0:
        return np.nan, {"reason": "empty_human", "n_target": n_target, "n_used": n_used, "n_avail": n_avail}

    phat, r, ci_info = ci_multinomial_klball(counts, gamma=gamma, n=None)
    if phat is None or not np.isfinite(r):
        return np.nan, {"reason": "empty_human"}

    d = int(phat.size)
    qhat, k_used_model = _model_prob_from_samples(y_model_samples, k=k, d=d, rng=rng)
    if k_used_model <= 0:
        return np.nan, {"reason": "empty_model"}

    tv_ph_q = 0.5 * float(np.sum(np.abs(phat - qhat)))
    tv_lb = max(tv_ph_q - math.sqrt(0.5 * r), 0.0)

    if loss_kind == "tv":
        d_minus = tv_lb
        extra = {}
    elif loss_kind == "kl":
        u_star, val, min_info = _multinomial_kl_inner_min_cvxpy(phat=phat, qhat=qhat, r=float(r))
        d_minus = float(val)
        extra = {"kl_inner_min_info": min_info}
    else:
        raise ValueError("multinomial: loss_kind must be in {'tv','kl'}")

    info = {
        "family": "multinomial",
        "loss_kind": loss_kind,
        "gamma": gamma,
        "delta": ci_info.get("delta"),
        "n_h": ci_info.get("n"),
        "n_target": n_target,
        "n_used": n_used,
        "n_avail": n_avail,
        "d_used": d_used,
        "d": ci_info.get("d"),
        "r": ci_info.get("r"),
        "k_used": k_used_model,
        "tv(phat,qhat)": tv_ph_q,
        "tv_lower": tv_lb,
        **extra,
    }
    return float(d_minus), info

def compute_pseudo_delta_lower(
    y_human: Any,
    y_model: Any,
    k: int = 200,
    gamma: float = 0.5,
    ci_family: str = "bounded",
    loss_kind: str = "abs",
    ci_kwargs: Optional[Dict[str, Any]] = None,
    n_target: Optional[int] = None,   # NEW
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[float, Dict[str, Any]]:
    ci_kwargs = ci_kwargs or {}
    fam = str(ci_family).lower()

    if fam == "bounded":
        return lower_delta_bounded(
            y_human=y_human, y_model=y_model, k=k, gamma=gamma, loss_kind=loss_kind,
            ci_method=ci_kwargs.get("method", "hoeffding"),
            bounds=ci_kwargs.get("bounds", (-1.0, 1.0)),
            n_target=n_target,
            rng=rng,
        )
    if fam == "bernoulli":
        return lower_delta_bernoulli(
            y_human=y_human, y_model=y_model, k=k, gamma=gamma, loss_kind=loss_kind,
            n_target=n_target,
            rng=rng,
        )
    if fam == "multinomial":
        return lower_delta_multinomial(
            y_human_counts=y_human, y_model_samples=y_model, k=k, gamma=gamma, loss_kind=loss_kind,
            n_target=n_target,
            rng=rng,
        )
    raise ValueError(f"Unknown ci_family={ci_family}")

# ============================================================
# (5) MODIFY: Pairwise pseudo-gaps to be n_target-aware
# Keep documentation, consistent with gamma-only API.
# Drop-in replacements for:
#   - pseudo_gap_scalar_bounded_pair
#   - pseudo_gap_bernoulli_pair
#   - pseudo_gap_multinomial_pair
# Plus a small dispatcher tweak: compute_pseudo_gap_pairwise now passes n_target.
# ============================================================

def pseudo_gap_scalar_bounded_pair(
    y_human: Any,
    y_model1: Any,
    y_model2: Any,
    k: int,
    gamma: float,
    loss_kind: str = "sq",
    ci_method: str = "hoeffding",
    bounds: Tuple[float, float] = (-1.0, 1.0),
    n_target: Optional[int] = None,   # NEW
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Pairwise pseudo-gap for bounded scalar outcomes.

    Definition
    ----------
    For scenario j:
        δ̂_j = sup_{u ∈ [L,U]} [ L(u, q̂1) - L(u, q̂2) ],
    where [L,U] is a (coverage=gamma) CI for the human mean, built from (subsampled) human data,
    and q̂1, q̂2 are model means estimated from k model samples (subsampled without replacement).

    Human subsampling
    -----------------
    If n_target is not None, we:
      1) drop missing values from y_human,
      2) subsample without replacement up to n_target.
    Let n_used be the realized sample size (<= n_target due to missingness).

    Loss cases
    ----------
    - 'sq': g(u) = (u-q1)^2 - (u-q2)^2 is linear in u -> maximum at an endpoint.
    - 'abs': g(u) = |u-q1| - |u-q2| piecewise linear -> maximum at endpoints and kinks.

    Returns
    -------
    (delta_hat, info)
      delta_hat : float
      info      : dict with CI diagnostics, (n_used, n_avail), and model k usage.
    """
    rng = _ensure_rng(rng)

    yh, n_used, n_avail = _subsample_human_1d(y_human, n_target=n_target, rng=rng)
    if n_used == 0:
        return np.nan, {
            "reason": "empty_human",
            "family": "bounded_pairwise",
            "loss_kind": loss_kind,
            "gamma": gamma,
            "n_target": n_target,
            "n_used": 0,
            "n_avail": n_avail,
        }

    m_h, (L, U), stats = ci_bounded_mean(yh, gamma=gamma, bounds=bounds, method=ci_method)

    q1, k1 = _model_mean_from_samples(y_model1, k=k, rng=rng)
    q2, k2 = _model_mean_from_samples(y_model2, k=k, rng=rng)
    if not np.isfinite(q1) or not np.isfinite(q2) or k1 <= 0 or k2 <= 0:
        return np.nan, {"reason": "empty_model", "family": "bounded_pairwise", "gamma": gamma}

    lk = str(loss_kind).lower()
    if lk in ("sq", "l2"):
        # (u-q1)^2 - (u-q2)^2 = 2(q2-q1)u + (q1^2 - q2^2)
        slope = 2.0 * (q2 - q1)
        intercept = q1 * q1 - q2 * q2
        u_star = U if slope >= 0 else L
        delta_hat = slope * u_star + intercept

    elif lk == "abs":
        def g(u: float) -> float:
            return abs(u - q1) - abs(u - q2)

        candidates = [L, U]
        for q in (q1, q2):
            if L <= q <= U:
                candidates.append(q)
        delta_hat = max(g(u) for u in candidates)

    else:
        raise ValueError("bounded pairwise: loss_kind must be in {'sq','abs'}")

    info = {
        "family": "bounded_pairwise",
        "loss_kind": lk,
        "gamma": gamma,
        "delta": stats.get("delta"),
        "n_h": stats.get("n"),
        "n_target": n_target,
        "n_used": n_used,
        "n_avail": n_avail,
        "ci_interval": (L, U),
        "q1_hat": q1,
        "q2_hat": q2,
        "k1_used": k1,
        "k2_used": k2,
        "m_h": m_h,
        "ci_method": stats.get("method"),
        "rad": stats.get("rad"),
        "bounds": bounds,
    }
    return float(delta_hat), info


def pseudo_gap_bernoulli_pair(
    y_human: Any,
    y_model1: Any,
    y_model2: Any,
    k: int,
    gamma: float,
    loss_kind: str = "tv",
    n_target: Optional[int] = None,   # NEW
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Pairwise pseudo-gap for Bernoulli outcomes.

    Definition
    ----------
    For scenario j:
        δ̂_j = sup_{u ∈ [pL,pU]} [ L(u, q̂1) - L(u, q̂2) ],
    where [pL,pU] is the Bernoulli KL-ball confidence interval at coverage gamma.

    Human subsampling
    -----------------
    If n_target is not None, we drop missing values from y_human and subsample up to n_target.
    This affects the CI endpoints [pL,pU] through phat and n.

    Loss cases
    ----------
    - 'tv': piecewise linear -> max at endpoints and kinks {q1,q2}∩[pL,pU]
    - 'sq': linear in u -> endpoint
    - 'kl': KL(u||q1)-KL(u||q2) linear in u -> endpoint

    Returns
    -------
    (delta_hat, info)
    """
    rng = _ensure_rng(rng)

    yh, n_used, n_avail = _subsample_human_1d(y_human, n_target=n_target, rng=rng)
    if n_used == 0:
        return np.nan, {
            "reason": "empty_human",
            "family": "bernoulli_pairwise",
            "loss_kind": loss_kind,
            "gamma": gamma,
            "n_target": n_target,
            "n_used": 0,
            "n_avail": n_avail,
        }

    phat = float(np.mean(yh))
    _, (pL, pU), ci_info = ci_bernoulli_kl(phat, n=int(yh.size), gamma=gamma)

    q1, k1 = _model_mean_from_samples(y_model1, k=k, rng=rng)
    q2, k2 = _model_mean_from_samples(y_model2, k=k, rng=rng)
    if not np.isfinite(q1) or not np.isfinite(q2) or k1 <= 0 or k2 <= 0:
        return np.nan, {"reason": "empty_model", "family": "bernoulli_pairwise", "gamma": gamma}

    lk = str(loss_kind).lower()
    if lk == "tv":
        def g(u: float) -> float:
            return abs(u - q1) - abs(u - q2)

        candidates = [pL, pU]
        for q in (q1, q2):
            if pL <= q <= pU:
                candidates.append(q)
        delta_hat = max(g(u) for u in candidates)

    elif lk in ("sq", "l2"):
        slope = 2.0 * (q2 - q1)
        intercept = q1 * q1 - q2 * q2
        u_star = pU if slope >= 0 else pL
        delta_hat = slope * u_star + intercept

    elif lk == "kl":
        # KL(u||q1) - KL(u||q2) = u*log(q2/q1) + (1-u)*log((1-q2)/(1-q1)) -> linear in u
        eps = 1e-12
        qq1 = float(np.clip(q1, eps, 1.0 - eps))
        qq2 = float(np.clip(q2, eps, 1.0 - eps))
        a = math.log(qq2 / qq1) - math.log((1.0 - qq2) / (1.0 - qq1))
        b = math.log((1.0 - qq2) / (1.0 - qq1))
        u_star = pU if a >= 0 else pL
        delta_hat = a * u_star + b

    else:
        raise ValueError("bernoulli pairwise: loss_kind must be in {'tv','sq','kl'}")

    info = {
        "family": "bernoulli_pairwise",
        "loss_kind": lk,
        "gamma": gamma,
        "delta": ci_info.get("delta"),
        "n_h": ci_info.get("n"),
        "n_target": n_target,
        "n_used": n_used,
        "n_avail": n_avail,
        "r": ci_info.get("r"),
        "phat": phat,
        "pL": pL,
        "pU": pU,
        "q1_hat": q1,
        "q2_hat": q2,
        "k1_used": k1,
        "k2_used": k2,
    }
    return float(delta_hat), info


def pseudo_gap_multinomial_pair(
    y_human_counts: Any,
    y_model1_samples: Any,
    y_model2_samples: Any,
    k: int,
    gamma: float,
    loss_kind: str = "kl",
    n_target: Optional[int] = None,   # NEW
    rng: Optional[Union[int, np.random.Generator]] = None,
    eps: float = 1e-12,
    solver: str = "ECOS",
    verbose: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """
    Pairwise pseudo-gap for multinomial outcomes with KL loss.

    Mathematical target
    -------------------
    Let phat be the empirical human distribution (dimension d), and let the confidence set be
        C = { u in simplex : KL(phat || u) <= r },
    where r is derived from (gamma, n, d) via ci_multinomial_klball.

    For loss_kind='kl' with L(u,q) = KL(u||q), we have:
        KL(u||q1) - KL(u||q2) = sum_i u_i log(q2_i / q1_i) = <u, c>,
    where c_i = log(q2_i / q1_i).

    Thus the pairwise pseudo-gap is:
        δ̂ = sup_{u in C} <u, c>.

    Human subsampling
    -----------------
    If y_human_counts is individual-level labels or 2D rows, and n_target is not None, we:
      - subsample individuals/rows up to n_target,
      - convert to counts,
      - then build the KL-ball CI around that phat.
    If y_human_counts is already a counts vector, n_target is ignored (cannot subsample aggregates).

    Parameters
    ----------
    y_human_counts : Any
        Human multinomial data: counts vector, label array, or 2D prob/one-hot rows.
    y_model1_samples, y_model2_samples : Any
        Model samples (labels or 2D probs/rows), as accepted by _model_prob_from_samples.
    k : int
        Model subsample size per simulator.
    gamma : float
        Coverage level in (0,1).
    loss_kind : str
        Only 'kl' implemented.
    n_target : Optional[int]
        Target per-scenario human subsample size before CI construction.
    rng : Optional[int or Generator]
    eps : float
        Floor for log ratios.
    solver, verbose : CVXPY solve options.

    Returns
    -------
    (delta_hat, info)
    """
    if str(loss_kind).lower() != "kl":
        raise ValueError("multinomial pairwise: currently only loss_kind='kl' is implemented.")

    rng = _ensure_rng(rng)

    counts, n_used, n_avail, d_used = _human_to_multinomial_counts(
        y_human_full=y_human_counts, n_target=n_target, rng=rng, d=None
    )
    if counts.size == 0 or n_used == 0:
        return np.nan, {
            "reason": "empty_human",
            "family": "multinomial_pairwise",
            "loss_kind": "kl",
            "gamma": gamma,
            "n_target": n_target,
            "n_used": n_used,
            "n_avail": n_avail,
        }

    # Human side: KL-ball around phat
    phat, r, ci_info = ci_multinomial_klball(counts, gamma=gamma, n=None, eps=eps)
    if phat is None or not np.isfinite(r):
        return np.nan, {
            "reason": "empty_human",
            "family": "multinomial_pairwise",
            "loss_kind": "kl",
            "gamma": gamma,
            "n_target": n_target,
            "n_used": n_used,
            "n_avail": n_avail,
        }

    d = int(phat.size)

    # Model side: two categorical probability vectors
    q1, k1 = _model_prob_from_samples(y_model1_samples, k=k, d=d, rng=rng)
    q2, k2 = _model_prob_from_samples(y_model2_samples, k=k, d=d, rng=rng)
    if k1 <= 0 or k2 <= 0 or q1 is None or q2 is None:
        return np.nan, {"reason": "empty_model", "family": "multinomial_pairwise", "loss_kind": "kl", "gamma": gamma}

    # Stabilize q1, q2 for log ratio
    q1 = np.clip(np.asarray(q1, float), eps, 1.0); q1 /= q1.sum()
    q2 = np.clip(np.asarray(q2, float), eps, 1.0); q2 /= q2.sum()
    c = np.log(q2) - np.log(q1)  # c_i = log(q2_i/q1_i)

    # Solve: max_u <u,c> s.t. u in simplex, KL(phat||u) <= r
    try:
        import cvxpy as cp
    except ImportError as e:
        raise ImportError("cvxpy is required for multinomial pairwise pseudo-gap with KL loss.") from e

    u = cp.Variable(d, nonneg=True)
    objective = cp.Maximize(c @ u)
    constraints = [
        cp.sum(u) == 1,
        cp.sum(cp.rel_entr(phat, u)) <= float(r),
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=verbose)

    status = prob.status
    if status not in ("optimal", "optimal_inaccurate"):
        return np.nan, {
            "reason": "solver_failed",
            "status": status,
            "family": "multinomial_pairwise",
            "loss_kind": "kl",
            "gamma": gamma,
            "delta": ci_info.get("delta"),
            "n_h": ci_info.get("n"),
            "n_target": n_target,
            "n_used": n_used,
            "n_avail": n_avail,
            "d_used": d_used,
            "d": ci_info.get("d"),
            "r": ci_info.get("r"),
            "k1_used": k1,
            "k2_used": k2,
            "solver": solver,
        }

    delta_hat = float(prob.value)

    info = {
        "family": "multinomial_pairwise",
        "loss_kind": "kl",
        "gamma": gamma,
        "delta": ci_info.get("delta"),
        "n_h": ci_info.get("n"),
        "n_target": n_target,
        "n_used": n_used,
        "n_avail": n_avail,
        "d_used": d_used,
        "d": ci_info.get("d"),
        "r": ci_info.get("r"),
        "k1_used": k1,
        "k2_used": k2,
        "solver": solver,
        "status": status,
    }
    return float(delta_hat), info


def compute_pseudo_gap_pairwise(
    y_human: Any,
    y_model1: Any,
    y_model2: Any,
    k: int = 200,
    gamma: Optional[float] = None,
    base_beta: float = 0.5,
    ci_family: str = "bounded",      # {"bounded","bernoulli","multinomial"}
    loss_kind: str = "sq",
    ci_kwargs: Optional[Dict[str, Any]] = None,
    n_target: Optional[int] = None,   # NEW
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Pairwise pseudo-gap δ (two simulators vs same human-side confidence set).

    Mathematical target
    -------------------
    For scenario j:
        δ̂_j = sup_{u ∈ C_j( p̂_j ; γ )} [ L(u, q̂1_j) - L(u, q̂2_j) ].

    This dispatcher is n_target-aware: the human-side CI C_j is built from
    a missingness-filtered subsample of size up to n_target (if provided).

    Parameters
    ----------
    n_target : Optional[int]
        Target per-scenario human subsample size used to build C_j. If None, uses all available.

    Other parameters: see your original docstring (unchanged in spirit).
    """
    ci_kwargs = ci_kwargs or {}

    gamma_eff = float(base_beta if gamma is None else gamma)

    fam = str(ci_family).lower()
    lk = str(loss_kind).lower()

    if fam == "bounded":
        return pseudo_gap_scalar_bounded_pair(
            y_human=y_human,
            y_model1=y_model1,
            y_model2=y_model2,
            k=int(k),
            gamma=gamma_eff,
            loss_kind=lk,
            ci_method=ci_kwargs.get("method", "hoeffding"),
            bounds=ci_kwargs.get("bounds", (-1.0, 1.0)),
            n_target=n_target,
            rng=rng,
        )

    if fam in ("bernoulli", "binomial"):
        return pseudo_gap_bernoulli_pair(
            y_human=y_human,
            y_model1=y_model1,
            y_model2=y_model2,
            k=int(k),
            gamma=gamma_eff,
            loss_kind=lk,
            n_target=n_target,
            rng=rng,
        )

    if fam == "multinomial":
        if lk != "kl":
            raise NotImplementedError("Pairwise multinomial pseudo-gap is implemented only for loss_kind='kl'.")
        return pseudo_gap_multinomial_pair(
            y_human_counts=y_human,
            y_model1_samples=y_model1,
            y_model2_samples=y_model2,
            k=int(k),
            gamma=gamma_eff,
            loss_kind="kl",
            n_target=n_target,
            rng=rng,
            eps=ci_kwargs.get("eps", 1e-12),
            solver=ci_kwargs.get("solver", "ECOS"),
            verbose=ci_kwargs.get("verbose", False),
        )

    raise ValueError(f"Unknown ci_family={ci_family} for pairwise pseudo-gap.")


# ============================================================
# Simulator selector Stage-1 labels (historical L-hat)
#   PDF: Simulator_Selector.pdf (Feb 2026)
#   Focus here: bounded scalar outcomes (e.g., WVS mapped to [-1, 1])
# ============================================================

def _hoeffding_rad(width: float, n: int, delta: float) -> float:
    """
    Hoeffding half-width for bounded variables in an interval of length `width`.
    """
    if n <= 0:
        return float("nan")
    d = float(np.clip(delta, 1e-12, 1.0 - 1e-12))
    return float(width * math.sqrt(math.log(2.0 / d) / (2.0 * float(n))))


def infer_set_bounded_mean_from_qhat(
    qhat: float,
    n: int,
    gamma: float,
    bounds: Tuple[float, float] = (-1.0, 1.0),
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Step-A set in Simulator_Selector Stage-1 for bounded scalar mean:
      C(qhat, n, gamma) = [qhat - rad, qhat + rad] clipped to bounds,
    where delta = 1 - gamma.
    """
    a0, b0 = float(bounds[0]), float(bounds[1])
    width = b0 - a0
    delta = _delta_from_gamma(float(gamma))
    if not np.isfinite(qhat) or n <= 0 or width <= 0:
        return np.nan, np.nan, {"n": int(n), "rad": np.nan, "gamma": float(gamma), "delta": float(delta), "bounds": bounds}

    rad = _hoeffding_rad(width=width, n=int(n), delta=float(delta))
    L = float(np.clip(float(qhat) - rad, a0, b0))
    U = float(np.clip(float(qhat) + rad, a0, b0))
    return L, U, {"n": int(n), "rad": float(rad), "gamma": float(gamma), "delta": float(delta), "bounds": bounds}


def forward_set_bounded_mean(
    q_latent: float,
    n: int,
    gamma: float,
    bounds: Tuple[float, float] = (-1.0, 1.0),
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Step-B set in Simulator_Selector Stage-1 for a fresh replicate mean around latent q:
      C_tilde(q, n, gamma) = [q - rad, q + rad] clipped to bounds,
    where delta = 1 - gamma.
    """
    return infer_set_bounded_mean_from_qhat(
        qhat=float(q_latent),
        n=int(n),
        gamma=float(gamma),
        bounds=bounds,
    )


def two_step_union_set_bounded_mean(
    qhat: float,
    n: int,
    gamma_infer: float,
    gamma_forward: float,
    bounds: Tuple[float, float] = (-1.0, 1.0),
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Union set in Simulator_Selector Stage-1:
      Cbar(qhat) = U_{q in C_infer(qhat)} C_forward(q),
    for bounded scalar means under Hoeffding radii.
    """
    a0, b0 = float(bounds[0]), float(bounds[1])
    width = b0 - a0
    if not np.isfinite(qhat) or n <= 0 or width <= 0:
        info = {
            "n": int(n),
            "gamma_infer": float(gamma_infer),
            "gamma_forward": float(gamma_forward),
            "rad_infer": np.nan,
            "rad_forward": np.nan,
            "bounds": bounds,
        }
        return np.nan, np.nan, info

    delta_infer = _delta_from_gamma(float(gamma_infer))
    delta_forward = _delta_from_gamma(float(gamma_forward))
    rad_i = _hoeffding_rad(width=width, n=int(n), delta=float(delta_infer))
    rad_f = _hoeffding_rad(width=width, n=int(n), delta=float(delta_forward))

    L = float(np.clip(float(qhat) - rad_i - rad_f, a0, b0))
    U = float(np.clip(float(qhat) + rad_i + rad_f, a0, b0))

    info = {
        "n": int(n),
        "gamma_infer": float(gamma_infer),
        "gamma_forward": float(gamma_forward),
        "delta_infer": float(delta_infer),
        "delta_forward": float(delta_forward),
        "rad_infer": float(rad_i),
        "rad_forward": float(rad_f),
        "bounds": bounds,
    }
    return L, U, info


def lhat_stage1_bounded_from_intervals(
    p_interval: Tuple[float, float],
    qbar_interval: Tuple[float, float],
    loss_kind: str = "sq",
) -> float:
    """
    Stage-1 label:
      Lhat = sup_{p in P_int} sup_{qtilde in Qbar_int} L(p, qtilde)
    for bounded scalar losses {'sq','abs'}.
    """
    pL, pU = float(p_interval[0]), float(p_interval[1])
    qL, qU = float(qbar_interval[0]), float(qbar_interval[1])

    if not (np.isfinite(pL) and np.isfinite(pU) and np.isfinite(qL) and np.isfinite(qU)):
        return float("nan")
    if pL > pU or qL > qU:
        return float("nan")

    lk = str(loss_kind).lower()
    # For both abs and sq, the supremum over rectangles is attained at corners.
    d_max = max(abs(pL - qL), abs(pL - qU), abs(pU - qL), abs(pU - qU))
    if lk in ("sq", "l2"):
        return float(d_max * d_max)
    if lk == "abs":
        return float(d_max)
    raise ValueError("loss_kind for bounded Stage-1 must be in {'sq','abs'}.")


def compute_lhat_stage1_bounded(
    y_human: Any,
    y_model: Any,
    k: int = 200,
    gamma_human: float = 0.9,              # coverage for p-interval
    gamma_sim_infer: float = 0.95,         # coverage for q latent infer step
    gamma_sim_forward: float = 0.95,       # coverage for fresh replicate step
    bounds: Tuple[float, float] = (-1.0, 1.0),
    ci_method_human: str = "hoeffding",
    n_target_human: Optional[int] = None,
    loss_kind: str = "sq",
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    One historical Stage-1 label Lhat_{j,s,k} for bounded outcomes.
    """
    rng = _ensure_rng(rng)

    yh, n_used, n_avail = _subsample_human_1d(y_human, n_target=n_target_human, rng=rng)
    if n_used <= 0:
        return np.nan, {"reason": "empty_human", "n_target_human": n_target_human, "n_used": 0, "n_avail": n_avail}

    _, (pL, pU), p_stats = ci_bounded_mean(
        y=yh,
        gamma=float(gamma_human),
        bounds=bounds,
        method=ci_method_human,
    )

    qhat, k_used = _model_mean_from_samples(y_model=y_model, k=int(k), rng=rng)
    if not np.isfinite(qhat) or k_used <= 0:
        return np.nan, {"reason": "empty_model"}

    qL, qU, q_info = two_step_union_set_bounded_mean(
        qhat=float(qhat),
        n=int(k_used),
        gamma_infer=float(gamma_sim_infer),
        gamma_forward=float(gamma_sim_forward),
        bounds=bounds,
    )

    lhat = lhat_stage1_bounded_from_intervals(
        p_interval=(pL, pU),
        qbar_interval=(qL, qU),
        loss_kind=loss_kind,
    )

    beta = 1.0 - float(gamma_human) * float(gamma_sim_infer) * float(gamma_sim_forward)
    info = {
        "family": "bounded_stage1_lhat",
        "loss_kind": str(loss_kind).lower(),
        "gamma_human": float(gamma_human),
        "gamma_sim_infer": float(gamma_sim_infer),
        "gamma_sim_forward": float(gamma_sim_forward),
        "beta_stage1": float(beta),
        "n_human_used": int(n_used),
        "n_human_avail": int(n_avail),
        "k_model_used": int(k_used),
        "p_interval": (float(pL), float(pU)),
        "qhat": float(qhat),
        "qbar_interval": (float(qL), float(qU)),
        "p_ci_stats": p_stats,
        "q_union_stats": q_info,
        "bounds": bounds,
    }
    return float(lhat), info


def build_historical_lhat_wvs(
    human_dict: Union[Dict[Any, Any], pd.DataFrame],
    simulators_dict: Dict[str, Union[Dict[Any, Any], pd.DataFrame]],
    *,
    k_values: Sequence[int] = (200,),
    gamma_human: float = 0.9,
    gamma_sim_infer: float = 0.95,
    gamma_sim_forward: float = 0.95,
    bounds: Tuple[float, float] = (-1.0, 1.0),
    ci_method_human: str = "hoeffding",
    n_target_human: Optional[int] = None,
    loss_kind: str = "sq",
    seed: int = 0,
    drop_q7_to_q17: bool = True,
) -> pd.DataFrame:
    """
    Construct historical Stage-1 labels Lhat_{j,s,k} for WVS-like bounded outcomes.

    Returns a long DataFrame with one row per (qid, simulator, k).
    """
    human = _as_dict_series(human_dict)
    sims = {name: _as_dict_series(sim) for name, sim in simulators_dict.items()}

    common = set(human.keys())
    for sim in sims.values():
        common &= set(sim.keys())

    qids = sorted(common)
    if drop_q7_to_q17:
        qids2 = []
        for qid in qids:
            m = re.match(r"(?i)^q\s*0*(\d+)$", str(qid).strip()) or re.match(r"^0*(\d+)$", str(qid).strip())
            if m and 7 <= int(m.group(1)) <= 17:
                continue
            qids2.append(qid)
        qids = qids2

    rng_master = np.random.default_rng(int(seed))
    rows = []

    for sim_name, sim_map in sims.items():
        for k in k_values:
            kk = int(k)
            for qid in qids:
                local_seed = int(rng_master.integers(0, 2**32 - 1))
                rng_local = np.random.default_rng(local_seed)

                y_h = human[qid]
                y_m = sim_map[qid]

                lhat, info = compute_lhat_stage1_bounded(
                    y_human=y_h,
                    y_model=y_m,
                    k=kk,
                    gamma_human=float(gamma_human),
                    gamma_sim_infer=float(gamma_sim_infer),
                    gamma_sim_forward=float(gamma_sim_forward),
                    bounds=bounds,
                    ci_method_human=ci_method_human,
                    n_target_human=n_target_human,
                    loss_kind=loss_kind,
                    rng=rng_local,
                )

                if not np.isfinite(lhat):
                    continue

                rows.append(
                    {
                        "qid": str(qid),
                        "sim": str(sim_name),
                        "k": int(kk),
                        "lhat": float(lhat),
                        "beta_stage1": float(info.get("beta_stage1", np.nan)),
                        "gamma_human": float(info.get("gamma_human", np.nan)),
                        "gamma_sim_infer": float(info.get("gamma_sim_infer", np.nan)),
                        "gamma_sim_forward": float(info.get("gamma_sim_forward", np.nan)),
                        "n_human_used": int(info.get("n_human_used", 0)),
                        "n_human_avail": int(info.get("n_human_avail", 0)),
                        "k_model_used": int(info.get("k_model_used", 0)),
                        "pL": float(info.get("p_interval", (np.nan, np.nan))[0]),
                        "pU": float(info.get("p_interval", (np.nan, np.nan))[1]),
                        "qhat": float(info.get("qhat", np.nan)),
                        "qbarL": float(info.get("qbar_interval", (np.nan, np.nan))[0]),
                        "qbarU": float(info.get("qbar_interval", (np.nan, np.nan))[1]),
                    }
                )

    cols = [
        "qid", "sim", "k", "lhat",
        "beta_stage1", "gamma_human", "gamma_sim_infer", "gamma_sim_forward",
        "n_human_used", "n_human_avail", "k_model_used",
        "pL", "pU", "qhat", "qbarL", "qbarU",
    ]
    return pd.DataFrame(rows, columns=cols)
