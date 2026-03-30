from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


ArrayLike = Union[np.ndarray, pd.Series, Sequence[float]]


MODEL_DISPLAY_NAMES = {
    "DecisionTree": "Decision Tree (DT)",
    "LinearModel": "Linear Regression",
    "MLP": "Multilayer Perceptron (MLP / ANN)",
}

MODEL_LITERATURE_FAMILIES = {
    "DecisionTree": "Decision-tree family",
    "LinearModel": "Linear-regression baseline",
    "MLP": "Artificial neural network family",
}

MODEL_PAPER_ALIGNMENT = {
    "DecisionTree": "Included as a deliberately shallow decision-tree baseline rather than a tuned high-performing tree model.",
    "LinearModel": "Plain linear regression on the causal feature set, used as the simplest interpretable baseline.",
    "MLP": "Used here as an ANN/MLP representative from the neural-network family discussed in the literature.",
}


def ensure_rng(rng: Optional[Union[int, np.random.Generator]] = None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def to_1d_float(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(getattr(x, "values", x), dtype=float).ravel()
    return arr[np.isfinite(arr)]


def empirical_quantile_curve(values: Sequence[float], alpha_grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.full_like(np.asarray(alpha_grid, dtype=float), np.nan, dtype=float)

    alpha_grid = np.asarray(alpha_grid, dtype=float)
    alpha_grid = np.clip(alpha_grid, 0.0, 1.0)
    s = np.sort(values)
    m = s.size
    idx = np.ceil(m * alpha_grid) - 1.0
    idx = np.clip(idx.astype(int), 0, m - 1)
    q = s[idx]
    return np.maximum.accumulate(q)


def literature_model_name(model_name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(str(model_name), str(model_name))


def literature_model_family(model_name: str) -> str:
    return MODEL_LITERATURE_FAMILIES.get(str(model_name), "Other")


def literature_model_note(model_name: str) -> str:
    return MODEL_PAPER_ALIGNMENT.get(str(model_name), "")


def add_model_annotations(df: pd.DataFrame, model_col: str = "model_name") -> pd.DataFrame:
    out = df.copy()
    out["model_display_name"] = out[model_col].map(literature_model_name)
    out["literature_family"] = out[model_col].map(literature_model_family)
    out["paper_alignment_note"] = out[model_col].map(literature_model_note)
    return out


def gamma_schedule_power(n_eff: int, beta: float = 1.0 / 3.0, eps: float = 1e-6) -> float:
    n_eff = int(n_eff)
    if n_eff <= 0:
        return np.nan
    gamma = 1.0 - n_eff ** (-float(beta))
    return float(np.clip(gamma, eps, 1.0 - eps))


def hoeffding_radius(width: float, n: int, gamma: float) -> float:
    if width <= 0 or n <= 0:
        return np.nan
    delta = float(np.clip(1.0 - float(gamma), 1e-12, 1.0 - 1e-12))
    return float(width * np.sqrt(np.log(2.0 / delta) / (2.0 * float(n))))


def bounded_mean_ci(
    y: ArrayLike,
    gamma: float,
    bounds: Tuple[float, float],
) -> Tuple[float, Tuple[float, float], Dict[str, float]]:
    y_arr = to_1d_float(y)
    n = int(y_arr.size)
    if n == 0:
        return np.nan, (np.nan, np.nan), {"n": 0, "gamma": float(gamma), "rad": np.nan}

    a0, b0 = float(bounds[0]), float(bounds[1])
    width = b0 - a0
    mean = float(np.mean(y_arr))
    rad = hoeffding_radius(width=width, n=n, gamma=float(gamma))
    lower = float(np.clip(mean - rad, a0, b0))
    upper = float(np.clip(mean + rad, a0, b0))
    return mean, (lower, upper), {"n": n, "gamma": float(gamma), "rad": float(rad)}


def variance_ci_bootstrap(
    y: ArrayLike,
    gamma: float,
    n_boot: int = 400,
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[float, Tuple[float, float], Dict[str, float]]:
    y_arr = to_1d_float(y)
    n = int(y_arr.size)
    if n == 0:
        return np.nan, (np.nan, np.nan), {"n": 0, "gamma": float(gamma), "n_boot": int(n_boot), "rad": np.nan}
    if n == 1:
        return 0.0, (0.0, 0.0), {"n": 1, "gamma": float(gamma), "n_boot": int(n_boot), "rad": 0.0}

    gamma = float(gamma)
    n_boot = int(max(50, n_boot))
    rng_obj = ensure_rng(rng)
    point = float(np.var(y_arr, ddof=1))

    boot_idx = rng_obj.integers(0, n, size=(n_boot, n))
    boot_samples = y_arr[boot_idx]
    boot_vars = np.var(boot_samples, axis=1, ddof=1)
    alpha = float(np.clip((1.0 - gamma) / 2.0, 1e-6, 0.5 - 1e-6))
    lower = float(np.quantile(boot_vars, alpha))
    upper = float(np.quantile(boot_vars, 1.0 - alpha))
    lower = max(0.0, lower)
    upper = max(lower, upper)
    rad = max(abs(point - lower), abs(upper - point))
    return point, (lower, upper), {"n": n, "gamma": gamma, "n_boot": n_boot, "rad": float(rad)}


def pseudo_delta_upper_bounded(
    y_real: ArrayLike,
    y_sim: ArrayLike,
    gamma: float,
    bounds: Tuple[float, float],
    loss_kind: str = "sq",
) -> Tuple[float, Dict[str, float]]:
    mean_real, (lower, upper), info = bounded_mean_ci(y_real, gamma=gamma, bounds=bounds)
    y_sim_arr = to_1d_float(y_sim)
    if y_sim_arr.size == 0:
        return np.nan, {"reason": "empty_sim"}

    qhat = float(np.mean(y_sim_arr))
    if loss_kind == "sq":
        delta_plus = max((lower - qhat) ** 2, (upper - qhat) ** 2)
    elif loss_kind == "abs":
        delta_plus = max(abs(lower - qhat), abs(upper - qhat))
    else:
        raise ValueError("loss_kind must be 'sq' or 'abs'.")

    return float(delta_plus), {
        "n_real": int(info["n"]),
        "k_sim": int(y_sim_arr.size),
        "gamma": float(gamma),
        "rad": float(info["rad"]),
        "p_hat": float(mean_real),
        "q_hat": float(qhat),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }


def pseudo_delta_upper_variance(
    y_real: ArrayLike,
    y_sim: ArrayLike,
    gamma: float,
    loss_kind: str = "sq",
    n_boot: int = 400,
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[float, Dict[str, float]]:
    var_real, (lower, upper), info = variance_ci_bootstrap(y_real, gamma=gamma, n_boot=n_boot, rng=rng)
    y_sim_arr = to_1d_float(y_sim)
    if y_sim_arr.size < 2:
        return np.nan, {"reason": "insufficient_sim"}

    qhat = float(np.var(y_sim_arr, ddof=1))
    if loss_kind == "sq":
        delta_plus = max((lower - qhat) ** 2, (upper - qhat) ** 2)
    elif loss_kind == "abs":
        delta_plus = max(abs(lower - qhat), abs(upper - qhat))
    else:
        raise ValueError("loss_kind must be 'sq' or 'abs'.")

    return float(delta_plus), {
        "n_real": int(info["n"]),
        "k_sim": int(y_sim_arr.size),
        "gamma": float(gamma),
        "rad": float(info["rad"]),
        "n_boot": int(info["n_boot"]),
        "p_hat": float(var_real),
        "q_hat": float(qhat),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }


def pseudo_delta_lower_variance(
    y_real: ArrayLike,
    y_sim: ArrayLike,
    gamma: float,
    loss_kind: str = "sq",
    n_boot: int = 400,
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[float, Dict[str, float]]:
    var_real, (lower, upper), info = variance_ci_bootstrap(y_real, gamma=gamma, n_boot=n_boot, rng=rng)
    y_sim_arr = to_1d_float(y_sim)
    if y_sim_arr.size < 2:
        return np.nan, {"reason": "insufficient_sim"}

    qhat = float(np.var(y_sim_arr, ddof=1))
    if lower <= qhat <= upper:
        delta_minus = 0.0
    elif loss_kind == "sq":
        delta_minus = min((lower - qhat) ** 2, (upper - qhat) ** 2)
    elif loss_kind == "abs":
        delta_minus = min(abs(lower - qhat), abs(upper - qhat))
    else:
        raise ValueError("loss_kind must be 'sq' or 'abs'.")

    return float(delta_minus), {
        "n_real": int(info["n"]),
        "k_sim": int(y_sim_arr.size),
        "gamma": float(gamma),
        "rad": float(info["rad"]),
        "n_boot": int(info["n_boot"]),
        "p_hat": float(var_real),
        "q_hat": float(qhat),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }


def pseudo_delta_lower_bounded(
    y_real: ArrayLike,
    y_sim: ArrayLike,
    gamma: float,
    bounds: Tuple[float, float],
    loss_kind: str = "sq",
) -> Tuple[float, Dict[str, float]]:
    mean_real, (lower, upper), info = bounded_mean_ci(y_real, gamma=gamma, bounds=bounds)
    y_sim_arr = to_1d_float(y_sim)
    if y_sim_arr.size == 0:
        return np.nan, {"reason": "empty_sim"}

    qhat = float(np.mean(y_sim_arr))
    if lower <= qhat <= upper:
        delta_minus = 0.0
    elif loss_kind == "sq":
        delta_minus = min((lower - qhat) ** 2, (upper - qhat) ** 2)
    elif loss_kind == "abs":
        delta_minus = min(abs(lower - qhat), abs(upper - qhat))
    else:
        raise ValueError("loss_kind must be 'sq' or 'abs'.")

    return float(delta_minus), {
        "n_real": int(info["n"]),
        "k_sim": int(y_sim_arr.size),
        "gamma": float(gamma),
        "rad": float(info["rad"]),
        "p_hat": float(mean_real),
        "q_hat": float(qhat),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }


@dataclass
class ResidualBootstrapEmulator:
    name: str
    estimator: object
    sample_mode: str = "bootstrap"

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: ArrayLike) -> "ResidualBootstrapEmulator":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()
        self.estimator.fit(X_arr, y_arr)
        train_pred = np.asarray(self.estimator.predict(X_arr), dtype=float).ravel()
        residuals = y_arr - train_pred
        residuals = residuals[np.isfinite(residuals)]
        if residuals.size == 0:
            residuals = np.array([0.0], dtype=float)
        self.residuals_ = residuals - residuals.mean()
        self.resid_std_ = float(np.std(self.residuals_, ddof=1)) if residuals.size > 1 else 0.0
        self.feature_count_ = X_arr.shape[1]
        return self

    def predict_mean(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        return np.asarray(self.estimator.predict(X_arr), dtype=float).ravel()

    def sample_y(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        rng: Optional[Union[int, np.random.Generator]] = None,
    ) -> np.ndarray:
        rng = ensure_rng(rng)
        mean = self.predict_mean(X)
        if self.sample_mode == "gaussian":
            noise = rng.normal(loc=0.0, scale=self.resid_std_, size=mean.shape[0])
        else:
            noise = rng.choice(self.residuals_, size=mean.shape[0], replace=True)
        return mean + noise


def make_default_emulators(random_state: int = 0) -> Dict[str, ResidualBootstrapEmulator]:
    return {
        "DecisionTree": ResidualBootstrapEmulator(
            name="DecisionTree",
            estimator=DecisionTreeRegressor(
                max_depth=3,
                min_samples_leaf=120,
                random_state=random_state,
            ),
            sample_mode="bootstrap",
        ),
        "LinearModel": ResidualBootstrapEmulator(
            name="LinearModel",
            estimator=Pipeline(
                steps=[
                    ("model", LinearRegression()),
                ]
            ),
            sample_mode="gaussian",
        ),
        "MLP": ResidualBootstrapEmulator(
            name="MLP",
            estimator=Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    (
                        "model",
                        MLPRegressor(
                            hidden_layer_sizes=(24, 12),
                            activation="relu",
                            alpha=1e-3,
                            learning_rate_init=1e-3,
                            early_stopping=True,
                            validation_fraction=0.1,
                            n_iter_no_change=5,
                            max_iter=25,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            sample_mode="bootstrap",
        ),
    }


def summarize_series(values: Iterable[float], quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9)) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}

    summary = {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }
    for q in quantiles:
        summary[f"q{int(round(100 * q))}"] = float(np.quantile(arr, q))
    return summary


def rmse_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true_arr = np.asarray(y_true, dtype=float).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=float).ravel()
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def w1_confidence_radius_subgaussian(n: int, gamma: float, sigma: float) -> float:
    n = int(n)
    gamma = float(gamma)
    sigma = float(sigma)
    if n <= 0 or sigma <= 0:
        return np.nan
    miss = float(np.clip(1.0 - gamma, 1e-12, 1.0 - 1e-12))
    term1 = 512.0 * sigma / math.sqrt(float(n))
    term2 = sigma * math.sqrt((256.0 * math.e / float(n)) * math.log(1.0 / miss))
    return float(term1 + term2)


def w1_pseudo_discrepancy_ball(
    y_real: ArrayLike,
    y_sim: ArrayLike,
    gamma: float,
    sigma: float,
) -> Tuple[float, float, Dict[str, float]]:
    y_real_arr = to_1d_float(y_real)
    y_sim_arr = to_1d_float(y_sim)
    if y_real_arr.size == 0 or y_sim_arr.size == 0:
        return np.nan, np.nan, {"reason": "empty_sample"}

    w1_emp = wasserstein_1d_empirical(y_real_arr, y_sim_arr)
    radius = w1_confidence_radius_subgaussian(n=int(y_real_arr.size), gamma=gamma, sigma=sigma)
    delta_plus = w1_emp + radius
    delta_minus = max(0.0, w1_emp - radius)
    return float(delta_plus), float(delta_minus), {
        "n_real": int(y_real_arr.size),
        "k_sim": int(y_sim_arr.size),
        "gamma": float(gamma),
        "sigma": float(sigma),
        "radius": float(radius),
        "w1_emp": float(w1_emp),
    }


def affine_scale_to_minus1_plus1(x: ArrayLike, bounds: Tuple[float, float]) -> np.ndarray:
    arr = np.asarray(getattr(x, "values", x), dtype=float)
    a0, b0 = float(bounds[0]), float(bounds[1])
    width = b0 - a0
    if width <= 0:
        raise ValueError("bounds must have positive width.")
    return 2.0 * (arr - a0) / width - 1.0


def build_asymptotic_quantile_curves(
    delta_plus: ArrayLike,
    delta_minus: ArrayLike,
    gamma_values: ArrayLike,
    tau_grid: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    tau_grid = np.asarray(tau_grid, dtype=float)
    gamma_arr = np.asarray(gamma_values, dtype=float)
    gamma_arr = gamma_arr[np.isfinite(gamma_arr)]
    if gamma_arr.size == 0:
        raise ValueError("gamma_values must contain at least one finite value.")

    bar_gamma = float(np.mean(gamma_arr))
    alpha_low = np.clip(bar_gamma * tau_grid, 0.0, 1.0)
    alpha_up = np.clip((1.0 - bar_gamma) + bar_gamma * tau_grid, 0.0, 1.0)
    alpha_emp = np.clip(tau_grid, 0.0, 1.0)

    v_minus = empirical_quantile_curve(delta_minus, alpha_low)
    v_plus = empirical_quantile_curve(delta_plus, alpha_up)
    v_emp_plus = empirical_quantile_curve(delta_plus, alpha_emp)

    curve_df = pd.DataFrame(
        {
            "tau": tau_grid,
            "alpha_low": alpha_low,
            "alpha_up": alpha_up,
            "alpha_emp": alpha_emp,
            "v_minus": v_minus,
            "v_plus": v_plus,
            "v_emp_plus": v_emp_plus,
            "bar_gamma": bar_gamma,
        }
    )
    return curve_df, {"bar_gamma": bar_gamma}


def wasserstein_1d_empirical(
    x: ArrayLike,
    y: ArrayLike,
    n_grid: int = 512,
) -> float:
    """
    Approximate 1D Wasserstein-1 distance by integrating the gap between
    empirical quantile functions on a dense uniform grid.

    This is intended for lightweight scenario-level distribution comparison
    when an exact transport routine is not required.
    """
    x_arr = to_1d_float(x)
    y_arr = to_1d_float(y)
    if x_arr.size == 0 or y_arr.size == 0:
        return np.nan

    u = np.linspace(0.0, 1.0, max(16, int(n_grid)))
    qx = np.quantile(x_arr, u)
    qy = np.quantile(y_arr, u)
    return float(np.trapz(np.abs(qx - qy), u))
