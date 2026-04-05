from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def coerce_finite_1d(x: Any) -> np.ndarray:
    arr = np.asarray(getattr(x, "values", x), dtype=float).ravel()
    return arr[np.isfinite(arr)]


def subsample_human_empirical_distributions(
    actual_dict: Dict[str, Any],
    qids: Sequence[str],
    *,
    n_target: Optional[int],
    seed: int,
    replace: bool = False,
    gamma_mode: str = "adaptive_power",
    gamma_value: float = 0.5,
    beta: float = 1.0 / 3.0,
    gamma_clip_eps: float = 1e-6,
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    rng_master = np.random.default_rng(seed)
    human_samples: Dict[str, np.ndarray] = {}
    rows = []

    for qid in qids:
        local_seed = int(rng_master.integers(0, 2**32 - 1))
        rng_local = np.random.default_rng(local_seed)
        values = coerce_finite_1d(actual_dict[qid])
        n_available = int(values.size)
        if n_available <= 0:
            continue

        if n_target is None or int(n_target) <= 0:
            sample = values
        else:
            n_used = int(min(int(n_target), n_available))
            if (not replace) and (n_used == n_available):
                sample = values
            else:
                idx = rng_local.choice(n_available, size=n_used, replace=bool(replace))
                sample = values[idx]

        n_eff = int(sample.size)
        if n_eff <= 0:
            continue

        gamma_j = resolve_gamma(
            n_eff,
            mode=gamma_mode,
            value=gamma_value,
            beta=beta,
            clip_eps=gamma_clip_eps,
        )
        delta_j = miscoverage_from_gamma(gamma_j)

        human_samples[str(qid)] = sample.astype(float, copy=False)
        rows.append(
            {
                "qid": str(qid),
                "n_eff": int(n_eff),
                "n_available": int(n_available),
                "human_seed": int(local_seed),
                "gamma_mode": str(gamma_mode),
                "gamma_j": float(gamma_j),
                "delta_j": float(delta_j),
                "beta": float(beta) if str(gamma_mode) == "adaptive_power" else np.nan,
            }
        )

    return human_samples, pd.DataFrame(rows)


def resolve_gamma(
    n_eff: int,
    *,
    mode: str = "adaptive_power",
    value: float = 0.5,
    beta: float = 1.0 / 3.0,
    clip_eps: float = 1e-6,
) -> float:
    n_eff = int(n_eff)
    if n_eff <= 0:
        raise ValueError("n_eff must be positive.")

    if mode == "fixed":
        gamma = float(value)
    elif mode == "adaptive_power":
        if float(beta) <= 0.0:
            raise ValueError("beta must be positive for adaptive_power gamma.")
        gamma = 1.0 - (n_eff ** (-float(beta)))
    else:
        raise ValueError(f"Unknown gamma mode: {mode}")

    return float(np.clip(gamma, clip_eps, 1.0 - clip_eps))


def miscoverage_from_gamma(gamma: float, eps: float = 1e-12) -> float:
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 0.0 or gamma >= 1.0:
        raise ValueError("gamma must lie in (0, 1).")
    return float(np.clip(1.0 - gamma, eps, 1.0 - eps))


def support_probabilities(values: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = coerce_finite_1d(values)
    if arr.size == 0:
        return np.empty((0,), dtype=float), np.empty((0,), dtype=float), np.empty((0,), dtype=int)
    support, counts = np.unique(arr, return_counts=True)
    probs = counts.astype(float) / float(counts.sum())
    return support.astype(float), probs.astype(float), counts.astype(int)


def positive_support_gap(values_x: np.ndarray, values_y: np.ndarray, fallback: float = 1.0) -> float:
    combined = np.unique(np.concatenate([np.asarray(values_x, dtype=float), np.asarray(values_y, dtype=float)]))
    if combined.size <= 1:
        return float(fallback)
    gaps = np.diff(np.sort(combined))
    gaps = gaps[np.isfinite(gaps) & (gaps > 0.0)]
    if gaps.size == 0:
        return float(fallback)
    return float(np.median(gaps))


def resolve_bandwidth(
    support_x: np.ndarray,
    support_y: np.ndarray,
    *,
    bandwidth: Optional[float] = None,
    bandwidth_mode: str = "median_gap",
    fallback: float = 1.0,
) -> float:
    if bandwidth is not None:
        bandwidth = float(bandwidth)
        if bandwidth <= 0.0:
            raise ValueError("bandwidth must be positive.")
        return bandwidth

    if bandwidth_mode == "median_gap":
        return positive_support_gap(support_x, support_y, fallback=fallback)
    if bandwidth_mode == "unit":
        return 1.0
    raise ValueError(f"Unknown bandwidth mode: {bandwidth_mode}")


def kernel_matrix(
    x: np.ndarray,
    y: np.ndarray,
    *,
    kernel: str,
    bandwidth: Optional[float] = None,
    bandwidth_mode: str = "median_gap",
    fallback_bandwidth: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if kernel == "delta":
        return (x[:, None] == y[None, :]).astype(float), {"bandwidth": np.nan}

    bw = resolve_bandwidth(
        x,
        y,
        bandwidth=bandwidth,
        bandwidth_mode=bandwidth_mode,
        fallback=fallback_bandwidth,
    )
    scaled_abs = np.abs(x[:, None] - y[None, :]) / max(bw, 1e-12)

    if kernel in {"rbf", "gaussian"}:
        return np.exp(-0.5 * (scaled_abs**2)), {"bandwidth": float(bw)}
    if kernel == "laplace":
        return np.exp(-scaled_abs), {"bandwidth": float(bw)}
    raise ValueError(f"Unknown kernel: {kernel}")


def empirical_mmd2_from_supports(
    support_p: np.ndarray,
    probs_p: np.ndarray,
    support_q: np.ndarray,
    probs_q: np.ndarray,
    *,
    kernel: str,
    bandwidth: Optional[float] = None,
    bandwidth_mode: str = "median_gap",
    fallback_bandwidth: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    if support_p.size == 0 or support_q.size == 0:
        return np.nan, {"bandwidth": np.nan}

    shared_bandwidth = bandwidth
    if kernel != "delta":
        shared_bandwidth = resolve_bandwidth(
            support_p,
            support_q,
            bandwidth=bandwidth,
            bandwidth_mode=bandwidth_mode,
            fallback=fallback_bandwidth,
        )

    k_pp, meta_pp = kernel_matrix(
        support_p,
        support_p,
        kernel=kernel,
        bandwidth=shared_bandwidth,
        bandwidth_mode=bandwidth_mode,
        fallback_bandwidth=fallback_bandwidth,
    )
    k_qq, _ = kernel_matrix(
        support_q,
        support_q,
        kernel=kernel,
        bandwidth=shared_bandwidth,
        bandwidth_mode=bandwidth_mode,
        fallback_bandwidth=fallback_bandwidth,
    )
    k_pq, _ = kernel_matrix(
        support_p,
        support_q,
        kernel=kernel,
        bandwidth=shared_bandwidth,
        bandwidth_mode=bandwidth_mode,
        fallback_bandwidth=fallback_bandwidth,
    )

    p = np.asarray(probs_p, dtype=float).ravel()
    q = np.asarray(probs_q, dtype=float).ravel()

    mmd2 = float(p @ k_pp @ p + q @ k_qq @ q - 2.0 * (p @ k_pq @ q))
    return max(mmd2, 0.0), meta_pp


def bounded_kernel_supremum(kernel: str) -> float:
    if kernel in {"delta", "rbf", "gaussian", "laplace"}:
        return 1.0
    raise ValueError(f"No bounded-kernel supremum registered for kernel: {kernel}")


def mmd_confidence_radius(
    n_eff: int,
    *,
    kernel: str,
    gamma_j: float,
    safe: bool = False,
) -> float:
    """One-sample RKHS mean-embedding radius for bounded kernels.

    This returns the computable radius used to define
    {U : MMD_k(U, \hat P_j) <= r_j}. The main branch is the standard
    bounded-kernel one-sample concentration form; `safe=True` keeps the
    more conservative variant used only as an optional sensitivity check.
    """
    n_eff = int(n_eff)
    if n_eff <= 0:
        return np.nan
    k_diag = bounded_kernel_supremum(kernel)
    delta_j = miscoverage_from_gamma(gamma_j)
    if safe:
        return float(
            2.0 * math.sqrt(k_diag / n_eff)
            + math.sqrt(2.0 * k_diag * math.log(2.0 / delta_j) / n_eff)
        )
    return float(
        math.sqrt(k_diag / n_eff)
        + math.sqrt(2.0 * k_diag * math.log(1.0 / delta_j) / n_eff)
    )


def kernel_label(spec: Dict[str, Any]) -> str:
    kernel = str(spec["kernel"])
    if kernel == "delta":
        return "Delta"
    if kernel in {"rbf", "gaussian"}:
        return "RBF"
    if kernel == "laplace":
        return "Laplace"
    return kernel


def kernel_encoding_note(spec: Dict[str, Any]) -> str:
    kernel = str(spec["kernel"])
    if kernel == "delta":
        return "Equality on the repo's discrete answer codes."
    if kernel in {"rbf", "gaussian", "laplace"}:
        return "Numeric answer codes from the repo's existing [-1, 1] question-specific encoding."
    return "Kernel-specific encoding."


def compute_mmd_benchmark(
    human_samples: Dict[str, np.ndarray],
    simulator_dict: Dict[str, Any],
    human_meta_df: pd.DataFrame,
    *,
    kernel_specs: Sequence[Dict[str, Any]],
    simulator_order: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if human_meta_df.empty:
        return pd.DataFrame()

    human_meta = human_meta_df.set_index("qid").copy()
    ordered_sims = list(simulator_order) if simulator_order is not None else list(simulator_dict.keys())
    rows = []

    for sim_name in ordered_sims:
        sim_map = simulator_dict[sim_name]
        for qid, p_vals in human_samples.items():
            q_vals = coerce_finite_1d(sim_map[qid])
            if q_vals.size == 0:
                continue

            support_p, probs_p, counts_p = support_probabilities(p_vals)
            support_q, probs_q, counts_q = support_probabilities(q_vals)
            if support_p.size == 0 or support_q.size == 0:
                continue

            gamma_j = float(human_meta.loc[qid, "gamma_j"])
            delta_j = float(human_meta.loc[qid, "delta_j"])
            n_eff = int(human_meta.loc[qid, "n_eff"])

            for spec in kernel_specs:
                kernel = str(spec["kernel"])
                bandwidth = spec.get("bandwidth")
                bandwidth_mode = str(spec.get("bandwidth_mode", "median_gap"))
                fallback_bandwidth = float(spec.get("fallback_bandwidth", 1.0))
                mmd2, kernel_meta = empirical_mmd2_from_supports(
                    support_p,
                    probs_p,
                    support_q,
                    probs_q,
                    kernel=kernel,
                    bandwidth=bandwidth,
                    bandwidth_mode=bandwidth_mode,
                    fallback_bandwidth=fallback_bandwidth,
                )
                raw_mmd = float(math.sqrt(max(mmd2, 0.0)))
                radius = mmd_confidence_radius(n_eff, kernel=kernel, gamma_j=gamma_j, safe=False)
                radius_safe = mmd_confidence_radius(n_eff, kernel=kernel, gamma_j=gamma_j, safe=True)
                k_diag = bounded_kernel_supremum(kernel)
                bw_used = kernel_meta.get("bandwidth", np.nan)

                rows.append(
                    {
                        "qid": str(qid),
                        "sim": str(sim_name),
                        "kernel": kernel,
                        "kernel_label": kernel_label(spec),
                        "encoding_note": kernel_encoding_note(spec),
                        "bandwidth_mode": bandwidth_mode if kernel != "delta" else "not_used",
                        "bandwidth_used": float(bw_used) if np.isfinite(bw_used) else np.nan,
                        "K_k": float(k_diag),
                        "n_eff": int(n_eff),
                        "n_model": int(q_vals.size),
                        "gamma_j": float(gamma_j),
                        "delta_j": float(delta_j),
                        "raw_mmd": float(raw_mmd),
                        "raw_mmd_sq": float(mmd2),
                        "radius": float(radius),
                        "radius_safe": float(radius_safe),
                        "proxy_mmd": float(raw_mmd + radius),
                        "proxy_mmd_sq": float((raw_mmd + radius) ** 2),
                        "proxy_mmd_safe": float(raw_mmd + radius_safe),
                        "proxy_mmd_safe_sq": float((raw_mmd + radius_safe) ** 2),
                        "human_support_size": int(support_p.size),
                        "model_support_size": int(support_q.size),
                        "human_counts_json": counts_p.tolist(),
                        "human_support_json": support_p.tolist(),
                        "model_counts_json": counts_q.tolist(),
                        "model_support_json": support_q.tolist(),
                    }
                )

    return pd.DataFrame(rows)


def summarize_mmd_curves(curve_df: pd.DataFrame) -> pd.DataFrame:
    if curve_df.empty:
        return pd.DataFrame()

    rows = []
    for (kernel_label_value, curve_metric, sim_name), sub in curve_df.groupby(
        ["kernel_label", "curve_metric", "sim"], dropna=False
    ):
        sub = sub.sort_values("quantile_level")
        alpha = sub["quantile_level"].to_numpy(dtype=float)
        values = sub["curve_value"].to_numpy(dtype=float)
        tail_mask = alpha >= 0.90
        rows.append(
            {
                "kernel_label": str(kernel_label_value),
                "curve_metric": str(curve_metric),
                "sim": str(sim_name),
                "auc": float(np.trapz(values, alpha)),
                "tail_mean_alpha_ge_0_90": (
                    float(np.trapz(values[tail_mask], alpha[tail_mask]) / max(alpha[tail_mask][-1] - alpha[tail_mask][0], 1e-12))
                    if tail_mask.sum() >= 2
                    else np.nan
                ),
                "q_at_tau_0_90": float(np.interp(0.90, alpha, values)),
            }
        )
    return pd.DataFrame(rows)
