from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t


def _two_sample_t_pvalue(y: np.ndarray, z: np.ndarray) -> float:
    treated = y[z == 1]
    control = y[z == 0]
    n1 = treated.shape[0]
    n0 = control.shape[0]
    s1 = float(np.var(treated, ddof=1))
    s0 = float(np.var(control, ddof=1))
    se = float(np.sqrt(s1 / n1 + s0 / n0))
    diff = float(np.mean(treated) - np.mean(control))
    if se == 0.0:
        return 1.0
    t_stat = diff / se
    df_num = (s1 / n1 + s0 / n0) ** 2
    df_den = ((s1 / n1) ** 2) / (n1 - 1) + ((s0 / n0) ** 2) / (n0 - 1)
    if df_den == 0.0:
        return 1.0
    df = df_num / df_den
    return float(2.0 * t.sf(np.abs(t_stat), df=df))


def simulate_null_pvalues(config: dict[str, Any]) -> pd.DataFrame:
    """
    Generate p-values under the complete null for L simulations.
    Return columns: sim_id, hypothesis_id, p_value.
    """
    rng = np.random.default_rng(int(config["seed_null"]))
    n = int(config["N"])
    m = int(config["M"])
    l = int(config["L"])
    p_treat = float(config["p_treat"])

    rows: list[dict[str, float | int]] = []
    for sim_id in range(l):
        z = (rng.random(n) < p_treat).astype(int)
        for hypothesis_id in range(m):
            y = rng.normal(loc=0.0, scale=1.0, size=n)
            p_value = _two_sample_t_pvalue(y=y, z=z)
            rows.append(
                {
                    "sim_id": sim_id,
                    "hypothesis_id": hypothesis_id,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def simulate_mixed_pvalues(config: dict[str, Any]) -> pd.DataFrame:
    """
    Generate p-values under mixed true and false null hypotheses for L simulations.
    Return columns: sim_id, hypothesis_id, p_value, is_true_null.
    """
    rng = np.random.default_rng(int(config["seed_mixed"]))
    n = int(config["N"])
    m = int(config["M"])
    m0 = int(config["M0"])
    l = int(config["L"])
    p_treat = float(config["p_treat"])
    tau_alt = float(config["tau_alternative"])

    rows: list[dict[str, float | int | bool]] = []
    for sim_id in range(l):
        z = (rng.random(n) < p_treat).astype(int)
        for hypothesis_id in range(m):
            is_true_null = hypothesis_id >= (m - m0)
            effect = 0.0 if is_true_null else tau_alt
            y = rng.normal(loc=0.0, scale=1.0, size=n) + effect * z
            p_value = _two_sample_t_pvalue(y=y, z=z)
            rows.append(
                {
                    "sim_id": sim_id,
                    "hypothesis_id": hypothesis_id,
                    "p_value": p_value,
                    "is_true_null": is_true_null,
                }
            )
    return pd.DataFrame(rows)


def _as_1d_array(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).reshape(-1)


def _sorted_pvalue_order(p_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p_values_1d = np.asarray(p_values, dtype=float).reshape(-1)
    order = np.argsort(p_values_1d, kind="mergesort")
    return p_values_1d, order


def bonferroni_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Bonferroni correction.
    """
    p_values_1d = np.asarray(p_values, dtype=float).reshape(-1)
    threshold = float(alpha) / p_values_1d.size
    return p_values_1d <= threshold


def holm_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Holm step-down correction.
    """
    p_values_1d, order = _sorted_pvalue_order(p_values)
    m = p_values_1d.size
    sorted_p = p_values_1d[order]
    sorted_rejections = np.zeros(m, dtype=bool)

    for rank, p_value in enumerate(sorted_p, start=1):
        threshold = float(alpha) / (m - rank + 1)
        if p_value <= threshold:
            sorted_rejections[rank - 1] = True
        else:
            break

    rejections = np.zeros(m, dtype=bool)
    rejections[order] = sorted_rejections
    return rejections


def benjamini_hochberg_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Benjamini-Hochberg correction.
    """
    p_values_1d, order = _sorted_pvalue_order(p_values)
    m = p_values_1d.size
    sorted_p = p_values_1d[order]
    thresholds = float(alpha) * np.arange(1, m + 1) / m
    passing = np.flatnonzero(sorted_p <= thresholds)
    sorted_rejections = np.zeros(m, dtype=bool)
    if passing.size > 0:
        sorted_rejections[: passing[-1] + 1] = True

    rejections = np.zeros(m, dtype=bool)
    rejections[order] = sorted_rejections
    return rejections


def benjamini_yekutieli_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Benjamini-Yekutieli correction.
    """
    p_values_1d, order = _sorted_pvalue_order(p_values)
    m = p_values_1d.size
    sorted_p = p_values_1d[order]
    harmonic = np.sum(1.0 / np.arange(1, m + 1))
    thresholds = float(alpha) * np.arange(1, m + 1) / (m * harmonic)
    passing = np.flatnonzero(sorted_p <= thresholds)
    sorted_rejections = np.zeros(m, dtype=bool)
    if passing.size > 0:
        sorted_rejections[: passing[-1] + 1] = True

    rejections = np.zeros(m, dtype=bool)
    rejections[order] = sorted_rejections
    return rejections


def compute_fwer(rejections_null: np.ndarray) -> float:
    """
    Return family-wise error rate from a [L, M] rejection matrix under the complete null.
    """
    rejections_null_2d = np.asarray(rejections_null, dtype=bool)
    return float(np.mean(np.any(rejections_null_2d, axis=1)))


def compute_fdr(rejections: np.ndarray, is_true_null: np.ndarray) -> float:
    """
    Return FDR for one simulation: false discoveries among all discoveries.
    Use 0.0 when there are no rejections.
    """
    rejections_1d = np.asarray(rejections, dtype=bool).reshape(-1)
    is_true_null_1d = np.asarray(is_true_null, dtype=bool).reshape(-1)
    discoveries = int(np.sum(rejections_1d))
    if discoveries == 0:
        return 0.0
    false_discoveries = int(np.sum(rejections_1d & is_true_null_1d))
    return float(false_discoveries / discoveries)


def compute_power(rejections: np.ndarray, is_true_null: np.ndarray) -> float:
    """
    Return power for one simulation: true rejections among false null hypotheses.
    """
    rejections_1d = np.asarray(rejections, dtype=bool).reshape(-1)
    is_true_null_1d = np.asarray(is_true_null, dtype=bool).reshape(-1)
    false_nulls = ~is_true_null_1d
    total_false_nulls = int(np.sum(false_nulls))
    if total_false_nulls == 0:
        return 0.0
    true_rejections = int(np.sum(rejections_1d & false_nulls))
    return float(true_rejections / total_false_nulls)


def summarize_multiple_testing(
    null_pvalues: pd.DataFrame,
    mixed_pvalues: pd.DataFrame,
    alpha: float,
) -> dict[str, float]:
    """
    Return summary metrics:
      fwer_uncorrected, fwer_bonferroni, fwer_holm,
      fdr_uncorrected, fdr_bh, fdr_by,
      power_uncorrected, power_bh, power_by.
    """
    null_sorted = null_pvalues.sort_values(
        ["sim_id", "hypothesis_id"],
        kind="mergesort",
    )
    null_matrix = (
        null_sorted.pivot(index="sim_id", columns="hypothesis_id", values="p_value")
        .sort_index()
        .sort_index(axis=1)
        .to_numpy()
    )

    uncorrected_null = null_matrix <= float(alpha)
    bonferroni_null = np.apply_along_axis(bonferroni_rejections, 1, null_matrix, alpha)
    holm_null = np.apply_along_axis(holm_rejections, 1, null_matrix, alpha)

    mixed_sorted = mixed_pvalues.sort_values(
        ["sim_id", "hypothesis_id"],
        kind="mergesort",
    )

    fdr_uncorrected: list[float] = []
    fdr_bh: list[float] = []
    fdr_by: list[float] = []
    power_uncorrected: list[float] = []
    power_bh: list[float] = []
    power_by: list[float] = []

    for _, sim_df in mixed_sorted.groupby("sim_id", sort=True):
        p_values = sim_df["p_value"].to_numpy(dtype=float)
        is_true_null = sim_df["is_true_null"].to_numpy(dtype=bool)

        uncorrected = p_values <= float(alpha)
        bh = benjamini_hochberg_rejections(p_values, float(alpha))
        by = benjamini_yekutieli_rejections(p_values, float(alpha))

        fdr_uncorrected.append(compute_fdr(uncorrected, is_true_null))
        fdr_bh.append(compute_fdr(bh, is_true_null))
        fdr_by.append(compute_fdr(by, is_true_null))

        power_uncorrected.append(compute_power(uncorrected, is_true_null))
        power_bh.append(compute_power(bh, is_true_null))
        power_by.append(compute_power(by, is_true_null))

    return {
        "fwer_uncorrected": compute_fwer(uncorrected_null),
        "fwer_bonferroni": compute_fwer(bonferroni_null),
        "fwer_holm": compute_fwer(holm_null),
        "fdr_uncorrected": float(np.mean(fdr_uncorrected)),
        "fdr_bh": float(np.mean(fdr_bh)),
        "fdr_by": float(np.mean(fdr_by)),
        "power_uncorrected": float(np.mean(power_uncorrected)),
        "power_bh": float(np.mean(power_bh)),
        "power_by": float(np.mean(power_by)),
    }
