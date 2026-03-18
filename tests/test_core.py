import numpy as np
import pandas as pd

from src.multiple_testing import (
    benjamini_hochberg_rejections,
    benjamini_yekutieli_rejections,
    bonferroni_rejections,
    compute_fdr,
    compute_fwer,
    compute_power,
    holm_rejections,
    simulate_mixed_pvalues,
    simulate_null_pvalues,
    summarize_multiple_testing,
)


def test_simulate_null_pvalues_columns_and_counts():
    config = {
        "seed_null": 11,
        "N": 40,
        "M": 7,
        "L": 5,
        "p_treat": 0.5,
    }
    df = simulate_null_pvalues(config=config)
    assert list(df.columns) == ["sim_id", "hypothesis_id", "p_value"]
    assert len(df) == 35
    assert df["sim_id"].nunique() == 5
    assert df["hypothesis_id"].nunique() == 7


def test_simulate_mixed_pvalues_columns_and_truth_labels():
    config = {
        "seed_mixed": 12,
        "N": 50,
        "M": 10,
        "M0": 6,
        "L": 4,
        "p_treat": 0.5,
        "tau_alternative": 0.2,
    }
    df = simulate_mixed_pvalues(config=config)
    assert list(df.columns) == ["sim_id", "hypothesis_id", "p_value", "is_true_null"]
    assert len(df) == 40
    assert df["is_true_null"].sum() == 24


def test_bonferroni_and_holm_on_fixed_vector():
    p_values = np.array([0.01, 0.02, 0.04, 0.20])
    alpha = 0.05
    bonf = bonferroni_rejections(p_values, alpha)
    holm = holm_rejections(p_values, alpha)
    assert np.array_equal(bonf, np.array([True, False, False, False]))
    assert np.array_equal(holm, np.array([True, False, False, False]))


def test_bh_and_by_on_fixed_vector():
    p_values = np.array([0.01, 0.02, 0.04, 0.20])
    alpha = 0.05
    bh = benjamini_hochberg_rejections(p_values, alpha)
    by = benjamini_yekutieli_rejections(p_values, alpha)
    assert np.array_equal(bh, np.array([True, True, False, False]))
    assert np.array_equal(by, np.array([False, False, False, False]))


def test_fwer_fdr_power_helpers():
    rejections_null = np.array(
        [
            [False, False, False],
            [True, False, False],
            [True, True, False],
        ]
    )
    assert np.isclose(compute_fwer(rejections_null), 2.0 / 3.0)

    rejections = np.array([True, False, True, False])
    is_true_null = np.array([True, True, False, False])
    assert np.isclose(compute_fdr(rejections, is_true_null), 0.5)
    assert np.isclose(compute_power(rejections, is_true_null), 0.5)


def test_summary_metrics_on_fixed_tables():
    null_pvalues = pd.DataFrame(
        {
            "sim_id": [0, 0, 0, 1, 1, 1],
            "hypothesis_id": [0, 1, 2, 0, 1, 2],
            "p_value": [0.2, 0.3, 0.4, 0.01, 0.3, 0.4],
        }
    )
    mixed_pvalues = pd.DataFrame(
        {
            "sim_id": [0, 0, 0, 0, 1, 1, 1, 1],
            "hypothesis_id": [0, 1, 2, 3, 0, 1, 2, 3],
            "p_value": [0.04, 0.2, 0.03, 0.8, 0.9, 0.8, 0.01, 0.02],
            "is_true_null": [True, True, False, False, True, True, False, False],
        }
    )
    out = summarize_multiple_testing(
        null_pvalues=null_pvalues,
        mixed_pvalues=mixed_pvalues,
        alpha=0.05,
    )

    expected = {
        "fwer_uncorrected": 0.5,
        "fwer_bonferroni": 0.5,
        "fwer_holm": 0.5,
        "fdr_uncorrected": 0.25,
        "fdr_bh": 0.0,
        "fdr_by": 0.0,
        "power_uncorrected": 0.75,
        "power_bh": 0.5,
        "power_by": 0.0,
    }
    for key, value in expected.items():
        assert np.isclose(out[key], value), f"Unexpected value for {key}"
