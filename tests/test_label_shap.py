import numpy as np
import pandas as pd

from src.label_shap import (
    choose_final_n_estimators,
    compute_shap,
    fit_final_model,
    imbalance_stats,
    run_cv,
)

# Mirrors build_feature_matrix's 22-column Variant A final_cols (src/conversions.py)
# -- not imported directly since that list is a local variable inside the
# function, not an exported constant.
VARIANT_A_COLUMNS = [
    "significance", "WISE_w1snr", "WISE_w2snr", "WISE_w3snr",
    "hard_hs", "hard_hs_lolim", "hard_hs_hilim",
    "hard_hm", "hard_hm_lolim", "hard_hm_hilim",
    "hard_ms", "hard_ms_lolim", "hard_ms_hilim",
    "xray_b_over_r", "xray_b_over_w1", "xray_b_over_w2", "xray_m_over_r",
    "optical_r_over_w1",
    "color_g_over_r", "color_r_over_z", "color_w1_over_w2", "color_w2_over_w3",
]


def _synthetic_gold_sample(n=500, random_state=0):
    rng = np.random.RandomState(random_state)
    X = pd.DataFrame(rng.randn(n, len(VARIANT_A_COLUMNS)), columns=VARIANT_A_COLUMNS)
    # Imbalanced binary target, majority-positive -- mirrors the real gold
    # sample's AGN-majority imbalance direction (see LABEL_SHAP.md).
    y = pd.Series((rng.rand(n) < 0.85).astype(int), index=X.index)
    return X, y


def test_label_shap_end_to_end_shap_shape_matches_n_samples_by_n_features():
    X, y = _synthetic_gold_sample(n=500)

    stats = imbalance_stats(y)
    assert stats["n_pos"] + stats["n_neg"] == len(y)

    oof_pred, best_iterations = run_cv(X, y, stats["scale_pos_weight"], n_splits=5, random_state=42)
    assert oof_pred.notna().all()
    assert len(best_iterations) == 5

    n_estimators = choose_final_n_estimators(best_iterations)
    model = fit_final_model(X, y, stats["scale_pos_weight"], n_estimators, random_state=42)

    shap_values = compute_shap(model, X)
    assert shap_values.values.shape == (X.shape[0], X.shape[1])
