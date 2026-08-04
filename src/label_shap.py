"""SHAP on a LightGBM trained to predict the actual gold AGN label.

Additive counterpart to interpretation.py's cluster-explainer SHAP (which
explains HDBSCAN cluster membership -- partly circular, since clusters are a
function of the same features). This module trains directly on
`agn_positive` vs `agn_negative` (silver/unlabeled rows excluded from
training -- silver membership is itself a features-derived heuristic, see
preprocessing.build_labels, so including it would reintroduce the same
circularity this module exists to avoid). See results/LABEL_SHAP.md for the
full writeup. Does not modify interpretation.py or any of its callers.
"""

from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

LGBM_FIXED_PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=63, verbosity=-1)


def filter_gold(df_sample, X):
    """Restrict to gold rows (agn_positive | agn_negative); silver and
    unlabeled rows are dropped entirely, not just excluded from y. Returns
    (X_gold, y) with y=1 for agn_positive, y=0 for agn_negative."""
    mask = (df_sample["agn_positive"] | df_sample["agn_negative"]).to_numpy()
    X_gold = X.loc[mask]
    y = df_sample.loc[mask, "agn_positive"].astype(int)
    return X_gold, y


def imbalance_stats(y):
    """n_pos/n_neg/ratio (AGN:non-AGN) and the scale_pos_weight LightGBM should
    use (count(neg)/count(pos) -- < 1 here since AGN is the majority class,
    which is correct and should not be inverted)."""
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "ratio": n_pos / n_neg,
        "scale_pos_weight": n_neg / n_pos,
    }


def run_cv(X, y, scale_pos_weight, n_splits=5, random_state=42, early_stopping_rounds=50):
    """StratifiedKFold OOF predictions. Early stopping happens inside each fold
    against that fold's own held-out test split (metric: average_precision) --
    this is for metrics only; the final SHAP model is fit separately on the
    full gold sample (see fit_final_model). Returns (oof_pred: Series aligned
    to X.index, best_iterations: list[int])."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_pred = pd.Series(np.nan, index=X.index)
    best_iterations = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = lgb.LGBMClassifier(
            scale_pos_weight=scale_pos_weight, random_state=random_state, **LGBM_FIXED_PARAMS
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)], eval_metric="average_precision",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        best_iterations.append(model.best_iteration_)
        oof_pred.iloc[test_idx] = model.predict_proba(X_test)[:, 1]

    return oof_pred, best_iterations


def evaluate_oof(y, oof_pred, threshold=0.5):
    """average_precision is the primary metric (see LABEL_SHAP.md) -- reported
    first in the dict and first in the report's metrics table, since ROC-AUC
    is optimistic under this much class imbalance."""
    y_pred = (oof_pred >= threshold).astype(int)
    return {
        "average_precision": average_precision_score(y, oof_pred),
        "roc_auc": roc_auc_score(y, oof_pred),
        "brier": brier_score_loss(y, oof_pred),
        "confusion_matrix": confusion_matrix(y, y_pred),
    }


def choose_final_n_estimators(best_iterations, cap=500):
    """Mean of the CV folds' best_iteration if all folds stopped before `cap`
    (cheaper final-model fit, same effective capacity); falls back to `cap`
    if any fold ran the full budget without triggering early stopping."""
    valid = [b for b in best_iterations if b is not None and b > 0]
    if valid and max(valid) < cap:
        return int(round(np.mean(valid)))
    return cap


def fit_final_model(X, y, scale_pos_weight, n_estimators, random_state=42):
    """Final model for SHAP, fit on the ENTIRE gold sample -- no held-out
    slice. CV (run_cv) already characterizes generalization; this fit exists
    only to explain, not to be evaluated, and the minority class (n_neg) is
    too small to further shrink with a validation split."""
    model = lgb.LGBMClassifier(
        scale_pos_weight=scale_pos_weight, random_state=random_state,
        **{**LGBM_FIXED_PARAMS, "n_estimators": n_estimators},
    )
    model.fit(X, y)
    return model


def compute_shap(model, X):
    """TreeSHAP on a binary LightGBM classifier. Asserts the values array is
    2D (n_samples, n_features) -- binary LightGBM models return this shape
    directly (unlike the multiclass cluster-explainer surrogate in
    interpretation.py, which returns 3D and must branch on ndim). Positive
    SHAP pushes toward the AGN (y=1) prediction; negative toward non-AGN."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    values = shap_values.values
    expected_shape = (len(X), X.shape[1])
    assert values.ndim == 2 and values.shape == expected_shape, (
        f"Unexpected SHAP shape {values.shape}, expected {expected_shape}"
    )
    return shap_values


def top_features(shap_values, X, n=20):
    importance = np.abs(shap_values.values).mean(axis=0)
    return (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
        .head(n)
    )


def plot_shap_bar(top_df, save_path, title):
    top = top_df.sort_values("mean_abs_shap")
    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.xlabel("Mean |SHAP value| (toward AGN prediction)")
    plt.title(title)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_shap_beeswarm(shap_values, save_path, title, max_display=20):
    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.title(title)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
