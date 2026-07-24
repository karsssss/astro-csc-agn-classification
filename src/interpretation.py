"""LightGBM cluster-explainer surrogate, TreeSHAP, and reporting plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split


def train_lightgbm_surrogate(X, cluster_labels, minimum_model_cluster_size=50, random_state=42):
    """Train an LGBM multiclass surrogate to predict HDBSCAN cluster membership
    (excluding noise, and clusters smaller than minimum_model_cluster_size).
    Returns (model, X_train, X_test, y_train, y_test)."""
    valid_mask = cluster_labels != -1
    X_valid = X.loc[valid_mask].copy()
    y_valid = pd.Series(cluster_labels[valid_mask], index=X_valid.index, name="cluster")

    counts = y_valid.value_counts()
    keep_clusters = counts[counts >= minimum_model_cluster_size].index
    model_mask = y_valid.isin(keep_clusters)
    X_model, y_model = X_valid.loc[model_mask], y_valid.loc[model_mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X_model, y_model, test_size=0.25, stratify=y_model, random_state=random_state,
    )

    num_class = len(np.unique(y_train))
    model = LGBMClassifier(
        objective="multiclass", num_class=num_class, n_estimators=600,
        learning_rate=0.04, num_leaves=31, min_child_samples=30,
        subsample=0.85, subsample_freq=1, colsample_bytree=0.85, reg_lambda=2.0,
        class_weight="balanced", random_state=random_state, n_jobs=-1, verbosity=-1,
    )
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def compute_shap_values(model, X_shap):
    """TreeSHAP explanation, branching on ndim (3D multiclass vs 2D binary) --
    NOT the un-branched pattern that crashes on multiclass models (see
    results/SHAP_SUMMARY.md notes on the notebook's cells 80-81)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_shap)
    values = shap_values.values

    if values.ndim == 3:
        global_importance = np.abs(values).mean(axis=(0, 2))
    elif values.ndim == 2:
        global_importance = np.abs(values).mean(axis=0)
    else:
        raise ValueError(f"Unexpected SHAP shape: {values.shape}")

    global_shap = (
        pd.DataFrame({"feature": X_shap.columns, "mean_abs_shap": global_importance})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return shap_values, global_shap


def per_cluster_shap_tables(shap_values, X_shap, model_classes, top_n=10):
    """Return {cluster_id: top-N DataFrame of mean |SHAP| features} for each class."""
    values = shap_values.values
    tables = {}
    for i, cluster_id in enumerate(model_classes):
        if values.ndim == 3:
            importance = np.abs(values[:, :, i]).mean(axis=0)
        else:
            importance = np.abs(values).mean(axis=0)
        tables[int(cluster_id)] = (
            pd.DataFrame({"feature": X_shap.columns, "mean_abs_shap": importance})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
            .head(top_n)
        )
    return tables


def plot_shap_global(global_shap, top_n=15, save_path=None, title="Global feature importance"):
    top = global_shap.head(top_n).sort_values("mean_abs_shap")
    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top["mean_abs_shap"])
    plt.xlabel("Mean |SHAP value|")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_shap_per_cluster(cluster_shap_tables, save_dir=None, title_prefix="Cluster"):
    for cluster_id, table in sorted(cluster_shap_tables.items()):
        top = table.sort_values("mean_abs_shap")
        plt.figure(figsize=(8.5, 6))
        plt.barh(top["feature"], top["mean_abs_shap"])
        plt.xlabel("Mean |SHAP value|")
        plt.title(f"{title_prefix} {cluster_id} -- top feature importance")
        plt.tight_layout()
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / f"shap_cluster_{cluster_id}.png", dpi=150)
        plt.show()


def plot_cluster_composition(df, cluster_col="cluster", label_col="label_3class", save_path=None):
    """Stacked bar chart of %AGN (silver+confirmed) / %non-AGN / %unlabeled per cluster."""
    rows = []
    for cid, grp in df.groupby(cluster_col):
        rows.append({
            "cluster": cid,
            "pct_agn": (grp[label_col].isin([1, 2])).mean() * 100,
            "pct_nonagn": (grp[label_col] == 0).mean() * 100,
            "pct_unlabeled": (grp[label_col] == -1).mean() * 100,
        })
    composition = pd.DataFrame(rows).sort_values("cluster").set_index("cluster")

    plt.figure(figsize=(9, 6))
    bottom = np.zeros(len(composition))
    for col, label in [("pct_agn", "AGN"), ("pct_nonagn", "non-AGN"), ("pct_unlabeled", "unlabeled")]:
        plt.bar(composition.index.astype(str), composition[col], bottom=bottom, label=label)
        bottom += composition[col].to_numpy()
    plt.ylabel("% of cluster")
    plt.xlabel("Cluster")
    plt.title("Cluster composition by AGN label")
    plt.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()
    return composition
