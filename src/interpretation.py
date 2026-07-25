"""LightGBM cluster-explainer surrogate, TreeSHAP, and reporting plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split

# label_3class taxonomy from preprocessing.build_labels: -1 unlabeled, 0
# confirmed non-AGN, 1 silver AGN, 2 confirmed AGN. Draw order puts unlabeled
# in the background and confirmed non-AGN (the rarest, most diagnostic class)
# on top so it isn't hidden under denser AGN classes.
LABEL_3CLASS_STYLE = {
    -1: {"name": "Unlabeled", "color": "lightgray", "alpha": 0.15},
    1: {"name": "Silver AGN", "color": "orange", "alpha": 0.6},
    2: {"name": "Confirmed AGN", "color": "red", "alpha": 0.6},
    0: {"name": "Confirmed non-AGN", "color": "blue", "alpha": 0.8},
}
LABEL_3CLASS_DRAW_ORDER = [-1, 1, 2, 0]


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


def _cluster_color_map(cluster_labels):
    """Assign a stable color to each non-noise cluster id (tab10), independent
    of which embedding it's plotted on -- so the same cluster is the same
    color across panels."""
    cluster_ids = sorted(int(c) for c in set(cluster_labels) if c != -1)
    cmap = plt.get_cmap("tab10" if len(cluster_ids) <= 10 else "tab20")
    return {cid: cmap(i % cmap.N) for i, cid in enumerate(cluster_ids)}


def _scatter_embedding(ax, embedding, cluster_labels, color_map, title, point_size=4):
    counts = pd.Series(cluster_labels).value_counts()

    noise_mask = cluster_labels == -1
    if noise_mask.any():
        ax.scatter(
            embedding[noise_mask, 0], embedding[noise_mask, 1],
            s=point_size, c="lightgray", alpha=0.3, linewidths=0,
            label=f"Noise (n={counts.get(-1, 0)})",
        )
    for cid, color in color_map.items():
        mask = cluster_labels == cid
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            s=point_size, color=color, alpha=0.7, linewidths=0,
            label=f"Cluster {cid} (n={counts.get(cid, 0)})",
        )
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")


def plot_embedding_comparison(
    embedding_left, embedding_right, cluster_labels,
    title_left="UMAP", title_right="PaCMAP", save_path=None, figsize=(18, 8),
):
    """Two-panel scatter of two embeddings of the SAME rows, colored by the
    SAME HDBSCAN cluster labels with a shared color map -- for eyeballing
    whether one reduction resolves structure the other doesn't. Noise (-1)
    is drawn first, in light gray, so real clusters render on top."""
    color_map = _cluster_color_map(cluster_labels)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize)
    _scatter_embedding(ax_left, embedding_left, cluster_labels, color_map, title_left)
    _scatter_embedding(ax_right, embedding_right, cluster_labels, color_map, title_right)

    handles, labels = ax_right.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), markerscale=3)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_embedding_single(embedding, cluster_labels, title="PaCMAP", save_path=None, figsize=(11, 9)):
    """Single larger-panel scatter, same color/legend convention as
    plot_embedding_comparison -- for a standalone read of one embedding."""
    color_map = _cluster_color_map(cluster_labels)

    fig, ax = plt.subplots(figsize=figsize)
    _scatter_embedding(ax, embedding, cluster_labels, color_map, title, point_size=5)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), markerscale=3)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_embedding_by_label(
    embedding, label_3class, title="UMAP (v_final) -- colored by ground-truth label",
    save_path=None, figsize=(11, 9), point_size=4,
):
    """Single-panel scatter of an embedding colored by the label_3class
    ground-truth taxonomy (see LABEL_3CLASS_STYLE), not by HDBSCAN cluster --
    for eyeballing whether a given HDBSCAN cluster region visually
    concentrates a particular ground-truth label (e.g. confirmed non-AGN).
    Same drawn-first-in-background / external-legend-with-counts convention
    as plot_embedding_comparison/plot_embedding_single."""
    labels = np.asarray(label_3class)
    counts = pd.Series(labels).value_counts()

    unrecognized = set(np.unique(labels)) - set(LABEL_3CLASS_DRAW_ORDER)
    if unrecognized:
        print(f"plot_embedding_by_label: unrecognized label_3class values not plotted: {unrecognized}")

    fig, ax = plt.subplots(figsize=figsize)
    for val in LABEL_3CLASS_DRAW_ORDER:
        mask = labels == val
        if not mask.any():
            continue
        style = LABEL_3CLASS_STYLE[val]
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            s=point_size, color=style["color"], alpha=style["alpha"], linewidths=0,
            label=f"{style['name']} (n={counts.get(val, 0)})",
        )
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), markerscale=3)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_distributions_by_cluster(
    X, cluster_labels, features, noise_label=-1, save_path=None, ncols=4, figsize=None,
):
    """Grid of one boxplot panel per feature: x = real HDBSCAN cluster id,
    y = feature value (from X, in its native/unscaled units). The noise
    pool's IQR is drawn as a light gray reference band behind each panel's
    boxes so a cluster's shift away from "typical" (mostly-noise) values is
    visible at a glance.

    X must be row-aligned with cluster_labels (same order/length) -- e.g. the
    same X passed to train_lightgbm_surrogate / fit_umap upstream."""
    cluster_labels = np.asarray(cluster_labels)
    if len(cluster_labels) != len(X):
        raise ValueError(f"cluster_labels length {len(cluster_labels)} != X length {len(X)}")

    real_clusters = sorted(int(c) for c in set(cluster_labels) if c != noise_label)
    noise_mask = cluster_labels == noise_label

    n = len(features)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (4.5 * ncols, 3.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for i, feature in enumerate(features):
        ax = axes[i // ncols][i % ncols]
        values = X[feature].to_numpy()

        if noise_mask.any():
            noise_vals = values[noise_mask]
            q25, q75 = np.nanpercentile(noise_vals, [25, 75])
            ax.axhspan(q25, q75, color="lightgray", alpha=0.5, zorder=0)
            ax.axhline(np.nanmedian(noise_vals), color="gray", linestyle="--", linewidth=1, zorder=0)

        data = [values[cluster_labels == cid] for cid in real_clusters]
        ax.boxplot(data, positions=real_clusters, widths=0.6, showfliers=False)
        ax.set_title(feature, fontsize=10)
        ax.set_xlabel("Cluster")
        ax.set_xticks(real_clusters)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    legend_handles = [
        Patch(color="lightgray", alpha=0.5, label="Noise IQR (25-75%)"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1, label="Noise median"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    fig.suptitle("Per-cluster feature distributions vs. noise pool -- v_final", y=1.06)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
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
