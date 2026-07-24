"""UMAP embedding and HDBSCAN grid search over a scaled feature matrix."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import hdbscan
import umap
from sklearn.metrics import silhouette_score


@dataclass
class ClusteringResult:
    """One HDBSCAN configuration's outcome."""

    config: dict
    labels: np.ndarray
    n_clusters: int
    n_model_clusters: int
    noise_fraction: float
    silhouette: float
    cluster_counts: pd.Series = field(repr=False)


def fit_umap(X_scaled, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42):
    """Fit UMAP and return the 2D (by default) embedding array."""
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric="euclidean",
        random_state=random_state,
        low_memory=True,
    )
    return reducer.fit_transform(X_scaled)


def run_hdbscan_grid(X_scaled, configs, minimum_model_cluster_size=50):
    """Run HDBSCAN over a list of {min_cluster_size, min_samples, cluster_selection_method}
    configs. Returns (results: dict[key -> ClusteringResult], summary: DataFrame)."""
    results = {}
    summary_rows = []

    for config in configs:
        mcs = config["min_cluster_size"]
        ms = config["min_samples"]
        csm = config.get("cluster_selection_method", "eom")
        key = f"mcs_{mcs}_ms_{ms}_{csm}"

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs, min_samples=ms, metric="euclidean",
            cluster_selection_method=csm, prediction_data=True, core_dist_n_jobs=-1,
        )
        labels = clusterer.fit_predict(X_scaled)

        valid_mask = labels != -1
        n_valid = valid_mask.sum()
        n_clusters = len(set(labels) - {-1})
        noise_fraction = float((labels == -1).mean())
        cluster_counts = pd.Series(labels[valid_mask]).value_counts().sort_index()
        n_model_clusters = int((cluster_counts >= minimum_model_cluster_size).sum())

        if n_clusters >= 2 and n_valid > n_clusters:
            silhouette = silhouette_score(
                X_scaled[valid_mask], labels[valid_mask],
                sample_size=min(10_000, n_valid), random_state=42,
            )
        else:
            silhouette = np.nan

        results[key] = ClusteringResult(
            config=config, labels=labels, n_clusters=n_clusters,
            n_model_clusters=n_model_clusters, noise_fraction=noise_fraction,
            silhouette=silhouette, cluster_counts=cluster_counts,
        )
        summary_rows.append({
            "configuration": key, "min_cluster_size": mcs, "min_samples": ms,
            "cluster_selection_method": csm, "n_clusters": n_clusters,
            "n_model_clusters": n_model_clusters, "noise_fraction": noise_fraction,
            "silhouette": silhouette,
        })

    return results, pd.DataFrame(summary_rows)


def select_best_config(summary, target_min_clusters=4):
    """Pick the config with the highest silhouette among those reaching
    target_min_clusters modelable clusters; falls back to the config with the
    most modelable clusters (then best silhouette) if none reach the target.
    Returns (best_key, met_target: bool)."""
    candidates = summary[summary["n_model_clusters"] >= target_min_clusters]
    if len(candidates) > 0:
        best_row = candidates.sort_values("silhouette", ascending=False).iloc[0]
        met_target = True
    else:
        best_row = summary.sort_values(
            ["n_model_clusters", "silhouette"], ascending=[False, False]
        ).iloc[0]
        met_target = False
    return best_row["configuration"], met_target
