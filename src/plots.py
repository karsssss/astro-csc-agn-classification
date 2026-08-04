"""Redshift-coded UMAP embedding plots.

New, additive module -- does not refactor or replace any of
interpretation.py's existing plotting functions (which color by cluster or
by label_3class categorically, never by a continuous value with optional
size/shape encodings). See results/REDSHIFT_PLOTS.md for the driving
analysis.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def _scale_sizes(size_col, size_range=(10, 100), clip_percentiles=(1, 99)):
    """Clip to [clip_percentiles] then linearly scale into size_range. Returns
    (sizes, (clip_lo, clip_hi)) -- the clip bounds are needed to build a
    faithful size legend."""
    size_col = np.asarray(size_col, dtype=float)
    clip_lo, clip_hi = np.percentile(size_col, clip_percentiles)
    clipped = np.clip(size_col, clip_lo, clip_hi)
    if clip_hi > clip_lo:
        frac = (clipped - clip_lo) / (clip_hi - clip_lo)
    else:
        frac = np.zeros_like(clipped)
    sizes = size_range[0] + frac * (size_range[1] - size_range[0])
    return sizes, (clip_lo, clip_hi)


def _size_for_value(value, clip_bounds, size_range=(10, 100)):
    clip_lo, clip_hi = clip_bounds
    v = np.clip(value, clip_lo, clip_hi)
    frac = (v - clip_lo) / (clip_hi - clip_lo) if clip_hi > clip_lo else 0.0
    return size_range[0] + frac * (size_range[1] - size_range[0])


def plot_embedding_by_value(
    embedding, value_col, subset_mask, size_col=None, shape_col=None,
    value_label="combo_Z (redshift)", cmap="viridis",
    size_range=(10, 100), size_clip_percentiles=(1, 99), size_legend_values=None,
    shape_labels=("Confirmed AGN", "Confirmed non-AGN"), shape_markers=("o", "^"),
    title="", caption=None, save_path=None, figsize=(11, 9), point_size=18,
):
    """Single-panel UMAP (or any 2D embedding) scatter colored by a continuous
    value, with optional size and/or shape encodings -- consolidates the three
    redshift-plot variants in results/REDSHIFT_PLOTS.md into one function.

    embedding: (n, 2) array. value_col, size_col, shape_col: length-n
    array-likes aligned to embedding's rows (not column-name strings -- this
    function takes raw arrays, not a DataFrame). subset_mask: length-n
    boolean array selecting which rows to actually plot.

    shape_col, if given, must be boolean or take exactly 2 distinct values
    within the subset; the two groups are drawn with shape_markers[0] (for
    the True/first value) and shape_markers[1], labeled by shape_labels.
    size_col, if given, is clipped to size_clip_percentiles and linearly
    scaled into size_range; a manual size legend is added using
    size_legend_values (representative *data* values, not pixel sizes) or,
    if not given, the min/median/max of size_col within the subset.

    value_col may contain NaN for rows that have no value at all (e.g. no
    redshift) -- those rows are drawn in a fixed neutral gray with no
    colormap/colorbar contribution, rather than dropped or miscolored,
    so a group entirely lacking the value (e.g. non-AGN with no
    spectroscopic redshift) can still be shown alongside a colored group.
    """
    embedding = np.asarray(embedding)
    value_col = np.asarray(value_col, dtype=float)
    subset_mask = np.asarray(subset_mask, dtype=bool)

    emb_sub = embedding[subset_mask]
    val_sub = value_col[subset_mask]
    size_sub = np.asarray(size_col, dtype=float)[subset_mask] if size_col is not None else None
    shape_sub = np.asarray(shape_col)[subset_mask] if shape_col is not None else None
    nan_sub = ~np.isfinite(val_sub)

    fig, ax = plt.subplots(figsize=figsize)

    if size_sub is not None:
        sizes, clip_bounds = _scale_sizes(size_sub, size_range, size_clip_percentiles)
    else:
        sizes, clip_bounds = np.full(len(emb_sub), point_size), None

    if nan_sub.all():
        raise ValueError("value_col is NaN for every row in subset_mask -- nothing to color by")
    vmin, vmax = val_sub[~nan_sub].min(), val_sub[~nan_sub].max()

    def _draw_group(mask, marker, label_suffix=""):
        """Draws one shape group's valid-value points (colormap) and NaN-value
        points (fixed gray) separately; returns the colormap scatter handle
        (or None if this group has no valid-value points)."""
        valid_mask = mask & ~nan_sub
        gray_mask = mask & nan_sub
        sc = None
        if valid_mask.any():
            sc = ax.scatter(
                emb_sub[valid_mask, 0], emb_sub[valid_mask, 1],
                c=val_sub[valid_mask], cmap=cmap, vmin=vmin, vmax=vmax,
                s=sizes[valid_mask], marker=marker, alpha=0.75,
                linewidths=0.2, edgecolors="black",
                label=f"{label_suffix} (n={valid_mask.sum()})" if label_suffix else None,
            )
        if gray_mask.any():
            ax.scatter(
                emb_sub[gray_mask, 0], emb_sub[gray_mask, 1],
                color="lightgray", s=sizes[gray_mask], marker=marker, alpha=0.6,
                linewidths=0.2, edgecolors="dimgray",
                label=f"{label_suffix}, no {value_label} (n={gray_mask.sum()})" if label_suffix else None,
            )
        return sc

    if shape_sub is not None:
        groups = [True, False] if set(np.unique(shape_sub)) <= {True, False} else list(np.unique(shape_sub))[:2]
        scatter = None
        for group_val, marker, label in zip(groups, shape_markers, shape_labels):
            mask = shape_sub == group_val
            sc = _draw_group(mask, marker, label)
            scatter = sc or scatter
        shape_legend = ax.legend(loc="upper left", title="Marker shape")
        ax.add_artist(shape_legend)
    else:
        scatter = _draw_group(np.ones(len(emb_sub), dtype=bool), "o")

    if scatter is None:
        raise ValueError("No valid-value points to build a colorbar from -- every row was NaN")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(value_label)

    if size_sub is not None:
        if size_legend_values is None:
            size_legend_values = [np.min(size_sub), np.median(size_sub), np.max(size_sub)]
        handles = [
            Line2D(
                [], [], marker="o", linestyle="none", color="gray",
                markersize=np.sqrt(_size_for_value(v, clip_bounds, size_range)),
                label=f"z = {v:.2f}",
            )
            for v in size_legend_values
        ]
        size_legend = ax.legend(handles=handles, loc="lower left", title=f"Marker size ({value_label})")
        ax.add_artist(size_legend)
        if shape_sub is not None:
            ax.add_artist(shape_legend)

    ax.set_title(f"{title} (n={len(emb_sub)})")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    if caption:
        fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=9, style="italic")

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig, ax
