"""Capture a pre-migration behavioral baseline from notebooks/unsupervised.ipynb.

This script NEVER reads back into or writes notebooks/unsupervised.ipynb -- it
loads the source notebook once (read-only) and builds two independent
in-memory nbformat notebook objects from *copies* of its cells, each executed
in its own fresh kernel via nbclient. The source file on disk is untouched.

Why two notebook objects instead of one linear run: cell 29 (the "Reduced
Dataset Case") mutates `df_umap_base` in place, reducing it to 24 columns.
Cell 31 (the "General UMAP case") then reads `df_umap_base` expecting the
full ~174-column catalog. Running the source notebook top-to-bottom in one
kernel would silently feed the General case cell 29's already-reduced frame
instead of the full one -- the two "parallel configurations" CLAUDE.md
describes cannot be captured correctly from a single linear execution. See
the chat/PLAN.md discussion this script's design responds to.

Two variants are built:

- "reduced": the Reduced Dataset Case branch, run in full (cells 0-29,
  36-89, 114-149 minus exclusions below) -- the path CLAUDE.md documents as
  running end to end with no known errors.
- "general_partial": the General UMAP case branch, scoped to only the part
  CLAUDE.md documents as verified (StandardScaler/PCA + a single UMAP fit),
  via cells 0-28 + 31-35, plus an injected `neighbors_list = [15]` override
  placed *before* the otherwise-unmodified UMAP-sweep cell (cell 37) so only
  one fit runs instead of the full 4-setting sweep. The General case's
  HDBSCAN grid and PaCMAP sweep at full ~170K-row scale are intentionally
  NOT run here -- see the "notes" field of the output JSON.

Excluded entirely (not executed, not migrated):
- Cells 90-113 ("Previous UMAP algorithm" section) -- superseded, see
  LAB_NOTEBOOK.md.
- Cell 142 ("Multi-iterative model") -- non-functional self-training loop,
  see references/notes/cell_142_analysis.md and LAB_NOTEBOOK.md.
- Cell 136 -- references `df_sup["train_weight"]`, a column never assigned
  anywhere in the notebook (only ever listed as "drop this column if
  present" in two exclude-lists); raises KeyError if executed. Like cell
  142, nothing downstream reads this cell's outputs (cells 137+ redefine
  their own `models`/`X`/`y`/`w`/`oof_preds` independently) -- same
  isolated-dead-cell pattern, flagged in the output JSON's "notes" rather
  than given its own trace writeup.
- Cells 80-81 -- cell 80 recomputes global SHAP importance without the
  ndim==3-vs-2 branch cell 75 has, so it raises ValueError on this
  multiclass cluster-explainer model's 3D SHAP array. Nothing downstream
  reads cell 81's `cluster_shap_table`. Unlike cell 136/142, cells 80-81
  implement functionality the migration DOES want (global + per-cluster
  SHAP importance tables) -- just incorrectly; `interpretation/shap_utils.py`
  must follow cell 75-77's ndim-branching pattern, not cell 80-81's. See
  the output JSON's "notes" field.

Run from the `quant` conda env:
    python scripts/capture_baseline.py
"""

from __future__ import annotations

import copy
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # headless: no GUI backend during execution

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "unsupervised.ipynb"
OUTPUT_PATH = REPO_ROOT / "references" / "baselines" / "pre_migration.json"

PER_CELL_TIMEOUT_S = 600  # constraint: surface any cell over 10 min, don't silently hang
LOAD_BEARING_PACKAGES = [
    "numpy", "pandas", "scikit-learn", "umap-learn", "hdbscan",
    "lightgbm", "catboost", "shap", "pacmap",
]

EXCLUDED_CELLS = {80, 81, 136, 142}  # broken / dead-end, see module docstring

REDUCED_CASE_HEADER = """
import matplotlib
matplotlib.use("Agg")

import numpy as _np
from sklearn.model_selection import StratifiedKFold as _SKFold
from sklearn.metrics import (
    roc_auc_score as _roc_auc,
    average_precision_score as _ap,
    balanced_accuracy_score as _bacc,
)

_classifiers = []
_hdbscan_grid = []


def _cv_binary_scores(cv, X, y, oof):
    scores = {"roc_auc": [], "pr_auc": []}
    for _, va in cv.split(X, y):
        y_va = _np.asarray(y)[va]
        p_va = _np.asarray(oof)[va]
        scores["roc_auc"].append(float(_roc_auc(y_va, p_va)))
        scores["pr_auc"].append(float(_ap(y_va, p_va)))
    return scores


def _cv_multiclass_scores(cv, X, y, oof, classes):
    scores = {"roc_auc_ovr": [], "pr_auc_macro": []}
    for _, va in cv.split(X, y):
        y_va = _np.asarray(y)[va]
        p_va = _np.asarray(oof)[va]
        scores["roc_auc_ovr"].append(
            float(_roc_auc(y_va, p_va, multi_class="ovr", labels=classes))
        )
        pr = _np.mean([
            _ap((y_va == c).astype(int), p_va[:, i])
            for i, c in enumerate(classes)
        ])
        scores["pr_auc_macro"].append(float(pr))
    return scores


def _summarize(scores):
    return (
        {k: float(_np.mean(v)) for k, v in scores.items()},
        {k: float(_np.std(v)) for k, v in scores.items()},
    )
"""

# Injected immediately before cell 52 (the HDBSCAN grid search). Forces
# core_dist_n_jobs=1 for every hdbscan.HDBSCAN(...) instance used downstream
# in THIS kernel only. Patches HDBSCAN.fit (not __init__) so it sets
# self.core_dist_n_jobs = 1 right before delegating to the original fit --
# cell 52's own `import hdbscan` just returns the same cached module object,
# so its `core_dist_n_jobs=-1` constructor kwarg is overridden transparently
# at fit time. Patching .fit instead of __init__ leaves sklearn's
# BaseEstimator.get_params() (which introspects __init__'s signature) fully
# intact -- no subclass, no varargs __init__ anywhere in this path. Does not
# touch the source notebook or cluster_results' structure. Baseline-capture-
# only: see the "notes" field of the output JSON.
FORCE_DETERMINISTIC_HDBSCAN = """
import hdbscan as _hdbscan_module

_original_hdbscan_fit = _hdbscan_module.HDBSCAN.fit


def _deterministic_hdbscan_fit(self, *args, **kwargs):
    self.core_dist_n_jobs = 1
    return _original_hdbscan_fit(self, *args, **kwargs)


_hdbscan_module.HDBSCAN.fit = _deterministic_hdbscan_fit
"""

CAPTURE_AFTER_HDBSCAN_GRID = """
for _key, _row in zip(cluster_results.keys(), summary_rows):
    _labels = cluster_results[_key]["labels"]
    _counts = cluster_results[_key]["cluster_counts"]
    _hdbscan_grid.append({
        "config": {
            "min_cluster_size": _row["min_cluster_size"],
            "min_samples": _row["min_samples"],
        },
        "n_clusters": int(_row["n_clusters"]),
        "n_noise": int((_labels == -1).sum()),
        "noise_fraction": float(_row["noise_fraction"]),
        "cluster_sizes_sorted": sorted(
            [int(v) for v in _counts.values], reverse=True
        ),
        "silhouette": (
            None if _np.isnan(_row["silhouette"]) else float(_row["silhouette"])
        ),
        "dbcv": None,  # not computed anywhere in the source notebook
    })
"""

CAPTURE_AFTER_CELL_128 = """
_cv128 = _SKFold(n_splits=5, shuffle=True, random_state=42)
for _name, _model_type in (("lightgbm", "lgbm"), ("catboost", "catboost")):
    _scores = _cv_binary_scores(_cv128, X_labeled, y_labeled, oof_predictions[_name])
    _mean, _std = _summarize(_scores)
    _classifiers.append({
        "cell": 128, "model_type": _model_type, "task": "binary",
        "cv_folds": 5, "cv_scores": _scores,
        "mean_scores": _mean, "std_scores": _std,
    })
"""

CAPTURE_AFTER_CELL_138 = """
_cv138 = _SKFold(n_splits=5, shuffle=True, random_state=42)
for _name, _model_type in (("lightgbm", "lgbm"), ("catboost", "catboost")):
    _scores = _cv_multiclass_scores(
        _cv138, X_labeled_reset, y_labeled_arr, oof[_name], unique_classes
    )
    _mean, _std = _summarize(_scores)
    _classifiers.append({
        "cell": 138, "model_type": _model_type, "task": "multiclass",
        "cv_folds": 5, "cv_scores": _scores,
        "mean_scores": _mean, "std_scores": _std,
    })
"""

CAPTURE_AFTER_CELL_139 = """
_cv139 = _SKFold(n_splits=5, shuffle=True, random_state=42)
for _name, _model_type in (("lightgbm", "lgbm"), ("catboost", "catboost")):
    _scores = _cv_binary_scores(_cv139, X_labeled, y_labeled, oof[_name])
    _mean, _std = _summarize(_scores)
    _classifiers.append({
        "cell": 139, "model_type": _model_type, "task": "binary",
        "cv_folds": 5, "cv_scores": _scores,
        "mean_scores": _mean, "std_scores": _std,
    })
"""

REDUCED_FINAL_EXTRACTION = """
import json as _json

# --- cluster-explainer model (cells 67-73): single train/test split, not CV.
# Kept out of _classifiers -- it's SHAP instrumentation, not a scientific CV
# result, and lumping a cv_folds=1/std=0.0 entry in with real k-fold results
# would let a tolerance-check misread it as a strict-equality target.
_single_split_metrics = [{
    "cell": 71,
    "context": "cluster_explainer_lgbm",
    "model_type": "lgbm",
    "task": "multiclass",
    "test_size": 0.25,
    "random_state": 42,
    "metrics": {"balanced_accuracy": float(_bacc(y_test, y_pred))},
}]

_feature_matrix = {
    "shape": list(X_features.shape),
    "columns": list(X_features.columns),
    "dtypes": {c: str(X_features[c].dtype) for c in X_features.columns},
    "missing_counts": {c: int(X_features[c].isna().sum()) for c in X_features.columns},
}

_label_3class_counts = {
    str(k): int(v) for k, v in df["label_3class"].value_counts().sort_index().items()
}

_labels = {
    "n_gold_positive": int(df["agn_positive"].sum()),
    "n_silver_positive": int(df["silver_agn"].sum()),
    "n_unlabeled": int((df["label_3class"] == -1).sum()),
    "q80_r_threshold": float(q80_r),
    "label_3class_counts": _label_3class_counts,
}

_umap_configs = [
    {
        "branch": "reduced",
        "n_neighbors": int(n),
        "min_dist": 0.1,
        "embedding_shape": list(embeddings[n].shape),
    }
    for n in neighbors_list
]

_result = {
    "feature_matrix": _feature_matrix,
    "labels": _labels,
    "reduction": {"pca": None, "umap_configs": _umap_configs},
    "hdbscan_grid": _hdbscan_grid,
    "classifiers": _classifiers,
    "single_split_metrics": _single_split_metrics,
}

print("===BASELINE_REDUCED_JSON===" + _json.dumps(_result))
"""

GENERAL_PARTIAL_OVERRIDE = (
    "neighbors_list = [15]  "
    "# capture_baseline.py: single-fit override -- see script module docstring"
)

GENERAL_PARTIAL_EXTRACTION = """
import json as _json

_result = {
    "reduction": {
        "pca": {
            "n_components": int(n_pca),
            "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        },
        "umap_configs": [
            {
                "branch": "general_partial",
                "n_neighbors": int(n),
                "min_dist": 0.1,
                "embedding_shape": list(embeddings[n].shape),
            }
            for n in neighbors_list
        ],
    },
    "general_feature_matrix_shape": list(X.shape),
}

print("===BASELINE_GENERAL_JSON===" + _json.dumps(_result))
"""


def load_source_cells():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    return nb, list(nb.cells)


def build_reduced_case_notebook(nb_template, all_cells):
    inject_before = {
        52: FORCE_DETERMINISTIC_HDBSCAN,
    }
    inject_after = {
        52: CAPTURE_AFTER_HDBSCAN_GRID,
        128: CAPTURE_AFTER_CELL_128,
        138: CAPTURE_AFTER_CELL_138,
        139: CAPTURE_AFTER_CELL_139,
    }
    keep_ranges = [range(0, 30), range(36, 90), range(114, 150)]

    cells = [nbformat.v4.new_code_cell(REDUCED_CASE_HEADER)]
    for rng in keep_ranges:
        for i in rng:
            if i in EXCLUDED_CELLS:
                continue
            if i in inject_before:
                cells.append(nbformat.v4.new_code_cell(inject_before[i]))
            cells.append(copy.deepcopy(all_cells[i]))
            if i in inject_after:
                cells.append(nbformat.v4.new_code_cell(inject_after[i]))
    cells.append(nbformat.v4.new_code_cell(REDUCED_FINAL_EXTRACTION))

    variant = copy.deepcopy(nb_template)
    variant.cells = cells
    return variant


def build_general_partial_notebook(nb_template, all_cells):
    cells = [nbformat.v4.new_code_cell('import matplotlib; matplotlib.use("Agg")')]
    for i in list(range(0, 29)) + list(range(31, 36)):
        cells.append(copy.deepcopy(all_cells[i]))
    cells.append(nbformat.v4.new_code_cell(GENERAL_PARTIAL_OVERRIDE))
    cells.append(copy.deepcopy(all_cells[37]))  # verbatim UMAP-sweep cell, unmodified
    cells.append(nbformat.v4.new_code_cell(GENERAL_PARTIAL_EXTRACTION))

    variant = copy.deepcopy(nb_template)
    variant.cells = cells
    return variant


def execute_notebook(nb, label):
    client = NotebookClient(
        nb,
        timeout=PER_CELL_TIMEOUT_S,
        kernel_name="python3",
        allow_errors=False,
    )
    start = time.monotonic()
    try:
        client.execute()
    except CellTimeoutError as exc:
        elapsed = time.monotonic() - start
        raise SystemExit(
            f"[{label}] a cell exceeded the {PER_CELL_TIMEOUT_S}s per-cell timeout "
            f"after {elapsed:.0f}s total elapsed. Stopping per constraint 1 -- "
            f"tell the user before continuing rather than raising the timeout "
            f"and re-running blind.\n{exc}"
        ) from exc
    except CellExecutionError as exc:
        raise SystemExit(f"[{label}] a cell raised during execution:\n{exc}") from exc
    elapsed = time.monotonic() - start
    print(f"[{label}] executed {len(nb.cells)} cells in {elapsed:.1f}s", file=sys.stderr)
    return nb


def extract_marker_json(nb, marker):
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            text = output.get("text", "")
            for line in text.splitlines():
                if line.startswith(marker):
                    return json.loads(line[len(marker):])
    raise RuntimeError(f"marker {marker!r} not found in any cell output")


def git_commit_hash():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def package_versions():
    versions = {}
    for pkg in LOAD_BEARING_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def main():
    nb_template, all_cells = load_source_cells()

    reduced_nb = build_reduced_case_notebook(nb_template, all_cells)
    execute_notebook(reduced_nb, "reduced")
    reduced_result = extract_marker_json(reduced_nb, "===BASELINE_REDUCED_JSON===")

    general_nb = build_general_partial_notebook(nb_template, all_cells)
    execute_notebook(general_nb, "general_partial")
    general_result = extract_marker_json(general_nb, "===BASELINE_GENERAL_JSON===")

    reduced_result["reduction"]["pca"] = general_result["reduction"]["pca"]
    reduced_result["reduction"]["umap_configs"] += general_result["reduction"]["umap_configs"]

    notes = (
        "Baseline captured from the 'Reduced Dataset Case' branch in full "
        "(cells 0-29, 36-89, 114-149), per user-approved scope. The 'General "
        "UMAP case' branch (cells 31-35 + a single n_neighbors=15 UMAP fit) "
        "was captured only through the part CLAUDE.md documents as verified "
        "(StandardScaler/PCA + one UMAP fit); its full 4-setting UMAP sweep, "
        "8-config HDBSCAN grid, and PaCMAP sweep at ~170K-row scale were NOT "
        "run here (CLAUDE.md: a single HDBSCAN config didn't complete within "
        "20 minutes at that scale) -- general_case_feature_matrix_shape is "
        "included below for reference but hdbscan_grid/classifiers only "
        "reflect the reduced-case branch. "
        "Cells 90-113 (superseded 'Previous UMAP algorithm' section, see "
        "LAB_NOTEBOOK.md) and cell 142 (non-functional self-training loop, "
        "see references/notes/cell_142_analysis.md) were excluded entirely -- "
        "not migrated, so no baseline is meaningful for them. Cell 107's "
        "RF+SHAP subset analysis was also not captured: its exact subset "
        "definition depends on the dropped legacy UMAP embedding (cells "
        "90-106) and is being redefined during migration per PLAN.md open "
        "question 4 (classification/cluster_explainer.py's general "
        "train_and_explain_subset); no comparable historical number exists "
        "to capture. Cell 136 was excluded: it references "
        "df_sup['train_weight'], a column never assigned anywhere in the "
        "notebook (KeyError if run), and nothing downstream reads its "
        "outputs -- same isolated-dead-cell pattern as cell 142, flagged "
        "here rather than given a full trace writeup. "
        "Cells 80-81 excluded due to shape bug (naive .mean(axis=0) on "
        "multiclass SHAP values). Downstream never reads their outputs, so "
        "baseline is unaffected. IMPORTANT for migration: cells 80-81 "
        "implement functionality we want to keep (global + per-cluster SHAP "
        "importance tables), just incorrectly. The correct pattern exists "
        "in cells 75-77 (explicit ndim branch for 2D binary vs 3D "
        "multiclass SHAP arrays). interpretation/shap_utils.py must follow "
        "the 75-77 pattern, not the 80-81 pattern, when this stage of the "
        "migration is implemented. "
        "hdbscan.HDBSCAN (installed version, see package_versions) exposes "
        "no random_state parameter in this release -- not a missing seed in "
        "the notebook, the library doesn't have one. Its core algorithm is "
        "deterministic given fixed input, but the source notebook's "
        "core_dist_n_jobs=-1 (parallelized core-distance computation) is a "
        "secondary source of run-to-run float-level nondeterminism at "
        "distance tie-breaks. For THIS baseline capture only, every "
        "hdbscan.HDBSCAN(...) call was monkey-patched (a subclass "
        "overriding core_dist_n_jobs=1 in __init__, injected right before "
        "cell 52 in this run's in-memory copy) to force single-threaded "
        "core-distance computation, so the hdbscan_grid values below are "
        "reproducible run to run. The source notebook and cluster_results' "
        "structure were not modified -- the override is execution-time-only "
        "and applies to this script's kernel session, not to "
        "notebooks/unsupervised.ipynb or any future production run. "
        "Downstream/production runs may still use core_dist_n_jobs=-1 for "
        "speed, with the tie-break-nondeterminism caveat noted above; if "
        "reproducing this exact baseline, use core_dist_n_jobs=1. "
        "Every other stochastic call found in the notebook (UMAP, PaCMAP, "
        "PCA, train_test_split, DataFrame.sample, LightGBM, CatBoost, "
        "pseudo_positive_bagging, bagging_pnu_scores) sets an explicit seed "
        "(random_state/seed=42 throughout)."
    )

    output = {
        "commit": git_commit_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "package_versions": package_versions(),
        "feature_matrix": reduced_result["feature_matrix"],
        "labels": reduced_result["labels"],
        "reduction": reduced_result["reduction"],
        "hdbscan_grid": reduced_result["hdbscan_grid"],
        "classifiers": reduced_result["classifiers"],
        "single_split_metrics": reduced_result["single_split_metrics"],
        "general_case_feature_matrix_shape": general_result["general_feature_matrix_shape"],
        "notes": notes,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    # Round-trip + completeness check
    reloaded = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    required_nonnull = [
        "commit", "timestamp", "python_version", "feature_matrix", "labels",
        "hdbscan_grid", "classifiers",
    ]
    missing = [k for k in required_nonnull if reloaded.get(k) is None]
    if missing:
        raise SystemExit(f"baseline JSON failed completeness check, null fields: {missing}")

    print(f"wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
