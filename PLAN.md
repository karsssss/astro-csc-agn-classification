# Restructuring plan: `unsupervised.ipynb` → `src/agn_classification/`

Status: **DRAFT — awaiting approval. No code has been moved.**

## 0. Important context before you approve this

`unsupervised.ipynb` contains only **3 actual `def` functions** (`to_bool_safe`,
`pseudo_positive_bagging`, `bagging_pnu_scores`), plus the 3 already extracted to
`src/features.py` (`mag_to_flux_jy`, `apply_mag_to_flux_conversion`,
`cross_check_mag_zero_point`). Everything else — data loading, label construction,
UMAP/PCA prep, HDBSCAN sweeps, classifier CV loops, SHAP, plotting — is written as
flat procedural code directly in cells, not wrapped in functions.

So this plan does two things at once, and you should approve/adjust both:
1. Assigns the 6 functions that already exist to modules (straightforward).
2. Proposes **new function boundaries** around the inline logic blocks that will
   need to be *created* during the actual migration (not just moved). This is
   the bulk of the work and the bulk of the judgment calls — please read
   section 3 ("Open questions") and the duplication list before approving.

I did not find a clean home for classifier training (LightGBM/CatBoost CV loops,
PU/PNU semi-supervised bagging) in your five listed submodules
(`data/`, `features/`, `reduction/`, `clustering/`, `interpretation/`, `viz/`).
I'm proposing a 6th submodule, `classification/`, for this — see open question 1.

## 1. Open questions (need your input before I finalize)

1. **Missing submodule for classifier training.** Your spec lists
   `data/features/reduction/clustering/interpretation/viz` but ~40% of the
   notebook (cells 67-81, 107, 116-149) is LightGBM/CatBoost training, CV,
   PU-bagging, and PNU-bagging — this doesn't fit cleanly under `clustering/`
   (that's HDBSCAN) or `interpretation/` (that's SHAP-on-a-trained-model, not
   training itself). Proposing `classification/` as a 6th submodule. OK, or did
   you want this folded into one of the existing five?
2. **The "Previous UMAP algorithm" section (cells 90-113) looks superseded.**
   It re-runs a second, separate UMAP fit (`reducer15`) and a second, separate
   HDBSCAN fit, both duplicating cells 36-65's sweep for the `n_neighbors=15`
   case specifically, feeding a differently-named `umap_df` used only in that
   section's plots. Do you want this migrated as a real (if redundant) code
   path, or dropped in favor of the newer cells 36-65 pipeline it appears to
   supersede?
3. **Cell 142 (`for i in range(0, 10): ...`) looks like it may not do what it
   intends.** It refits `final_lgbm`/`final_cat` ten times but never updates
   `y_labeled` or actually folds newly-scored pseudo-labels back into training
   — each iteration re-trains on the exact same labeled data and overwrites
   `X_labeled` with an unrelated `df.copy()`. Worth you re-checking this cell's
   intent before I port its logic anywhere.
4. **Cell 107** is a self-contained RandomForest+SHAP analysis on one IR-AGN
   subset (`umap_sub`), structurally similar to but independent from the main
   LightGBM/CatBoost classifier pipeline. Treat it as a first-class
   `classification/` pipeline stage, or leave it notebook-only as exploratory
   analysis?

## 2. Proposed directory structure

```
src/agn_classification/
    __init__.py
    data/
        __init__.py
        loaders.py         # read_catalog, merge_csc_spectral_fits, merge_desi
        cleaning.py        # clean_types (bool coercion, blank-string -> NaN)
    features/
        __init__.py
        photometry.py      # = current src/features.py, moved as-is, plus:
                            #   mask_raw_magnitude_sentinels, cross_check_wise_vega_ab_offset
        colors.py          # WISE/optical/IR colors, mateos_wedge_mask (deduped),
                            # build_combo_photometry, xray ratio features
        labels.py          # to_bool_safe, gold/silver label construction,
                            # label_3class, train_label/sample_weight
    reduction/
        __init__.py
        preprocessing.py   # build_feature_matrix, drop_sparse_columns,
                            # log_transform_large_scale_columns,
                            # impute_with_missing_indicators, scale_features
        pca.py             # fit_pca_sweep, fit_pca
        umap_embed.py       # fit_umap_sweep
        pacmap_embed.py     # fit_pacmap_sweep
    clustering/
        __init__.py
        hdbscan_search.py  # run_hdbscan_grid_search, summarize_hdbscan_grid,
                            # assign_best_clusters
        profiles.py        # cluster_size_summary, cluster_label_crosstab,
                            # cluster_feature_profiles
    classification/         # NEW — see open question 1
        __init__.py
        training.py         # run_stratified_cv_oof (consolidates 4 near-duplicate
                            # CV loops), evaluate_classifier
        semi_supervised.py  # pseudo_positive_bagging, bagging_pnu_scores,
                            # fuse_oof_scores_logit, select_pseudo_positive_candidates,
                            # select_pnu_pseudo_labels
        cluster_explainer.py # train_cluster_explainer_model (LGBM on cluster labels,
                            # feeds interpretation/shap_utils.py)
    interpretation/
        __init__.py
        shap_utils.py       # compute_shap_importance (2D/3D shape handling)
    viz/
        __init__.py
        style.py             # shared plt.rcParams setup (currently ad hoc in one cell)
        embedding_plots.py   # plot_embedding_by_label, plot_embedding_highlight_columns,
                            # plot_embedding_by_value -- replaces ~13 near-duplicate cells
        color_color.py       # plot_wise_color_color, plot_subset_comparison_histograms
        shap_plots.py         # plot_shap_bar, beeswarm wrapper

configs/
    reduction.yaml          # max_feature_missing, n_pca, neighbors_list, big_col threshold
    hdbscan.yaml            # the 8-config grid from cell 52
    classifiers.yaml        # LGBM/CatBoost hyperparams (currently 5 separate inline dicts)

notebooks/
    01_data_merge_explore.ipynb        # was: load/merge cells of unsupervised.ipynb
    02_feature_engineering.ipynb       # colors, flux conversion, labels
    03_dimensionality_reduction.ipynb  # PCA/UMAP/PaCMAP, both Reduced/General cases
    04_clustering.ipynb                # HDBSCAN grid search + profiles
    05_classification.ipynb            # supervised + PU/PNU semi-supervised
    06_visualization.ipynb             # UMAP/PaCMAP/WISE plots
    (data_comparison.ipynb, subsetcode.ipynb, umapcopy.ipynb, umap_augmented.ipynb,
     new.ipynb, data_explore.ipynb -- not renumbered yet; separate pass once the
     numbering scheme for unsupervised.ipynb's split is agreed)

scripts/
    run_pipeline.py         # full load -> features -> reduce -> cluster -> classify
    train_classifier.py     # classification-only entry point (given a processed df)

tests/
    test_photometry.py      # mag_to_flux_jy, apply_mag_to_flux_conversion,
                            # cross_check_mag_zero_point, sentinel masking (already
                            # implicitly covered by CLAUDE.md's verification discipline
                            # section -- these should become real pytest tests)
    test_colors.py          # mateos_wedge_mask, xray ratio features
    test_labels.py          # gold/silver priority rule, label_3class encoding
```

## 3. Function & logic inventory

Legend for **Status**: `EXISTS` = already a real function today; `INLINE` = proposed
new function to extract from an unwrapped code block; `DUP` = flagged duplicate,
see section 4.

### Stage 1 — Data load & merge (`data/`)

| Cell(s) | Status | Current / proposed name | Target module | Notes |
|---|---|---|---|---|
| 0 | INLINE | `load_catalog(path)` | `data/loaders.py` | Hardcoded path; dead commented-out `.fits` branch to drop |
| 1 | INLINE | `merge_csc_spectral_fits(df, path)` | `data/loaders.py` | Hardcoded path |
| 2 | INLINE | `merge_desi(df, path)` | `data/loaders.py` | Hardcoded path + hardcoded `desi_cols` list |
| 3 | INLINE | `clean_types(df)` | `data/cleaning.py` | bool_cols fillna+astype, blank-string→NaN |
| 4 | INLINE / DUP | `agn_class` OR construction | `features/labels.py` | Same OR logic recomputed as `agn_positive` in cell 25 — see §4 |
| 104 | — | *(not a function — re-reads raw CSV mid-notebook)* | n/a | Anti-pattern: reload of data already in `df`; should not be ported, just reuse `df` |

### Stage 2 — Feature engineering & labeling (`features/`)

| Cell(s) | Status | Current / proposed name | Target module | Notes |
|---|---|---|---|---|
| 5 | INLINE | `add_wise_colors(df)` | `features/colors.py` | |
| 5 | INLINE / DUP | `mateos_wedge_mask(w2_w3, w1_w2)` | `features/colors.py` | Formula copy-pasted 3x — see §4 |
| 5 | INLINE | `add_optical_colors(df)` | `features/colors.py` | PS/LEG colors, PS/WISE optical-IR colors |
| 5 | INLINE | `add_legacy_xray_ratios(df)` | `features/colors.py` | `xray_opt_r/g_relation_legacy`; keep name as-is per CLAUDE.md (downstream cells depend on it) |
| 7, 9, 10 | — | *(notebook-only sanity-check cells)* | n/a | Stay in notebook 02; not extracted |
| 12 | EXISTS | `apply_mag_to_flux_conversion` | `features/photometry.py` | Already in `src/features.py`; move as-is |
| 14 | EXISTS | `cross_check_mag_zero_point` | `features/photometry.py` | Already in `src/features.py`; move as-is |
| 15 | INLINE | `cross_check_wise_vega_ab_offset(df, offsets)` | `features/photometry.py` | New — currently a bare loop |
| 17 | INLINE | `add_xray_ratio_features(df, zero_point_jy, xray_log_flux_cols)` | `features/photometry.py` | Systematic `xray_ratio_{b,h,s}_<band>` builder |
| 21 | INLINE | `mask_raw_magnitude_sentinels(df, zero_point_jy)` | `features/photometry.py` | High priority: this is the "must run upstream of X/X_all" sentinel fix CLAUDE.md calls out — needs a regression test |
| 22 | INLINE / DUP | uses `mateos_wedge_mask` (2nd copy) + plotting | `features/colors.py` + `viz/color_color.py` | Split: mask formula dedupes into colors.py; scatter/wedge-line plotting → viz |
| 23 | INLINE | `add_csc_spectral_fit_flag(df)` | `features/labels.py` | |
| 24 | EXISTS | `to_bool_safe(s)` | `features/labels.py` | Real function already; move as-is |
| 24 | INLINE | `add_redshift_quality_flags(df)` | `features/labels.py` | `has_combo_z` / `has_high_quality_combo_z`; large commented-out (Russian-language) dead block to drop |
| 25 | INLINE | `build_gold_labels(df, agn_cols, neg_col)` | `features/labels.py` | Priority-to-positive-label rule — load-bearing, needs a test per CLAUDE.md |
| 25 | INLINE | `build_silver_score(df)` | `features/labels.py` | Weighted heuristic; also produces module-level `q80_r` reused later — see cell 110 note |
| 25 | INLINE | `build_label_3class(df)` | `features/labels.py` | |
| 25 | INLINE | `build_train_targets(df)` | `features/labels.py` | `label_source` / `train_label` / `sample_weight` |
| 110 | INLINE | depends on `q80_r` from cell 25 | `features/labels.py` | Hidden-state coupling: `q80_r` is a bare notebook global computed ~85 cells earlier — must become an explicit return value/param, not a rediscovered global |

### Stage 3 — Dimensionality reduction (`reduction/`)

| Cell(s) | Status | Current / proposed name | Target module | Notes |
|---|---|---|---|---|
| 29 | INLINE / DUP | `build_combo_photometry(df)` | `features/colors.py` | `combo_g/r/z_mag` via `combine_first`; copy-pasted 3x total — see §4 |
| 29 | INLINE / DUP | `select_reduced_case_features(df, columns)` | `reduction/preprocessing.py` | `dropna(subset=[...])` column list copy-pasted with cell 116 — see §4 |
| 31 | INLINE / DUP | `build_feature_matrix(df, exclude_cols)` | `reduction/preprocessing.py` | `exclude_cols` list copy-pasted with cell 122's `metadata_cols` — see §4 |
| 32 | INLINE | `drop_sparse_columns(X, max_missing_pct=90)` | `reduction/preprocessing.py` | |
| 33 | INLINE | `log_transform_large_scale_columns(X, threshold=1e20)` | `reduction/preprocessing.py` | The `CSC_b_lum`-style special case |
| 34 | INLINE / DUP | `impute_with_missing_indicators(X, sample_size=None, random_state=42)` | `reduction/preprocessing.py` | Same pattern repeated in cells 49, 120 — see §4 |
| 35 | INLINE | `fit_pca_sweep(X, n_components_grid)` + `fit_pca(X, n_components)` | `reduction/pca.py` | Also wraps StandardScaler — consider whether scaling belongs here or in preprocessing.py |
| 37 | INLINE / DUP | `fit_umap_sweep(X_pca, neighbors_list, **kwargs)` | `reduction/umap_embed.py` | Same shape as PaCMAP sweep (cell 83) — see §4 |
| 45, 47 | INLINE | `clean_feature_matrix(X)` | `reduction/preprocessing.py` | to_numeric, inf→NaN, -999→NaN |
| 83 | INLINE / DUP | `fit_pacmap_sweep(X_pca, neighbors_list, **kwargs)` | `reduction/pacmap_embed.py` | See §4 |
| 94 | INLINE / DUP | single UMAP fit (`reducer15`) | n/a (legacy) | Part of "Previous UMAP algorithm" section — see open question 2 |

### Stage 4 — Clustering (`clustering/`)

| Cell(s) | Status | Current / proposed name | Target module | Notes |
|---|---|---|---|---|
| 49 | INLINE / DUP | imputation (3rd copy) | `reduction/preprocessing.py` | Reuses `impute_with_missing_indicators` |
| 50 | INLINE | `scale_features(X)` | `reduction/preprocessing.py` | |
| 52 | INLINE | `run_hdbscan_grid_search(X_scaled, configs)` | `clustering/hdbscan_search.py` | The 8-config grid → `configs/hdbscan.yaml` |
| 54 | INLINE | `summarize_hdbscan_grid(cluster_results)` | `clustering/hdbscan_search.py` | |
| 58 | INLINE | `assign_best_clusters(df, cluster_results, best_key)` | `clustering/hdbscan_search.py` | |
| 61-63 | INLINE | `cluster_size_summary(df_clusters)`, `cluster_label_crosstab(df_clusters, normalize)` | `clustering/profiles.py` | |
| 64-65 | INLINE | `cluster_feature_profiles(X_features, cluster_labels)` | `clustering/profiles.py` | Median/IQR-normalized profile |
| 91 | INLINE / DUP | second, simpler HDBSCAN fit | n/a (legacy) | Part of "Previous UMAP algorithm" section — see open question 2 |

### Stage 5 — Classification (`classification/`, new submodule — open question 1)

| Cell(s) | Status | Current / proposed name | Target module | Notes |
|---|---|---|---|---|
| 67-71 | INLINE | `train_cluster_explainer_model(X_imputed, cluster_labels, min_cluster_size=100)` | `classification/cluster_explainer.py` | LGBM trained on HDBSCAN cluster labels, purely to feed SHAP |
| 72, 73 | INLINE / DUP | `evaluate_classifier(model, X_test, y_test)` | `classification/training.py` | Cells 72 and 73 are **verbatim identical** — see §4 |
| 74-81 | INLINE / DUP | `compute_shap_importance(model, X_shap)` | `interpretation/shap_utils.py` | Cells 76-77 vs. 80 recompute the same top-20 bar plot twice — see §4 |
| 107 | INLINE | `train_and_explain_rf_subset(...)` | `classification/` or leave notebook-only | Self-contained RF+SHAP on `umap_sub` — see open question 4 |
| 116 | INLINE / DUP | reuses `build_combo_photometry` + reduced-case column list (3rd copy) | `features/colors.py` / `reduction/preprocessing.py` | See §4 |
| 117, 121 | INLINE / DUP | `binary_target` construction | `features/labels.py` | Cells 117 and 121 are **verbatim identical** `np.select` blocks — see §4 |
| 119-120 | INLINE / DUP | 4th copy of impute-with-missing-indicators pattern | `reduction/preprocessing.py` | |
| 122 | INLINE / DUP | `metadata_cols` (2nd copy of cell 31's `exclude_cols`) | `reduction/preprocessing.py` | See §4 |
| 123-124 | INLINE | missing-flag augmentation + `bad_keywords` column filter | `reduction/preprocessing.py` | |
| 126-131 | INLINE | `run_stratified_cv_oof(X, y, models, cv)`, `fuse_oof_scores_logit(*scores)`, `select_pseudo_positive_candidates(u_results, ...)` | `classification/training.py`, `classification/semi_supervised.py` | |
| 132 | EXISTS | `pseudo_positive_bagging` | `classification/semi_supervised.py` | Real function already; move as-is |
| 136, 138, 139 | INLINE / DUP | 3 more from-scratch CV loops | `classification/training.py` | Near-identical to cells 126-128's loop and each other — consolidate into one `run_stratified_cv_oof` call site with different args — see §4 |
| 140-143 | INLINE | fit-final-model + score-unknowns + iterative self-training loop | `classification/semi_supervised.py` | Cell 142's loop looks logically incomplete — see open question 3, do not port as-is without checking |
| 144 | EXISTS | `bagging_pnu_scores` | `classification/semi_supervised.py` | Real function already; move as-is |
| 145-148 | INLINE | `select_pnu_pseudo_labels(df, score_col, std_col, pos_threshold, neg_threshold)` | `classification/semi_supervised.py` | |

### Stage 6 — Visualization (`viz/`)

| Cell(s) | Status | Current / proposed name | Target module | Notes |
|---|---|---|---|---|
| 38, 40, 42, 43, 44 (UMAP), 85, 87, 88, 89 (PaCMAP) | INLINE / DUP | `plot_embedding_by_label(embedding_df, x_col, y_col, label_col, subset=None, title=...)` | `viz/embedding_plots.py` | **9 near-identical blocks** collapse to one parameterized function — see §4 |
| 97, 98, 99, 100 | INLINE / DUP | same target function | `viz/embedding_plots.py` | Legacy `umap_df` plots — part of open question 2 |
| 101, 102, 110, 111 | INLINE / DUP | `plot_embedding_highlight_columns(embedding_df, columns, base_mask_col="label_3class")` | `viz/embedding_plots.py` | 4 near-identical `for label_col in [...]` loops — see §4 |
| 22, 105 | INLINE | `plot_wise_color_color(df, wedge_mask, subsets=...)` | `viz/color_color.py` | Reuses deduped `mateos_wedge_mask` instead of recomputing |
| 106 | INLINE | `plot_subset_comparison_histograms(...)` | `viz/color_color.py` (or new `viz/histograms.py`) | 4-panel significance/redshift/hardness comparison |
| 112, 113 | INLINE | fold into `plot_embedding_by_label` / new `plot_embedding_by_value(embedding_df, value_col)` | `viz/embedding_plots.py` | |
| 22 | INLINE | shared `plt.rcParams.update(...)` | `viz/style.py` | Currently an ad hoc global mutation inside one plotting cell |

## 4. Duplication summary (copy-paste flagged across cells)

1. **Mateos wedge formula** (`y <= 0.315x+0.796 & ...`) — hand-copied 3x: cells 5, 22, 105. Should be one `mateos_wedge_mask()` helper.
2. **`combo_g`/`combo_r`/`combo_z_mag` construction** (`combine_first` of PS1/Legacy) — copy-pasted 3x: cells 29, 116 (and referenced again around cell 122). One `build_combo_photometry()` helper.
3. **Reduced-case `dropna(subset=[...])` column list** — identical ~24-column list copy-pasted 2x: cells 29 and 116.
4. **`exclude_cols` / `metadata_cols` list** — identical list of label/metadata columns to drop before building the feature matrix, copy-pasted 2x: cells 31 and 122.
5. **`agn_class` (cell 4) vs. `agn_positive` (cell 25)** — same `xray_agn | optical_agn | ir_agn` OR computed twice under two different names.
6. **`binary_target` construction** — cells 117 and 121 are word-for-word identical `np.select` blocks.
7. **Impute-with-missing-indicators pattern** (`SimpleImputer(strategy="median")` + `_missing` indicator columns + hstack/concat) — repeated with minor variation 4x: cells 34, 49, 120, 123.
8. **Classification-report + balanced-accuracy evaluation** — cells 72 and 73 are verbatim identical.
9. **Global SHAP importance + top-20 bar plot** — recomputed twice: cells 76-77 and cell 80.
10. **UMAP/PaCMAP sweep loop shape** (`for n_neighbors in neighbors_list: reducer = ...; embedding = reducer.fit_transform(...)`) — same structure in cells 37 and 83, differing only by reducer class.
11. **Label-colored embedding scatter plot** — the single most duplicated pattern: **9 near-identical blocks** (cells 38, 40, 42, 43, 44, 85, 87, 88, 89), each re-deriving `mask_agn`/`mask_nonagn`/`mask_silver`/`mask_unlabeled` and re-plotting with the same colors/alphas, varying only which subset(s) are shown and UMAP-vs-PaCMAP axis names.
12. **`for label_col in [...]: ... scatter(...)` highlight-loop** — near-identical bodies in cells 101, 102, 110, 111, varying only the column list being iterated.
13. **From-scratch StratifiedKFold + LGBM/CatBoost CV training loop** — reimplemented independently 4 times with small variations (sample weights, multiclass vs. binary, class-balancing) in cells 128, 136, 138, 139, instead of one parameterized function called 4 times.
14. **LGBM/CatBoost hyperparameter dicts** — 5 separate inline `models = {...}` blocks (cells 127, 136, 138, 139, 145) with slightly different hyperparameters each time; candidates to move into `configs/classifiers.yaml` so the differences become explicit and reviewable.

## 5. What I'm explicitly *not* proposing to move yet

- `data_comparison.ipynb`, `data_explore.ipynb`, `new.ipynb`, `subsetcode.ipynb`,
  `umapcopy.ipynb`, `umap_augmented.ipynb` — out of scope for this pass; renumbering
  and auditing those is a separate exercise once `unsupervised.ipynb`'s split is
  approved and done.
- Anything in the "Previous UMAP algorithm" section (cells 90-113) pending your
  answer to open question 2.
- Cell 107's RandomForest+SHAP subset analysis pending open question 4.

---

**Waiting for your approval / edits before touching any code.**
