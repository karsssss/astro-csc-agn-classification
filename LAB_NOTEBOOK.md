# Lab notebook

Chronological record of research decisions that aren't otherwise visible from
the code or `CLAUDE.md` — things that were tried, superseded, or deliberately
dropped, and why.

Entries are in reverse chronological order (newest at top). Undated context
and conventions live in this header.

## 2026-08-04 — Redshift-coded UMAP plots, Variant B (32-feat) addendum

Additive follow-up to the Variant A redshift plots below: the same two
figure types (gold-only colored by `combo_Z`, AGN-only with size+color
both encoding `combo_Z`) rebuilt for the Variant B (32-feat) feature
space, with UMAP parameters held fixed (`n_neighbors=15, min_dist=0.1,
n_components=2, random_state=42`) so the feature space is the only thing
that varies. Added as two extra sections in the same
`notebooks/redshift_plots.ipynb` (not a separate notebook — all the
loading/labeling/gating logic was already there). Embedding persisted to
`data/processed/umap_embedding_extended_32feat.npy`. Figures in
`results/figures/extended_32feat/`; full write-up appended to
`results/REDSHIFT_PLOTS.md`.

**Findings:**

1. **The zero-non-AGN-redshift finding holds in Variant B too** (8,899 of
   11,621 gold rows have valid `combo_Z`, and every one of them is AGN —
   0 of 566 gold non-AGN have one). Confirms the catalog-wide result below
   isn't an artifact of the Variant A sample specifically.
2. **Variant B shows the same qualitative redshift gradient as Variant A,
   but weaker.** Pearson r(UMAP1, `combo_Z`) = 0.462 for Variant B vs.
   -0.578 for Variant A; r(UMAP2, `combo_Z`) = 0.155 vs. -0.303. **The
   sign flip on both axes is not meaningful by itself** — UMAP's output
   orientation (rotation/reflection) is arbitrary per independent fit and
   carries no physical meaning; only the *magnitude* of the correlation is
   comparable across two separately-fit embeddings. Reading magnitudes:
   same primary-axis-dominant pattern, but consistently weaker in both
   axes for Variant B. The underlying AGN population is nearly identical
   either way (median z 0.82 vs. 0.77, similar IQR) — this is the same
   redshift signal expressed less cleanly in UMAP position once 10 extra
   columns (X-ray/optical/WISE log-fluxes, CSC variability indices) are
   added, not a different population being plotted.
3. **Converging cross-analysis result: the 22-feat space is doing as well
   or better than the 32-feat space on two independent axes.**
   `results/LABEL_SHAP.md` found near-identical gold-label predictive
   performance between variants (Variant A AP=0.9941 vs. Variant B
   AP=0.9947 — a negligible difference, well within noise for this
   sample-size gap). This redshift-gradient check adds a second,
   independent signal pointing the same direction: Variant A's embedding
   organizes by redshift *more* strongly (|r|=0.578/0.303) than Variant
   B's (|r|=0.462/0.155). Two unrelated analyses (a supervised classifier
   and an unsupervised embedding) now agree the simpler 22-feat space
   isn't losing anything by leaving out the 10 extra flux/variability
   columns — worth remembering before assuming "more features" is
   automatically an improvement for this catalog.

## 2026-08-04 — Extinction/reddening correction (Yang 2022 §6) feasibility check

~10-minute scoping check, written up in
`references/notes/extinction_feasibility.md`. No code changes, no
correction applied -- this was a "should we, and how much work" question,
not a deliverable.

**Three findings:**

1. **`gal_b` coverage is itself class-asymmetric**: 100% for
   `agn_positive` (33,136/33,136) but only 39.7% for `agn_negative`
   (2,398/6,037), despite 97.9% coverage catalog-wide. `ra`/`dec` (100%
   coverage) are a reliable fallback for the missing rows if this is ever
   needed.
2. **Zero gold rows (either class) fall below \|b\|=10°** -- the minimum
   \|gal_b\| in the entire gold sample is ~15° for both `agn_positive` and
   `agn_negative`. This is exactly the low-Galactic-latitude regime Yang
   §6 says reddening "bites hardest" in, and this catalog's footprint
   simply doesn't reach it (likely deliberate plane-avoidance in how the
   underlying X-ray source catalog was assembled).
3. **A full Yang §6 implementation (SFD map setup + per-band optical/IR
   extinction at the `mag_to_flux_jy`/`apply_mag_to_flux` layer + a
   separate, harder X-ray N_H absorption correction per Yang Eq. 5) is a
   multi-hour, likely multi-session effort** -- not a quick patch, given
   this project's own physical-units verification bar
   (`CLAUDE.md`'s "Verification discipline for physical-units work").
   Combined with finding 2, **not recommended as a near-term priority**:
   the correction would have essentially no footprint on this catalog's
   actual sample.

**Session-wide pattern worth flagging:** this is the **third**
coverage-asymmetry finding today where `agn_positive` rows are
systematically more complete than `agn_negative` rows on some auxiliary
column -- `combo_Z` redshift (100% vs. 0%, `REDSHIFT_PLOTS.md`), CSC X-ray
variability (2-4x higher coverage across all 18 columns,
`variability_check.md`), and now `gal_b` (100% vs. 39.7%). Not
investigated further this session, but this looks like a structural
property of how this catalog was assembled (AGN candidates preferentially
targeted for follow-up/cross-match across multiple independent auxiliary
datasets) rather than three unrelated coincidences -- worth keeping in
mind for *any* future AGN-vs-non-AGN comparison in this catalog, since a
missing-data pattern that tracks the label itself can bias naive
comparisons in non-obvious ways.

## 2026-08-04 — Variability check: WISE/optical variability claim untestable, CSC X-ray variability near-chance

Time-boxed (25 min) univariate check, written up in
`references/notes/variability_check.md`. No changes to either variant's
`X` in `src/conversions.py` -- analysis only, per scope.

**Three findings:**

1. **The catalog has no WISE variability columns and no optical
   variability columns at all** -- checked directly against every column
   in the general df and all three raw source CSVs (regex across
   `WISE_*var*`/`*_mjd*`/`WISE_pm*`/`*moon_lev*`/`*qual*`/`*cc*`,
   `*variab*`, `*phot_variable_flag*`): zero matches for WISE or optical.
   The only genuine variability data anywhere in this project is the 18
   CSC `var_intra_*`/`var_inter_*` X-ray columns. This means a meeting
   claim comparing "WISE variability vs. optical variability" as an AGN
   feature **cannot be tested in this catalog** -- not a case of one
   winning over the other, but of neither existing to compare.
2. **CSC X-ray variability itself carries near-chance single-feature
   signal.** The 4 of 18 CSC variability columns clearing a 50%
   gold-coverage bar (`var_intra_index_b/h/s/m`) all have ROC-AUC within
   0.05 of 0.5 for `agn_positive` vs. `agn_negative` (best: `b` at
   AUC=0.457). This triangulates with two independent prior results:
   `EXTENDED_FEATURES_COMPARISON.md`'s Variant B cluster-SHAP (where
   `var_intra_index_b/m` rank #6/#12 of 32 -- real but modest) and
   `results/LABEL_SHAP.md`'s gold-label SHAP (where neither variability
   column cracks the top-20 for either feature-space variant). Three
   independent methods (univariate AUC, multivariate cluster-SHAP,
   multivariate label-SHAP) now agree CSC X-ray variability is weak at
   best for AGN/non-AGN separation in this data.
3. **A striking coverage asymmetry, not a value asymmetry, may itself be
   signal.** Every one of the 18 CSC variability columns has 2-4x higher
   non-null coverage among `agn_positive` rows than `agn_negative` rows
   (e.g. `var_intra_index_b`: 98.1% vs. 37.1%). Confirmed AGN are simply
   far more likely to have *any* variability measurement recorded at all,
   independent of what the measurement says. Not investigated further
   here (out of this check's 25-minute scope), but worth a dedicated
   follow-up: missingness-as-feature (a boolean "has variability
   measurement" flag) might carry more signal than any of the 4 raw
   values checked above.

Recommendation from this check: 0 of 18 candidate columns added to `X`
(univariate evidence too weak to justify it, and this null result is
corroborated rather than contradicted by prior multivariate work). Full
ranked AUC table, coverage table, and the KDE figure are in
`references/notes/variability_check.md` and
`results/figures/reduced_22feat/variability_class_distributions.png`.

## 2026-08-04 — Redshift-coded UMAP plots; combo_Z found to be de facto AGN-only

Built `src/plots.py` (`plot_embedding_by_value`, new/additive -- does not
touch `interpretation.py`'s existing plotting functions) +
`notebooks/redshift_plots.ipynb` (executed, outputs stored) +
`results/REDSHIFT_PLOTS.md`. Reused the Variant A (22-feat) UMAP embedding
from `analysis.ipynb` -- same pipeline, same `fit_umap` defaults
(`n_neighbors=15, min_dist=0.1, n_components=2, random_state=42`),
re-executed since it wasn't persisted anywhere before this task. Now
persisted to `data/processed/umap_embedding_reduced_22feat.npy` (+ a
matching `_index.npy`) for reuse by future tasks.

**Three findings:**

1. **`combo_Z` is de facto an AGN-only column, catalog-wide.** Verified
   directly against the full 170,196-row df: of the 27,335 rows with a
   valid, high-quality `combo_Z`, 100% are `agn_positive` and **zero** are
   `agn_negative`, `silver_agn`, or unlabeled -- and `agn_negative` rows
   don't even have `combo_Z` *present* (native NaN), let alone
   high-quality. Likely explanation: the DESI/SDSS/AGN-database
   spectroscopic cross-match this catalog's `combo_Z` comes from
   preferentially targets AGN candidates for follow-up, rather than being
   a blind census of the X-ray source population. This forced a
   mid-task redesign of the third planned figure (see
   `REDSHIFT_PLOTS.md` for the full before/after) -- a naive "gold,
   filtered to valid z" subset would have silently rendered as AGN-only
   while still being labeled an AGN-vs-non-AGN comparison. Fixed by adding
   first-class NaN-handling to `plot_embedding_by_value` (NaN `value_col`
   entries draw as a fixed gray point with their own legend entry, instead
   of being dropped or miscolored), so the redesigned figure shows the
   full gold population with the coverage gap itself visible.
2. **UMAP1 carries a real redshift gradient among AGN.** Pearson
   r(UMAP1, `combo_Z`) = -0.578 (p≈0, n=9,992); r(UMAP2, `combo_Z`) =
   -0.303 (p=9.7e-211) -- a moderate-to-strong, highly significant
   gradient along the primary UMAP axis, weaker along the second.
3. **High-redshift AGN occupy a distinct, spatially compact sub-region --
   not a chance clustering.** The 999 highest-z AGN have a mean pairwise
   UMAP distance of 3.336, versus a random-same-size-draw distribution of
   mean 4.718 (std 0.078, n=200 draws) -- the true value falls at the
   **0.0th percentile** of that distribution, below every single random
   draw. This is an unambiguous result, not a borderline one.

Full diagnostics, all three figures' subset counts, and the figure-3
redesign rationale are in `results/REDSHIFT_PLOTS.md`.

## 2026-08-04 — Label-SHAP added: gold AGN vs. non-AGN, additive to cluster-SHAP

Built `src/label_shap.py` + `notebooks/label_shap.ipynb` (executed,
outputs stored) + `results/LABEL_SHAP.md` + `tests/test_label_shap.py`.
Pre-label-SHAP state is commit `febbea2` (`refactor: dual-variant
restructure with parallel Variant A / B presentation`) if this ever needs
to be checked out for comparison -- nothing in `src/interpretation.py`,
`analysis.ipynb`, or any other existing file was touched; this was purely
additive.

**Why additive, not a replacement:** the existing cluster-explainer SHAP
(`interpretation.py::train_lightgbm_surrogate` + `compute_shap_values`)
explains HDBSCAN cluster membership, which is partly circular -- clusters
are themselves a function of the feature matrix, so "what defines the
clusters" isn't quite the same question as "what predicts AGN-ness."
Label-SHAP trains directly on the gold label (`agn_positive` vs
`agn_negative` from `preprocessing.build_labels`), with silver and
unlabeled rows excluded from training entirely (silver membership is its
own features-derived heuristic, so including it would just reintroduce a
different circularity). Both feature-space variants (A: 22-feat/25,801
rows, B: 32-feat/22,417 rows) were run independently; this was also the
first driver call anywhere in the repo for
`build_feature_matrix_with_fluxes_and_variability` (Variant B), which
`STATE.md` §3 had flagged as defined but never exercised.

**Three findings that surprised us:**

1. Both variants' final models converge in only 14-17 boosting rounds
   (CV per-fold `best_iteration` range 12-20 of a 500-round budget) --
   the gold AGN/non-AGN boundary is close to trivially separable in this
   feature space, not a hyperparameter artifact (the same config does
   *not* converge quickly on the synthetic-noise test fixture).
2. Variant A vs. Variant B's differing top-10 SHAP features turned out to
   be almost entirely a **sample effect, not a feature effect** --
   restricting Variant A to exactly Variant B's 22,417-row index (a clean
   index intersection, no row mismatch) reproduces Variant A's own
   full-sample ranking almost exactly, while Variant B's extra 10 columns
   only pull in 2 of them (`log_flux_aper_m`, `log_flux_combo_r`) at real
   weight without reshuffling the rest.
3. Despite average precision of 0.994-0.996 in every condition, the
   0.5-threshold confusion matrix is much less flattering: only 34.0% of
   true non-AGN (Variant A) are classified correctly at that fixed
   threshold. `scale_pos_weight` helps but doesn't erase this -- most of
   the AP score is about correctly ordering probabilities, not about the
   default threshold splitting the classes evenly.

Full metrics tables, top-20 feature lists per variant, and the
circularity cross-check against `build_labels`' silver-score inputs
(Mateos wedge, legacy X-ray/optical ratio, CSC spectral-fit presence,
`high_quality_z`) are in `results/LABEL_SHAP.md`.

## 2026-07-24 — Baseline capture abandoned; not required by the final restructure

Attempted to run capture_baseline.py as pre-migration reference for the
src/ restructure. Debugged through two real issues this session (HDBSCAN
get_params() subclass crash resolved by patching .fit instead of __init__;
cells 80/81 excluded due to SHAP shape bug). Last executed run crashed with
NameError in the "Regular Training Procedure" section (references a
variable from a different code path than the one selected for the Reduced
Case run). At that point scope narrowed to the flux+UMAP+HDBSCAN+SHAP
pipeline directly, and reproduction was verified against v_final results
within analysis.ipynb itself (see SHAP_SUMMARY.md's Reproducibility
section). Baseline JSON never produced. Script preserved for future use if
pre-migration reference becomes needed.

## 2026-07-24 — Dropped "Previous UMAP algorithm" section (cells 90-113) during `src/agn_classification/` migration

**What it was:** `unsupervised.ipynb` cells 90-113 (markdown headers "Previous
UMAP algorithm" / "MISSC") contained a second, independent UMAP fit
(`reducer15`, fixed `n_neighbors=15`) and a second, independent HDBSCAN fit
(`min_cluster_size=80, min_samples=20`, run directly on the raw `embedding`
rather than the scaled feature matrix), feeding a separately-named `umap_df`
DataFrame used only by that section's own set of plots (including cells
97-100, four near-identical scatter plots that are themselves a copy-paste
progression of each other).

**Why it's being dropped:** This section duplicates, at a fixed single
configuration, what the cells 36-65 pipeline already does more thoroughly (a
4-way `neighbors_list` UMAP sweep, an 8-config HDBSCAN grid search scored by
silhouette, and a properly scaled/imputed feature matrix). No downstream cell
outside this section depends on `umap_df`, `reducer15`, or this section's
`clusters`/`clusterer` variables. It reads as an earlier iteration of the
notebook's UMAP+clustering approach that was superseded by cells 36-65 but
left in place rather than deleted.

**Disposition:** Not migrated into `src/agn_classification/`. The code is not
deleted from git history — it remains fully recoverable from commit
`b8be875e9d84b7d36dfdca76f0a2b276f08044b3` (2026-07-23), the last commit
before this migration began, by checking out `notebooks/unsupervised.ipynb`
at that revision:

```bash
git show b8be875e9d84b7d36dfdca76f0a2b276f08044b3:notebooks/unsupervised.ipynb
```

See `PLAN.md` §1 (open question 2) for the original proposal and resolution,
and `PLAN.md` §3 for the full per-cell inventory this decision is drawn from.

## 2026-07-24 — Deleted cell 142 ("Multi-iterative model") from migration

Deleted cell 142 (Multi-iterative model) from migration. See
`references/notes/cell_142_analysis.md` for full trace. Loop was
non-functional (raised `ValueError` on iteration 1 due to the `X_labeled`
overwrite bug at `X_labeled = df.copy()`) and its outputs were not consumed by
any downstream cell (cell 143 reads `df["score_supervised_lgbm"]` set in cell
140). Existing bagging methods (`pseudo_positive_bagging`, `bagging_pnu_scores`)
provide the semi-supervised functionality; iterative self-training, if
reintroduced later, should be designed fresh rather than patched from this
cell.

Decision: do not port cell 142 into `classification/semi_supervised.py` in any
form, and do not attempt to repair or reconstruct its apparent intent. Reasons
recorded at decision time: (1) `pseudo_positive_bagging` and
`bagging_pnu_scores` already implement a defensible semi-supervised approach
(bagging over bootstrap samples with pseudo-label injection); (2) iterative
self-training is a distinct methodology (sequential retraining with
confidence-thresholded pseudo-labels) requiring independent design decisions
— confidence threshold schedule, promotion weight, stopping criterion,
evaluation protocol under label drift — none of which the broken cell
provides, and reconstructing it correctly was judged not worth the time
against the migration's scope; (3) if iterative self-training is added later,
it should be designed fresh from Lee (2013) "Pseudo-Label" or the
consistency-regularization literature, not reverse-engineered from this
cell's intent. Recoverable at commit
`b8be875e9d84b7d36dfdca76f0a2b276f08044b3` if ever wanted for reference.
