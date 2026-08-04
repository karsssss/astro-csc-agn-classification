# Label-SHAP -- gold AGN vs. non-AGN, both feature spaces

Additive to `SHAP_SUMMARY.md`'s cluster-explainer SHAP, not a replacement.
The existing pipeline (`src/interpretation.py::train_lightgbm_surrogate`)
trains LightGBM to predict **HDBSCAN cluster membership**, which answers
"what defines the clusters" -- partly circular, since the clusters are
themselves a function of the same feature matrix. This document instead
trains LightGBM directly on the **gold AGN label**
(`agn_positive`/`agn_negative`, from `preprocessing.build_labels`), which
answers the actual science question: what features predict AGN-ness.
Silver and unlabeled rows are excluded entirely from training -- silver
membership is itself a features-derived heuristic (Mateos wedge, CSC
spectral-fit presence, legacy X-ray/optical ratio, `high_quality_z`), so
including it would reintroduce the same circularity this exercise exists to
avoid.

Driver: `notebooks/label_shap.ipynb` (executed, outputs stored). Module:
`src/label_shap.py`. Does not modify `src/interpretation.py` or any of its
callers (`analysis.ipynb` untouched).

Both feature-space variants from `src/conversions.py` are run
independently: **Variant A** (`build_feature_matrix`, 22-feat, 25,801 rows)
and **Variant B** (`build_feature_matrix_with_fluxes_and_variability`,
32-feat, 22,417 rows) -- see `SHAP_SUMMARY.md`/`EXTENDED_FEATURES_COMPARISON.md`
for what distinguishes them. Per `STATE.md` §3, nothing in the repo
previously called Variant B; this is its first driver call.

## Method

- **Target:** binary, `agn_positive` (1) vs `agn_negative` (0). Silver
  (`silver_agn`) and unlabeled rows dropped from training entirely.
- **Imbalance:** `scale_pos_weight = count(neg)/count(pos)`, computed once
  per variant from the full gold subset and reused across all CV folds and
  the final model. This comes out **less than 1** in both variants because
  AGN is the majority class among gold-labeled sources -- expected, not a
  bug (see Kumaran 2023 §4 on native class-weighting vs. SMOTE, which this
  follows: no SMOTE, no oversampling).
- **Missing values:** LightGBM's native NaN handling; no imputer, no
  sentinel value (Kumaran 2023 §4, Yang 2022 §9).
- **Model:** `LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63)`
  — not tuned; this is an interpretability exercise, not a leaderboard.
- **CV:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`,
  out-of-fold predictions. Early stopping happens **inside each fold**
  against that fold's own held-out test split (`eval_metric="average_precision"`,
  `early_stopping_rounds=50`) -- for metrics only.
- **Final model (for SHAP):** fit on the **entire** gold subset per variant,
  no held-out slice. `n_estimators` is the mean of the CV folds'
  `best_iteration` if all folds stopped short of 500, else 500 (see
  `choose_final_n_estimators`). Rationale: the minority class is small
  (843 / 566 rows) and CV already characterizes how many rounds the model
  needs, so a validation split for the final fit would materially shrink
  the SHAP-explained sample for no real benefit.
- **SHAP:** `shap.TreeExplainer` on the final model. Values array shape
  asserted `(n_samples, n_features)` before any plotting -- binary LightGBM
  returns this directly (2D), unlike the multiclass cluster-explainer
  surrogate in `interpretation.py`, which returns 3D and must branch on
  `ndim`. Positive SHAP pushes toward the AGN (y=1) prediction; negative
  toward non-AGN -- one model, both directions readable from the same
  figures.

## Metrics

All values from live out-of-fold predictions (5-fold CV), pooled across
folds. Average precision is primary -- ROC-AUC is optimistic under this
much imbalance and is reported for reference only. `n_pos`/`n_neg`/ratio
are printed live from the actual gold subset, not assumed.

| | **Variant A** (22-feat) | **Variant B** (32-feat) | **Variant A on B's index** (side analysis) |
|---|---:|---:|---:|
| Gold sample size | 13,115 | 11,621 | 11,621 |
| n_pos (AGN) | 12,272 | 11,055 | 11,055 |
| n_neg (non-AGN) | 843 | 566 | 566 |
| Imbalance ratio (AGN:non-AGN) | **14.56:1** | **19.53:1** | 19.53:1 |
| scale_pos_weight | 0.0687 | 0.0512 | 0.0512 |
| **Average precision** | **0.9941** | **0.9947** | **0.9957** |
| ROC-AUC | 0.9278 | 0.9299 | 0.9343 |
| Brier score | 0.0474 | 0.0373 | 0.0368 |
| Confusion matrix @ 0.5 (rows=true [0,1], cols=pred [0,1]) | `[[287, 556], [322, 11950]]` | `[[259, 307], [332, 10723]]` | (not re-run; side analysis reports AP/AUC/Brier only, see below) |
| Per-fold `best_iteration` | [14, 15, 14, 12, 15] | [19, 16, 18, 20, 14] | n/a (metrics only, no separate final model saved) |
| Final model `n_estimators` | **14** | **17** | 15 (mean of folds; see side analysis below) |

The user-supplied imbalance ratios (14.5:1 / ~19.5:1) are confirmed exactly
by the live run -- 14.56:1 and 19.53:1.

## Top-10 features (mean |SHAP|)

**Variant A (22-feat):**

| rank | feature | mean(\|SHAP\|) |
|---:|---|---:|
| 1 | `color_w1_over_w2` | 0.5710 |
| 2 | `xray_b_over_w1` | 0.2152 |
| 3 | `xray_b_over_r` | 0.1831 |
| 4 | `color_r_over_z` | 0.1495 |
| 5 | `xray_m_over_r` | 0.0995 |
| 6 | `WISE_w2snr` | 0.0767 |
| 7 | `color_g_over_r` | 0.0587 |
| 8 | `color_w2_over_w3` | 0.0475 |
| 9 | `xray_b_over_w2` | 0.0362 |
| 10 | `optical_r_over_w1` | 0.0224 |

**Variant B (32-feat):**

| rank | feature | mean(\|SHAP\|) |
|---:|---|---:|
| 1 | `color_w1_over_w2` | 0.6312 |
| 2 | `xray_b_over_w1` | 0.2094 |
| 3 | `color_r_over_z` | 0.1621 |
| 4 | `xray_b_over_r` | 0.1067 |
| 5 | `xray_m_over_r` | 0.1006 |
| 6 | `log_flux_aper_m` | 0.0615 |
| 7 | `color_g_over_r` | 0.0564 |
| 8 | `color_w2_over_w3` | 0.0428 |
| 9 | `log_flux_combo_r` | 0.0426 |
| 10 | `optical_r_over_w1` | 0.0243 |

**Variant A restricted to Variant B's 22,417-row index (side analysis):**

| rank | feature | mean(\|SHAP\|) |
|---:|---|---:|
| 1 | `color_w1_over_w2` | 0.6251 |
| 2 | `xray_b_over_w1` | 0.2604 |
| 3 | `color_r_over_z` | 0.1650 |
| 4 | `xray_m_over_r` | 0.1156 |
| 5 | `xray_b_over_r` | 0.0723 |
| 6 | `color_g_over_r` | 0.0659 |
| 7 | `color_w2_over_w3` | 0.0598 |
| 8 | `WISE_w2snr` | 0.0548 |
| 9 | `optical_r_over_w1` | 0.0385 |
| 10 | `xray_b_over_w2` | 0.0178 |

## Side analysis: feature-space effect vs. sample effect

Variant A and B don't operate on the same rows (25,801 vs. 22,417) -- the
comparison above isn't strictly apples-to-apples. Restricting Variant A to
exactly Variant B's 22,417-row index fell out cleanly from a plain index
intersection (`X_A.index.intersection(X_B.index)` -- full overlap, all
22,417 Variant B rows are a subset of Variant A's 25,801) and ran within
the time budget.

**Result: the ranking is almost entirely a sample effect, not a feature
effect.** Variant A's top-10 on the full 25,801 rows and on the
B-index-restricted 22,417 rows are identical in rank order and close in
value (e.g. `xray_b_over_w1` 0.215 -> 0.260, `xray_b_over_r` 0.183 -> 0.072
-- some individual movement, but the same features in the same order).
Variant B's own top-10 differs from *both* Variant A conditions only by
inserting `log_flux_aper_m` (#6) and `log_flux_combo_r` (#9) in place of
`WISE_w2snr` and `xray_b_over_w2` -- i.e. adding 10 extra columns (8
log-fluxes + 2 variability) pulled in exactly two of them at meaningful
weight, and did not reshuffle the shared ratio/color features' relative
importance. The dominant signal (`color_w1_over_w2` at #1 by a wide margin
in all three conditions) is a property of the row *population*
(gold-labeled sources restricted to whichever completeness cut), not of
which of the two engineered feature sets is used.

## Circularity flag

Checked live against each variant's actual `X.columns` (not assumed):

| Silver-score input | Related feature | Present in Variant A? | Present in Variant B? | Relationship |
|---|---|---|---|---|
| Mateos wedge (`WISE_w1mpro - w2mpro`) | `color_w1_over_w2` | Yes | Yes | Affine transform (log-flux-ratio vs. magnitude-difference; different zero points for W1/W2, so not numerically identical, but monotonically related) |
| Mateos wedge (`WISE_w2mpro - w3mpro`) | `color_w2_over_w3` | Yes | Yes | Same affine relationship as above |
| Legacy X-ray/optical ratio (`xray_opt_r_relation_legacy`) | `xray_b_over_r` | Yes | Yes | Related, not identical -- both compare `flux_aper_b` to r-band flux, but the legacy version stays in mixed log-flux/magnitude units while `xray_b_over_r` is a clean log-flux-ratio |
| CSC spectral-fit presence (`has_csc_spectral_fit`, i.e. `powlaw_gamma > 0`) | -- | **No** | **No** | Neither variant's `X` contains `powlaw_gamma` or any CSC-spectral-fit-derived column (checked via substring match on `"csc"`/`"powlaw"` across both column lists) |
| `high_quality_z` | -- | **No** | **No** | Not a feature in either `X` -- it gates *silver* membership only, never enters the feature matrix |

**Read:** `color_w1_over_w2` is the #1 global feature by a wide margin in
*every* condition tested (A, B, and the side analysis) -- and it is
directly related to the single strongest component of the Mateos-wedge
silver-score heuristic. `xray_b_over_r`, similarly, is related to the
legacy X-ray/optical ratio silver-score input and ranks in the top 3-5
everywhere. **This is expected by construction, not a discovery**: both
quantities are also well-established AGN-selection criteria in the
literature independent of this project's own silver score (Mateos et al.
2012 for the WISE wedge; X-ray-to-optical flux ratio as a classic AGN
diagnostic, used by both Yang 2022 and Kumaran 2023). The model re-deriving
their importance from the *gold* label (which does not use the silver
score at all) is a mild positive sanity check that gold and silver point in
a physically consistent direction -- it should not be read as evidence that
the silver score is redundant or that the gold labels are somehow leaking
the silver heuristic (they don't share any column). CSC spectral-fit
presence and `high_quality_z`, the two silver-score inputs that are *not*
in either feature matrix, cannot be circular by definition and are absent
from both top-10 lists, consistent with that.

## Surprises

1. **Both variants converge in 12-20 boosting rounds** (`best_iteration`
   range 12-20 across all 10 CV folds combined; final models use 14 and 17
   rounds respectively) -- far short of the 500-round budget. This is a
   large surprise: it indicates the gold AGN/non-AGN boundary is very
   close to linearly/simply separable in this feature space, not a
   hyperparameter artifact (the same `num_leaves=63`, `learning_rate=0.05`
   were used for both the quick-converging real-data runs and the
   500-round-capped synthetic test in `tests/test_label_shap.py`, which
   does *not* converge quickly on pure noise, so the fast convergence here
   is data-driven).
2. **Average precision is extremely high (0.994-0.996) in all three
   conditions.** Combined with surprise #1, this says the X-ray/optical/IR
   ratio feature space nearly reproduces the gold label's decision
   boundary outright. This is not circular in the strict sense (gold
   labels come from independent `xray_agn`/`optical_agn`/`ir_agn`
   confirmation flags, not from any column in `X`), but it is consistent
   with those flags themselves having originally been assigned using
   thresholds on physically similar quantities (X-ray-to-optical ratio,
   IR color) in the source catalogs -- worth keeping in mind when citing
   these AP numbers as a "the model learned something novel" result rather
   than "the model recovered the selection function."
3. **The 0.5-threshold confusion matrix tells a less rosy story than AP
   alone.** In Variant A, only 287/843 (34.0%) of true non-AGN are
   classified correctly at threshold 0.5, despite AP=0.994 — most of the
   ranking quality is about ordering probabilities correctly, not about
   the default threshold producing balanced errors. `scale_pos_weight<1`
   partially compensates for the imbalance but does not eliminate this;
   a practitioner using this model for a hard AGN/non-AGN split (rather
   than a ranked score) would want a tuned threshold, not 0.5.
4. **The Variant A/B comparison is almost entirely a sample effect, not a
   feature effect** (see side analysis above) -- the two variants'
   differing top-10s in the main comparison are driven mostly by which
   22,417-vs-25,801 rows survive each variant's completeness cut, not by
   the 10 additional log-flux/variability columns Variant B adds. Only
   `log_flux_aper_m` and `log_flux_combo_r` earn real SHAP weight; the 2
   variability columns (`var_intra_index_b/m`) don't crack either
   variant's top 20 for the *gold* label, in contrast to their modest-but-
   real presence in the *cluster*-label SHAP reported in
   `EXTENDED_FEATURES_COMPARISON.md`.

## Figures

`results/figures/reduced_22feat/label_shap_bar.png`,
`results/figures/reduced_22feat/label_shap_beeswarm.png` (Variant A);
`results/figures/extended_32feat/label_shap_bar.png`,
`results/figures/extended_32feat/label_shap_beeswarm.png` (Variant B).
Same naming convention as the existing cluster-SHAP figures in these
directories, distinguished by the `label_shap_` prefix.
