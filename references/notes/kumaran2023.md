# Kumaran et al. 2023 — Automated Classification of Chandra X-ray Point Sources Using ML Methods

**Citation:** Kumaran, S., Mandal, S., Bhattacharyya, S., & Mishra, D. 2023, MNRAS, 520, 5065. https://doi.org/10.1093/mnras/stad414
**File:** `references/kumaran2023_csc_lightgbm_multiwavelength.pdf` (11 pp.)
**One-line summary:** Classifies CSC 2.0 point sources into 8 classes (AGN, STAR, YSO, HMXB, LMXB, ULX, CV, PULSAR) comparing RF/GBDT/LightGBM; **LightGBM wins** (93% precision/recall/F1, MCC 0.91) and is chosen specifically because it handles missing MW data natively and needs only class-weighting, not SMOTE. Directly relevant methodological sibling to [[yang2022]] — same CSC 2.0 + MW cross-match problem, different classifier and imbalance/missing-data strategy.

## 1. Classes and training set (contrast with Yang)

Classes: **AGN, STAR** (merges Yang's HM-STAR/LM-STAR into one), **YSO, HMXB, LMXB, ULX** (not in Yang), **CV, PULSAR** (Yang's "NS" class, ATNF+Fermi LAT pulsars only — no magnetars/isolated NS category). No silver/gold distinction — single training set from literature catalogs (VERONCAT for AGN, SKIFF stellar spectral catalog for STAR, Spitzer YSO catalogs, several HEASARC HMXB/LMXB catalogs, ULXRBCAT, Open CV Catalogue, ATNF+4FGL for pulsars), cross-matched to CSC 2.0 at a flat **1″ radius** (vs. Yang's PU-based 2σ combined-error circle).

- Training set: **7703 sources** — AGN 2395 (31%), STAR 2790 (36%), YSO 1149 (15%), HMXB 748 (10%), LMXB 143 (2%), ULX 211 (3%), CV 166 (2%), PULSAR 101 (1%). AGN+STAR+YSO = ~80% of the training set (similar dominance pattern to Yang's AGN+YSO).
- Applied to **269,366** unidentified point sources (out of 277,069 total point sources selected from CSC 2.0's ~317,000 via `extent_flag==0` + quality flags).

## 2. Features and MW catalogs — where this differs from Yang

**41 total features** (Table 2), narrower per-band coverage than Yang but broader catalog coverage:

- **X-ray:** `gal_l2`/`gal_b2`, five ACIS-band fluxes (b/u/s/m/h-csc — note Kumaran keeps **ultrasoft (u)** as a separate band, which Yang folds into soft), plus intra-/inter-observation variability — but **only in the b-band** (`var_inter_prob_b`, `var_inter_sigma_b`, `var_inter_index_b`, `var_intra_prob_b`, `ks_intra_prob`, `kp_intra_prob_b`, `var_intra_index_b`). Yang computes variability per-band across all bands; Kumaran restricts to broadband only.
- **MW catalogs cross-matched:** Gaia-EDR3 (G, Bp, Rp), **SDSS DR16** (u,g,r,i,z bands — new vs. Yang), **GALEX** (FUV, NUV — new vs. Yang, adds near/far-UV), WISE (**W1–W4**, i.e. keeps W4 which Yang drops for shallow depth/large PU), **MIPS-Spitzer 24 μm** (new vs. Yang), 2MASS (J, H, Ks).
- **Colors:** only **9 specific pre-selected colors** (B−R, G−J, G−W2, Bp−H, Bp−W3, Rp−K, J−H, J−W1, W1−W2) rather than Yang's full pairwise combinatorial set — a much smaller, hand-picked color list.
- MW data pulled via NED (cross-matched using MatchEx, 80% of CSC sources found already in NED) using `astroquery`; per-catalog match fractions: Gaia-EDR3 60%, 2MASS 55%, MIPS-Spitzer 41%, WISE 43%, SDSS 24%, GALEX 17%.
- **No extinction/absorption correction of any kind is applied** — unlike Yang, there is no SFD-reddening step for AGNs or anyone else in the training set. This is a notable methodological gap relative to Yang and worth flagging as unaddressed here.

### Mapping to our column-naming scheme

| Kumaran feature | Our closest column(s) | Notes |
|---|---|---|
| `gal_l2`, `gal_b2` | `gal_l`, `gal_b` | direct match |
| `b/u/s/m/h-csc` fluxes | `flux_aper_{b,u,s,m,h}` | direct match; we also carry lo/hi limits they don't use |
| `var_*_b` (b-band only) | `var_intra_index_b`, `var_inter_index_b` (we have all bands `{b,h,m,s,u,w}`) | we're closer to Yang's per-band scheme here |
| Gaia `G, Bp, Rp` | `GAIA21P_g` only | **we don't currently carry Bp/Rp** — a gap if we want their G−J/Bp−H/Bp−W3/Rp−K colors |
| SDSS `u,g,r,i,z` | `SDSS_CALIBFLUX_{u,g,r,i,z}` | matches on survey (SDSS, not PS1) — note our `PS21P_*` (Pan-STARRS) is a *different* optical survey than Kumaran's SDSS features; don't conflate the two when comparing feature importance |
| WISE `W1,W2,W3,W4` | `WISE_w1mpro, w2mpro, w3mpro` | **we don't carry W4** (same omission as Yang, for the same depth/PU reasons) |
| GALEX FUV/NUV | *(none)* | we have no UV cross-match at all — a catalog we could add |
| MIPS 24 μm | *(none)* | we have no Spitzer/MIPS cross-match — another catalog gap |
| 2MASS `J,H,Ks` | `2MASS21P_j_m, h_m, k_m` | direct match |
| `W1-W2` color | `WISE_W1_W2` | direct match |
| `J-W1`, `G-J`, `G-W2` colors | *(not currently built)* | optical/NIR-to-IR colors we don't compute; cheap to add given we have the base magnitudes |
| — | `LEG21P_*` (DESI Legacy Survey), `AGNDB_*` | **neither paper uses these** — our Legacy Survey photometry and external AGN-database cross-match are unique to our pipeline, not benchmarked by either reference |

## 3. Classifier comparison and LightGBM configuration

Tested: Multi-Layer Perceptron, KNN, RF, GBDT, **LightGBM** (all via scikit-learn except LightGBM itself). RF and GBDT require complete data (mode-imputation for missing values) and SMOTE; **LightGBM is chosen as final model specifically because it avoids both**.

- **Hyperparameter tuning:** unlike Yang (who reports explicit RF values — `n_estimators=100`, `max_depth=None`, `max_features=√n`, and shows performance is largely insensitive to them), **Kumaran does not disclose specific LightGBM hyperparameter values** — tuning was done via **FLAML** (Wang et al. 2019, an AutoML tuning library) and the paper reports only the resulting scores, not the final `num_leaves`/`learning_rate`/`n_estimators`/etc. This is a gap to be aware of if trying to replicate their exact setup — we'd need to run our own FLAML/Optuna sweep rather than copy their config.
- **Data normalization:** min-max scaled to **[0, 10]** (Eq. 2: `X_norm = 10 × (X − min)/(max − min)`) — different from Yang's z-score standardization. Min-max is somewhat unusual for tree models (which are scale-invariant) but harmless; mainly matters for the KNN/MLP baselines they also tested.

## 4. Missing-value handling: native LightGBM vs. imputation+SMOTE — the key methodological divergence from Yang

- MW coverage is highly uneven (Fig. 1): e.g. availability is high for AGN/STAR/YSO across most catalogs but very low for X-ray binaries in 2MASS/MIPS/WISE (faint at those wavelengths in quiescence); X-ray variability features are more available for XRBs than other classes.
- **RF/GBDT path:** missing values are mode-imputed per column, which they explicitly say "may not give satisfactory results for a high percentage of missing values, and in some cases is counter-productive... the missing values themselves may be important features" (e.g., an XRB's lack of optical/IR counterpart is itself informative — same argument Yang makes for isolated NSs).
- **LightGBM path:** uses LightGBM's native missing-value handling (**Block Propagation**, Josse et al. 2019) — at each split, only the non-missing samples for that feature are used to decide the split, and any sample missing that feature is routed to whichever branch minimizes the final loss. **No imputation of any kind is performed for the LightGBM model.** This is the paper's stated reason for preferring LightGBM over RF/GBDT, not just raw accuracy.
- This directly supports the option already flagged in the Yang note: **CatBoost/LightGBM can consume NaN natively**, and Kumaran is a working example of skipping imputation entirely rather than using a sentinel value (Yang's −100) or SimpleImputer(median) (our current notebooks) — worth trying as an ablation in our own pipeline.

## 5. Class imbalance: class-weight vs. SMOTE

- RF/GBDT: SMOTE applied to the **training fold only** (not validation), oversampling every minority class up to the majority class (AGN/STAR/YSO) count — same mechanism as Yang.
- **LightGBM: no SMOTE.** Because LightGBM works with missing data directly and SMOTE cannot operate on incomplete feature vectors, they instead use LightGBM's **class-weight** technique — higher weight assigned to minority-class samples in the cross-entropy loss (Eq. 1) so each class contributes equally regardless of sample count. This is exactly the alternative flagged in the Yang note (`class_weight`/`scale_pos_weight`-style approaches) — Kumaran is direct evidence it works about as well or better than RF+SMOTE for the majority classes, though **neither method fixes minority-class performance for LMXB/ULX/CV/PULSAR** — both papers converge on the same conclusion that severe class imbalance (≤3% of training data) isn't solved by either SMOTE or reweighting alone.
- RF SMOTE ablation (their Fig. 2): SMOTE improves pulsar recall from 18%→45% and gives ~10% gains for other minority classes, but does not reach a "satisfactory level" for LMXB/ULX/CV/PULSAR — consistent with Yang's finding that SMOTE helps but doesn't solve fundamentally data-starved classes.
- Confusion patterns (RF, Fig. 2): **without** upsampling, most CV and pulsar sources are misclassified as AGN/STAR/YSO (the majority classes "absorb" minority-class predictions, the classic imbalance failure mode); **with** SMOTE this contamination at the bottom-left of the confusion matrix shrinks noticeably, though ULX still gets misclassified as AGN 22% of the time (vs. 9% for LightGBM with all features, see §8) and CV/pulsar remain heavily cross-confused with each other.

### Does including minority classes hurt the majority classes?

They also trained a **4-class-only model** (AGN, YSO, STAR, HMXB — Table 8) to test whether the presence of minority classes (LMXB/ULX/CV/PULSAR) drags down majority-class performance. Comparing to the 8-class model (Table 7): majority-class F1 improves only marginally (~1 pt for AGN/STAR/YSO, ~2 pts for HMXB) when minority classes are excluded entirely. **Conclusion: keeping minority/rare classes in the training set costs almost nothing for the well-populated classes** — directly relevant to whether our `silver_agn`/unlabeled sources should be included alongside gold `agn_positive`/`agn_negative` when training rather than trained separately: this result suggests it's safe to keep everything in one model rather than fearing that noisy minority labels will degrade AGN/non-AGN performance.

## 6. Validation strategy

**Cumulative k-fold cross-validation (CCV)**, a custom variant: 20-fold, predictions on each held-out fold accumulated into one prediction table across all folds, then precision/recall/F1/MCC computed from that combined table — different from Yang's LOOCV (N-fold) but serves the same "unbiased validation despite small minority classes" purpose. 15 repeated iterations (different random seeds) used to estimate the spread/stability of scores.

## 7. Performance — weighted average (not macro, unlike Yang) across models

| Score | RF | GBDT | LightGBM |
|---|---|---|---|
| Precision | 90.7 | 91.3 | **93.2** |
| Recall | 90.0 | 89.0 | **93.2** |
| F1 | 90.3 | 89.9 | **93.2** |
| MCC | 0.87 | 0.86 | **0.91** |

Per-class LightGBM precision/recall/F1 (all-features model, %): AGN 96.8/97.6/97.2, STAR 96.0/95.7/95.9, YSO 92.7/95.4/94.1, HMXB 91.6/90.6/91.2, **LMXB 94.8/80.9/87.2** (high precision, lower recall), ULX 72.2/71.1/71.6, CV 61.5/55.3/58.2, **PULSAR 42.1/44.2/43.7** (worst class, same "smallest/most-heterogeneous class performs worst" pattern as Yang's LMXB).

- Unlike RF/GBDT (which show a large precision/recall gap for minority classes, e.g. RF pulsar precision 35.8% vs. recall 47.1%), **LightGBM's precision and recall track much more closely for every class** — its class-weighting balances the trade-off better than SMOTE-based oversampling does for RF/GBDT.
- Cumulative-probability plot (their Fig. 4) shows LightGBM is also the best-*calibrated*/most-confident model: only ~500/7703 training sources fall below CMP 0.8, vs. RF being the least confident of the three.

## 8. MW-feature ablation (all-features vs. X-ray-only, LightGBM)

Removing all MW features and keeping only X-ray properties: **pulsar F1 drops sharply, 43%→22%** (their single biggest MW-dependence result) and ULX F1 drops ~20%; **AGN/STAR/YSO lose only ~5 points** (93%→~88-90%) since these classes already separate well on X-ray alone; **HMXB/LMXB are nearly insensitive to removing MW features** (X-ray binaries are intrinsically faint at optical/IR, so those bands carry little signal for them either way — consistent with their high native-missingness for these classes noted above). Confirms that **not all classes benefit equally from MW cross-matching**, which is relevant when arguing for/against including a costly cross-match (e.g. GALEX at only 17% match fraction) for a specific target class like AGN.

## 9. Confidence definition: CMP thresholds vs. Yang's CT

Much simpler than Yang's MC-derived CT (§7 of the Yang note): **Class Membership Probability (CMP)** is just the raw LightGBM predicted-class probability, no Monte Carlo resampling of feature uncertainty and no probability-uncertainty term. They build a per-class **CMP probability density function** from the histogram of all classified sources and pick "confident" thresholds directly off that PDF: **3σ ≡ CMP > 0.997** and **4σ ≡ CMP > 0.9999** (terminology borrowed from Gaussian-tail language, but there's no actual Gaussian assumption or per-source uncertainty estimate — it's a fixed global cut on the single-point probability). Applied to the 269,366 unidentified sources: 54,770 sources classified at 3σ (8574 AGN, 5166 STAR, 208 YSO, 46 HMXB, 71 LMXB, 0 ULX, 1 CV, 0 pulsar), 14,066 at 4σ.
- This is a materially weaker confidence notion than Yang's CT (no accounting for measurement-uncertainty-driven probability spread), but far cheaper to compute — no 1000× MC resampling per source. Worth deciding explicitly whether our pipeline needs Yang-style uncertainty-aware CT or whether a Kumaran-style fixed-CMP-percentile cut is sufficient for our AGN/silver/non-AGN 3-class problem.

## 10. Implications for our pipeline (LightGBM + CatBoost)

- **Strongest actionable point:** try LightGBM/CatBoost's **native NaN handling with no imputation at all**, rather than our current `SimpleImputer(median)` + missing-indicator-column approach — this paper is a working precedent that it outperforms impute+SMOTE on the same CSC 2.0 problem.
- **Second: prefer class-weighting over SMOTE** for our AGN/silver/non-AGN imbalance (`scale_pos_weight` in LightGBM, `auto_class_weights`/`class_weights` in CatBoost) — both this paper and Yang converge on SMOTE being marginally helpful but not sufficient for small classes, and Kumaran shows class-weight+native-missingness gets comparable or better precision/recall balance without SMOTE's synthetic-interpolation risk.
- **Gap they share with us:** no extinction/absorption correction — if we adopt Yang's SFD-based AGN dereddening (flagged in the Yang note), we'd be doing something neither of these MW-classification papers on CSC combines with LightGBM/GBDT; worth noting as a potential novel contribution rather than just replicating either paper.
- **Catalogs to consider adding:** GALEX (UV) and MIPS-Spitzer (mid-IR 24 μm) are cross-matched here but not in Yang or our current pipeline — low match fractions (17%, 41%) mean limited coverage gain, but Kumaran's ablation shows MW features matter most for exactly the classes (pulsar/ULX-analog "weird" sources) we'd want silver-AGN candidates not to be confused with.
- **Hyperparameters are not directly reusable** from this paper (FLAML-tuned, values undisclosed) — plan to run our own tuning sweep (e.g. Optuna) rather than trying to replicate their LightGBM config verbatim.
