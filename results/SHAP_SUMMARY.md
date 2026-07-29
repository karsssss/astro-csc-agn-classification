# SHAP Summary -- UMAP + HDBSCAN + LightGBM + SHAP, Diagnostic History, and Dual-Variant Result

This document covers two parallel, independently-reproducible clustering
variants built on the same CSC Reduced Case starting columns: **Variant A**
(the canonical 22-feature "v_final" result, ratios/colors/hardness/SNR
only, zero raw flux or magnitude columns) and **Variant B** (a 32-feature
extension that adds back log-flux and X-ray variability columns as a
robustness check). Variant A is documented via
`src/conversions.py::build_feature_matrix`; Variant B via
`build_feature_matrix_with_fluxes_and_variability` in the same module
(both functions are independent, unmodified entry points -- neither
supersedes the other).

Variant A is the canonical result for downstream use (25,801 rows, 7
clusters). Variant B (22,417 rows, 4 clusters) exists specifically to test
whether dropping raw fluxes was necessary and whether X-ray variability
resolves finer sub-structure -- see "Comparison and interpretation" below
for the answer, and `results/EXTENDED_FEATURES_COMPARISON.md` for the full
methodological detail behind Variant B (coverage diagnostics, the
X-ray-band exclusion rationale, and a row-level crosswalk between the two
clusterings' memberships).

## Diagnostic history (v1 -> v3)

v1 (flux-converted `log_flux_*` features from the notebook's existing
Reduced Case columns) collapsed to 2 clusters dominated by absolute
IR/optical brightness (`log_flux_WISE_w1mpro`, `combo_log_flux_z`). v2 (81
candidate brightness-normalized ratio/color/hardness/variability columns,
complete-case dropna) removed the brightness dominance but complete-case
filtering across that many ratio columns collapsed the sample 27,537 ->
1,731 rows (94% loss, driven mainly by rare Gaia x-ray cross-matches), and
a single `xray_ratio_*_GAIA21P_g` pair took over as the new dominant
feature. v3 (same v2 columns, column-drop + row-coverage-filter +
median-impute instead of dropna) recovered 22,357 rows but its HDBSCAN
grid timed out before a verdict was reached; it was abandoned in favor of
a from-scratch rebuild once the row-completeness diagnosis was clear (see
below). Full numeric detail for v1 and v2 is preserved in the "Superseded"
section at the bottom of this file. Variant A (below) is that
from-scratch rebuild's result.

---

## Variant A -- Reduced 22-feature space (canonical, "v_final")

### Reproducibility

- **HDBSCAN config selected:** `mcs_20_ms_5_leaf` -- `min_cluster_size=20`, `min_samples=5`, `cluster_selection_method="leaf"`, `metric="euclidean"`, `core_dist_n_jobs=-1`.
- **Seeds:** `StandardScaler` (no seed; deterministic). UMAP: `n_neighbors=15, min_dist=0.1, n_components=2, metric="euclidean", random_state=42`. `silhouette_score`: `sample_size=min(10_000, n_valid), random_state=42`. `train_test_split`: `test_size=0.25, random_state=42`, stratified on cluster label. `LGBMClassifier`: `random_state=42, objective="multiclass", class_weight="balanced"`. `TreeExplainer` sample: `X_test.sample(n=min(5000, len(X_test)), random_state=42)`.
- **Row count:** 25,801 (of 27,537 in the sentinel-corrected pre-filter population).
- **Final feature column count:** 22.
- A reviewer should be able to reconstruct every number below from these facts plus `src/conversions.py::build_feature_matrix`, `src/clustering.py`, and `src/interpretation.py` (see the restructure note at the bottom of this file).
- **Reproduction:** a fresh execution of `analysis.ipynb` from the `src/` modules reproduces the v_final result at every scientifically meaningful level. Row count (25,801), final feature column count (22), HDBSCAN config selection (`mcs_20_ms_5_leaf`), cluster count (7 real + noise), silhouette (0.4326 vs. v_final's 0.432), and Cluster 1 (n=136, 58.8% AGN, 27.2% non-AGN, driven by `optical_r_over_w1` and `color_w1_over_w2`) all match. A 1-source drift is observed between noise (12,995 reproduced vs. 12,996 v_final) and Cluster 4 (86 reproduced vs. 85 v_final), root cause unidentified; `core_dist_n_jobs` parallelism was tested at both settings and produces identical results, so that is ruled out as the cause. Top-4 global SHAP features (`optical_r_over_w1`, `color_g_over_r`, `color_r_over_z`, `xray_b_over_r`) are unchanged in rank and value to within ~0.008. Candidates for the 1-row drift include subtle differences between the original inline v_final code and the `build_feature_matrix` module, or execution differences between the original `nbclient` exploration and a bare Python run; neither has been root-caused.

### Approach

Starting strictly from the notebook's own definitive 25-column "Reduced Case" set (13 X-ray aperture-flux columns, `combo_g`/`combo_r`/`combo_z_mag`, 6 WISE columns, `hard_hs`+errors) -- not the larger, more failure-prone candidate lists v2/v3 explored -- optical and WISE magnitudes were converted to flux (AB/Vega zero points via `mag_to_flux_jy`), and a small, deliberately restrained set of ratios and colors was built: broad- and medium-band X-ray-to-multiwavelength ratios (`xray_b_over_r`, `xray_b_over_w1`, `xray_b_over_w2`, `xray_m_over_r`, `optical_r_over_w1`) and four WISE/optical colors. **Soft- and hard-band X-ray ratios were deliberately excluded**, even though they're the most "obvious" ratios to build: requiring `flux_aper_s`/`flux_aper_h` to be strictly positive (not just non-null) discards 14.29%/8.70% of the pre-filter population respectively -- a real, legitimate per-band non-detection, but one that would have reintroduced exactly the brightness/detection selection effect this whole exercise was trying to remove. Hard/soft spectral shape is instead carried by `hard_hs`, and -- since they clear an 85%-coverage bar at negligible additional row cost (25,801 vs 25,803 rows) -- `hard_hm` and `hard_ms` were added too, giving three hardness ratios total (closer to Kumaran/Yang/Perez-Diaz's convention of using multiple hardness ratios rather than one).

**Final column list (22):** `significance`, `WISE_w1snr`, `WISE_w2snr`, `WISE_w3snr`, `hard_hs` (+`lolim`/`hilim`), `hard_hm` (+`lolim`/`hilim`), `hard_ms` (+`lolim`/`hilim`), `xray_b_over_r`, `xray_b_over_w1`, `xray_b_over_w2`, `xray_m_over_r`, `optical_r_over_w1`, `color_g_over_r`, `color_r_over_z`, `color_w1_over_w2`, `color_w2_over_w3`. Zero raw single-band flux/magnitude columns (asserted programmatically). **25,801 rows survive** -- 93.7% of the 27,537-row sentinel-corrected pre-filter population, and comparable in scale to Perez-Diaz (2024)'s working sample.

### HDBSCAN grid

All 3 `min_cluster_size` values (20/30/50) are irrelevant on their own -- the only thing that matters is `cluster_selection_method`: every `eom` config gives the same 2-cluster, 6.5%-noise, silhouette-0.338 result, and every `leaf` config gives the same 7-cluster, 50.4%-noise, silhouette-0.432 result. **`mcs_20_ms_5_leaf` was selected** (arbitrarily, among the 3 identical `leaf` configs) because it's the only family reaching the >=4-real-cluster target.

### Cluster composition (n_members, %AGN [silver+confirmed], %non-AGN, %unlabeled)

| Cluster | n | %AGN | %non-AGN | %unlabeled |
|---|---:|---:|---:|---:|
| -1 (noise) | 12,996 | 66.1 | 4.5 | 29.5 |
| 0 | 54 | 72.2 | 1.9 | 25.9 |
| 1 | 136 | 58.8 | **27.2** | 14.0 |
| 2 | 54 | 66.7 | 9.3 | 24.1 |
| 3 | 12,284 | 89.6 | 1.8 | 8.6 |
| 4 | 85 | 14.1 | 0.0 | 85.9 |
| 5 | 94 | 41.5 | 0.0 | 58.5 |
| 6 | 98 | 34.7 | 0.0 | 65.3 |

**Top 15 global SHAP features:** `optical_r_over_w1` (0.155), `color_g_over_r` (0.089), `color_r_over_z` (0.081), `xray_b_over_r` (0.064), `significance` (0.053), `WISE_w1snr` (0.050), `color_w1_over_w2` (0.039), `xray_b_over_w1` (0.039), `color_w2_over_w3` (0.034), `xray_b_over_w2` (0.028), `hard_ms_hilim` (0.019), `xray_m_over_r` (0.017), `WISE_w3snr` (0.015), `hard_hs` (0.015), `WISE_w2snr` (0.015). **No SNR/error-bar column ranks in the top 5** -- the top 5 are all physical ratios/colors/significance, a materially different (and flatter -- top feature is only ~1.7x the #2 feature, vs. 5-15x in v1/v2) distribution than every prior version.

**Balanced accuracy: 0.982** (not 1.00) -- with 7 real classes of wildly different sizes (12,284 down to 54), this is a more informative number than v1/v2's trivial 1.00, though the same definitional caveat still applies in part: the classifier is trained on the same 22-column feature matrix that UMAP/HDBSCAN derived the labels from.

### Honest astrophysical interpretation, per cluster

- **Cluster 3 (n=12,284, 47.6% of the sample, 89.6% AGN-labeled):** by far the dominant cluster. Its own top feature is `WISE_w1snr` (0.303) -- **a data-quality/SNR proxy, not a physical ratio** -- ahead of `xray_b_over_w1` (0.216) and `xray_b_over_r` (0.177). This doesn't show up in the *global* top-5 SHAP check (which passes) because it's diluted by averaging across all 7 classes, but it means the single largest cluster is partly separated by "how well-detected in WISE W1" rather than purely spectral shape. Worth being explicit about even though it doesn't trip the instructed global-top-5 flag.
- **Cluster 1 (n=136, 58.8% AGN, 27.2% non-AGN -- the highest non-AGN concentration of any cluster):** driven overwhelmingly by `optical_r_over_w1` (0.476) and `color_w1_over_w2` (0.123), both genuinely physical (optical-to-IR SED shape, a classic AGN-vs-galaxy discriminator since AGN tori reprocess more IR flux relative to optical than normal galaxies). This is the closest thing in any version of this analysis to a real, physically-motivated, partially non-AGN-enriched cluster -- small, but not a data-quality artifact. This is the scientifically most interesting cluster in the analysis: small (n=136), but the highest confirmed non-AGN concentration of any cluster, driven by physically motivated optical/IR SED features rather than a brightness or SNR proxy -- a reasonable candidate for follow-up spectroscopic confirmation.
- **Cluster 0 (n=54) and Cluster 6 (n=98):** both driven by `color_g_over_r`/`color_r_over_z` (optical color), likely a related but distinct region of optical color space from cluster 1.
- **Cluster 2 (n=54):** driven by `xray_b_over_r`, `xray_m_over_r`, and `hard_ms` -- the one cluster where X-ray-to-optical ratio and medium-band hardness dominate over IR/optical colors.
- **Clusters 4 (n=85) and 5 (n=94):** both mostly *unlabeled* (85.9% and 58.5%) with low AGN fractions (14.1%, 41.5%) and zero confirmed non-AGN. Cluster 4 is driven by `hard_hm`/`hard_hs` (genuine hardness); cluster 5 by `optical_r_over_w1` and `significance`. These may represent an astrophysical population not well covered by the existing gold/silver AGN labels (candidates: X-ray binaries, stars, or simply low-significance/ambiguous sources) rather than "clustering failure" -- but with n<100 and no ground truth to check against, this is speculative, not confirmed.
- **Noise (n=12,996, 50.4% of the sample):** HDBSCAN's `leaf` selection method achieves 7 clusters partly by aggressively calling half the sample noise. The noise pool's own label composition (66.1% AGN) is unremarkable relative to the clusters -- it isn't a discovered population, it's what's left over.

### Bottom line

**This is a partially physics-driven result, with one clear caveat.** Compared to v1 (single brightness feature dominating everything) and v2 (single Gaia-ratio dominating a collapsed sample), v_final's global SHAP importance is flatter, its top-5 features are physical (colors, X-ray/optical ratio, significance), and it reaches real, differentiated structure (>=4 clusters, non-mirror-image per-cluster SHAP profiles, one small cluster -- Cluster 1 -- with a genuinely elevated non-AGN fraction driven by a physically-motivated optical/IR ratio). That said: (1) the dominant cluster (Cluster 3, 47.6% of the sample) leans on a WISE SNR proxy as its top individual driver, even though this doesn't surface in the global top-5; (2) cluster sizes are extremely imbalanced (one cluster of 12,284 plus six clusters of 54-136, plus 50.4% noise) -- this reads as "one big blob, a large noise pool, and several small distinct outlier groups," not seven comparably-sized populations; (3) the HDBSCAN grid is entirely insensitive to `min_cluster_size` in the tested range and depends entirely on `eom` vs. `leaf`, which is a fairly blunt lever to be resting a >=4-cluster conclusion on.

**What this says about the CSC Reduced Case:** a genuinely richer clustering structure than 2 blobs *is* recoverable from these 25 base columns once brightness/detection-selection effects are properly removed from the feature ratios (not just from the labels) -- the earlier "brightness-normalized clustering doesn't resolve more than 2 groups" hypothesis from the v2/v3 diagnostic phase does **not** hold once soft/hard-band selection bias is also removed from the ratio construction itself, not just from the raw flux columns. Reaching Perez-Diaz (2024)'s 6-cluster resolution with comparably-sized, physically distinct populations (rather than 1 dominant blob + small outlier groups + heavy noise) likely still requires either per-detection variability features not present in this Reduced Case, or further HDBSCAN/UMAP hyperparameter work beyond the 6-config grid tested here -- but the core finding is that this feature-engineering approach (physically-motivated ratios, explicit exclusion of selection-biased ratios) is sound and worth carrying into the `src/` restructure, not a dead end.

### Post-hoc visualization: PaCMAP, label overlay, and feature distributions

**(a) PaCMAP robustness check** (`reduced_22feat/umap_vs_pacmap.png`, `reduced_22feat/pacmap_only.png`). Same feature matrix and the same HDBSCAN cluster labels, plotted with a matched color mapping across both embeddings. The six small clusters (0, 1, 2, 4, 5, 6; sizes 54-136) render as tight, separated blobs in both UMAP and PaCMAP, giving weak-but-meaningful evidence that they aren't a UMAP-specific artifact. Cluster 3 (n=12,284) and the noise pool (n=12,995) render cleaner in UMAP than in PaCMAP, where the two are more visually interleaved.

**(b) Label overlay validation** (`reduced_22feat/umap_by_label.png`). Ground-truth label distribution in the v_final 25,801-row sample: 12,272 confirmed AGN (2), 7,514 silver AGN (1), 843 confirmed non-AGN (0), 5,172 unlabeled (-1). Cluster 1's 27.2% confirmed-non-AGN rate (~37 of 136 sources) represents roughly a 9x enrichment relative to the sample-wide non-AGN rate (~3.3%). In the UMAP colored by label, confirmed non-AGN appear as a concentrated vertical band around Dim 1 = 0-2, Dim 2 = 2-6, rather than a clean single blob -- expected for a ~9x relative enrichment against this degree of class imbalance. The concentration survives brightness suppression (the feature matrix contains zero raw flux/magnitude columns; `test_build_feature_matrix_has_no_raw_flux_columns` enforces this invariant), strengthening the physical interpretation.

**(c) Per-cluster feature characterization** (`reduced_22feat/feature_distributions_by_cluster.png`). Clusters 2 and 3 sit high on X-ray-to-optical/IR ratios (`xray_b_over_r`, `xray_b_over_w1`); Cluster 4 sits low on them; Clusters 4/5/6 are elevated in `color_w2_over_w3` (IR-red). The distribution plot also visually confirms Cluster 3's `WISE_w1snr` caveat from the "Honest astrophysical interpretation" section above: Cluster 3 sits below the noise median on `WISE_w1snr`, meaning its identity partly reflects poorer WISE W1 detections rather than purely spectral physics.

Per-cluster SHAP bar charts: `reduced_22feat/shap_cluster_0.png` through `shap_cluster_6.png`. Global SHAP: `reduced_22feat/shap_global.png`. Cluster composition bars: `reduced_22feat/cluster_composition.png`.

---

## Variant B -- Extended 32-feature space (robustness check)

Built via `build_feature_matrix_with_fluxes_and_variability`, testing two
questions: (1) was dropping raw flux/magnitude columns after computing
ratios actually necessary, and (2) do X-ray variability features resolve
v_final's mostly-unlabeled small clusters into finer, physically-distinct
sub-populations. Full setup detail, coverage diagnostics, and the
X-ray-band exclusion rationale live in
`results/EXTENDED_FEATURES_COMPARISON.md`; this section summarizes the
authoritative result numbers.

### Setup

- **Row count:** 22,417 (of the same 27,537-row sentinel-corrected pre-filter population; -13.1% vs. Variant A's 25,801, driven by the added columns' coverage requirements).
- **Final feature column count:** 32 = Variant A's 22 + 8 log10(flux) columns (`log_flux_aper_b`, `log_flux_aper_m`, `log_flux_combo_g/r/z`, `log_flux_wise_w1/w2/w3`) + 2 variability columns (`var_intra_index_b`, `var_intra_index_m`, the only two variability columns clearing an 80% coverage bar on the Variant A sample -- inter-observation variability columns only cover 20-35%).
- X-ray log-fluxes limited to bands b/m, not b/h/m/s: including all 4 bands dropped row count to 19,259 (below a 20,000 floor), with `log_flux_aper_s` (88.5% coverage) and `log_flux_aper_h` (93.4%) the worst offenders -- the same soft/hard-band selection-bias effect Variant A's own ratio design already excludes for.
- Same StandardScaler -> UMAP (`n_neighbors=15, min_dist=0.1, n_components=2, random_state=42`) -> HDBSCAN grid -> LightGBM -> TreeSHAP pipeline as Variant A, unmodified (`src/clustering.py`, `src/interpretation.py` reused as-is).

### HDBSCAN grid

| configuration | n_clusters | n_model_clusters | noise_fraction | silhouette |
|---|---:|---:|---:|---:|
| mcs_20_ms_5_eom | 6 | 4 | 46.63% | 0.1202 |
| mcs_20_ms_5_leaf | 9 | 4 | 97.63% | 0.3248 |
| mcs_30_ms_5_eom | 5 | 4 | 46.72% | 0.1172 |
| mcs_30_ms_5_leaf | 7 | 4 | 97.03% | 0.2993 |
| mcs_50_ms_5_eom | 4 | 4 | 44.40% | 0.4161 |
| mcs_50_ms_5_leaf | 4 | 4 | 44.40% | 0.4161 |

**`mcs_50_ms_5_eom` selected** (4 clusters, met >=4-cluster target). Notably, `cluster_selection_method="leaf"` -- the setting that unlocked Variant A's 7-cluster resolution -- nearly collapses entirely here (86-97% noise at mcs 20/30); only at `mcs_50` do `eom` and `leaf` converge to the same usable partition.

### Cluster composition and top-5 per-cluster SHAP

| Cluster | n | %AGN | %non-AGN | %unlabeled | Top 5 SHAP features |
|---|---:|---:|---:|---:|---|
| -1 (noise) | 9,954 | 72.9 | 3.6 | 23.5 | -- |
| 0 | 118 | 67.8 | 0.0 | 32.2 | `var_intra_index_m` (0.221), `var_intra_index_b` (0.101), `log_flux_combo_z` (0.011), `WISE_w2snr` (0.006), `log_flux_combo_g` (0.004) |
| 1 | 834 | 30.9 | 0.0 | 69.1 | `log_flux_combo_z` (0.686), `optical_r_over_w1` (0.308), `color_w2_over_w3` (0.180), `color_r_over_z` (0.148), `xray_b_over_r` (0.109) |
| 2 | 75 | 64.0 | **25.3** | 10.7 | `optical_r_over_w1` (0.386), `color_w1_over_w2` (0.265), `log_flux_combo_z` (0.193), `color_r_over_z` (0.091), `xray_b_over_w1` (0.077) |
| 3 | 11,436 | 89.9 | 1.65 | 8.4 | `color_w2_over_w3` (0.613), `xray_b_over_w1` (0.571), `log_flux_wise_w1` (0.439), `xray_b_over_w2` (0.110), `WISE_w1snr` (0.069) |

**Top 15 global SHAP:** `log_flux_combo_z` (0.231), `color_w2_over_w3` (0.202), `optical_r_over_w1` (0.174), `xray_b_over_w1` (0.163), `log_flux_wise_w1` (0.121), `var_intra_index_m` (0.079), `color_w1_over_w2` (0.071), `color_r_over_z` (0.060), `xray_b_over_r` (0.037), `xray_b_over_w2` (0.036), `color_g_over_r` (0.033), `var_intra_index_b` (0.030), `WISE_w1snr` (0.028), `significance` (0.025), `log_flux_combo_r` (0.024). **No hardness-ratio column (`hard_hs`/`hard_hm`/`hard_ms`) appears anywhere in this top 15** -- a notable contrast with Variant A, where `hard_ms_hilim` and `hard_hs` both rank in the top 15.

**Balanced accuracy: 0.987.**

### Row-level crosswalk to Variant A (verified, not inferred from composition alone)

A join on shared source IDs between the two clusterings shows Variant A's Cluster 1 and Variant B's Cluster 2 are, to a large extent, literally the same sources:

| Extended cluster | Dominant Variant-A origin (of members present in both samples) |
|---|---|
| 3 (n=11,436) | 92.1% from Variant A Cluster 3 -- direct continuation |
| 2 (n=75) | 72.0% from Variant A Cluster 1 -- direct continuation (54 of the 86 Variant-A-Cluster-1 members that survived into the Variant B sample) |
| 0 (n=118) | 80.5% previously noise in Variant A; 15.3%/4.2% from Variant A Clusters 6/5 |
| 1 (n=834) | 77.0% previously noise in Variant A; 6.4%/9.0%/7.7% from Variant A Clusters 4/5/6 |

Variant A's Cluster 0 (n=54) does not survive as a distinct group at all -- it dissolves entirely into Variant B's noise pool.

### Figures

`extended_32feat/umap_cluster.png`, `extended_32feat/umap_label.png`, `extended_32feat/shap_global.png`, `extended_32feat/shap_cluster_2.png` (Cluster 2, the Variant-A-Cluster-1 analog).

---

## Comparison and interpretation

1. **The two variants agree on the single most scientifically interesting finding.** A small non-AGN-enriched population survives in both feature spaces: Variant A's Cluster 1 (n=136, 27.2% non-AGN) and Variant B's Cluster 2 (n=75, 25.3% non-AGN), driven by the same top two features (`optical_r_over_w1`, `color_w1_over_w2`) in both. The row-level crosswalk confirms this isn't coincidental similarity -- 72.0% of Variant A Cluster 1's surviving members land directly in Variant B Cluster 2. This is the strongest robustness result in either variant.

2. **Variant A resolves more fine structure than Variant B (7 clusters vs. 4).** Reduced-feature-space clustering is more sensitive to spectral-shape substructure precisely because brightness variance isn't dominating Euclidean distance in the scaled feature space; adding raw (log) fluxes back in Variant B coarsens that resolution rather than sharpening it. This is a legitimate, interpretable result rather than a simple "Variant B is broken" verdict -- but it does mean Variant B should not be read as an improvement on Variant A's resolution.

3. **Variant B surfaces one genuinely new small cluster driven by variability: Cluster 0** (n=118, 67.8% AGN, 32.2% unlabeled, 0% confirmed non-AGN), overwhelmingly dominated by `var_intra_index_m`/`var_intra_index_b` (0.221, 0.101 -- nothing else in its profile comes close). The crosswalk shows this cluster is mostly (80.5%) built from sources that were noise in Variant A, plus fragments of Variant A's Clusters 5/6. This is a genuine, if narrow, positive contribution from adding variability features -- a mostly-AGN, high-intra-observation-variability population, plausibly higher-variability AGN or blazar-like sources, though this is speculative without independent confirmation. It is not, however, evidence that variability resolved Variant A's *existing* mostly-unlabeled Clusters 4/5 into finer sub-structure -- those instead got absorbed into Variant B's larger, still-ambiguous Cluster 1 (69.1% unlabeled), which is itself 77% former noise.

4. **Reintroducing fluxes causes brightness to partially reclaim SHAP importance, displacing other signal rather than simply adding to it.** `log_flux_combo_z` is the #1 global feature in Variant B (0.231, though only ~1.15x the #2 feature -- not the 5-15x dominance seen in v1's full brightness collapse). Variant B's top 15 also loses every hardness-ratio column that appeared in Variant A's top 15 (`hard_ms_hilim`, `hard_hs`) -- brightness genuinely does discriminate when available, but it pushes out hardness signal rather than coexisting alongside it, and comes at the cost of the finer clustering resolution described in point 2.

**Net read:** Variant A remains the variant to lead with -- it resolves more real structure, keeps brightness/detection-selection effects out of the feature ratios by construction, and its central finding (the non-AGN-enriched small cluster) is independently corroborated by Variant B. Variant B is best presented as a robustness check and a lead for future work (Cluster 0's variability-driven signature), not as a competing primary result.

---

## Superseded: v1 and v2 full detail

### v1 (flux-based, brightness-dominated)

27,537 rows, 24 columns (`log_flux_*` WISE/PS1/Legacy + X-ray aperture fluxes + hardness + SNR), HDBSCAN config `mcs_50_ms_5`: 2 clusters, 35.8% noise. Dominant features: `log_flux_WISE_w1mpro` (~1.18), `combo_log_flux_z` (~0.75), 5-15x every other feature. Cluster 0 (n=2,211): 28.6% AGN-labeled. Cluster 1 (n=15,467): 82.5% AGN-labeled. Confirmed non-AGN sat at ~3.2-3.6% in both clusters and in noise.

### v2 (brightness-normalized, complete-case dropna)

69 columns (hardness, colors, `xray_ratio_*`, variability, SNR; zero raw flux confirmed), complete-case dropna collapsed 27,537 -> 1,731 rows. HDBSCAN grid (`min_cluster_size` in [20,30,50] x `eom`/`leaf`) gave byte-identical results across all 6 configs: 2 clusters, 12.8% noise, silhouette 0.538 -- the >=4-cluster target was not met. Dominant features: `xray_ratio_b_GAIA21P_g` (~0.27), `xray_ratio_h_GAIA21P_g` (~0.24), driven by a rare Gaia x-ray cross-match acting as a near-total AGN selector (Cluster 0, n=1,438: 97.8% AGN-labeled; confirmed non-AGN 0-4.1% everywhere).
