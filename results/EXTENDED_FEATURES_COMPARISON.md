# Extended Features Comparison -- raw fluxes + X-ray variability vs. v_final

Robustness-check variant, not a replacement for v_final. Built via
`src/conversions.py::build_feature_matrix_with_fluxes_and_variability`
(new function, `build_feature_matrix` unmodified). Tests two independent
methodological questions before finalizing the professor email:

1. Was dropping raw flux/magnitude columns after computing ratios actually
   necessary, or just a defensible-but-untested choice?
2. Do X-ray variability features resolve v_final's mostly-unlabeled small
   clusters (n=85, n=94) into more distinct, physically-identifiable
   sub-populations (candidate compact source classes: XRBs, CVs, pulsars),
   per Perez-Diaz (2024) and Yang (2022)'s use of variability for this
   purpose?

## Step 1: variability feature availability

Raw column names are `var_intra_index_{b,h,m,s,u,w}` and
`var_inter_index_{b,h,m,s,u,w}` (plus `var_inter_hard_prob/sigma_{hm,hs,ms}`)
-- not the `var_prob_*`/`var_intra_prob_*` names originally guessed.
Coverage on v_final's 25,801-row sample:

| column | coverage |
|---|---:|
| `var_intra_index_b` | 98.90% |
| `var_intra_index_m` | 87.11% |
| `var_intra_index_h` | 76.40% |
| `var_intra_index_s` | 76.29% |
| `var_inter_index_*` (all bands) | 20-35% |
| `var_inter_hard_prob/sigma_*` | ~31-32% |

Only `var_intra_index_b` and `var_intra_index_m` clear an 80% bar --
notably *intra*-observation variability, not *inter*-observation, has
usable coverage here. These two are the variability features used below.

## Setup

- **v_final:** 25,801 rows x 22 columns (ratios, colors, hardness ratios,
  SNRs, significance -- zero raw flux/magnitude columns).
- **Extended:** 22,417 rows x 32 columns = v_final's 22 + 8 log10(flux)
  columns (`log_flux_aper_b`, `log_flux_aper_m`, `log_flux_combo_g/r/z`,
  `log_flux_wise_w1/w2/w3`) + 2 variability columns (`var_intra_index_b`,
  `var_intra_index_m`).
- **Row count drop:** 25,801 -> 22,417 (-13.1%), driven by the added
  columns' coverage requirements.
- **X-ray bands deliberately limited to b/m, not b/h/m/s:** a first attempt
  at including all 4 X-ray aperture bands as log-flux columns dropped the
  row count to 19,259 (below a 20,000-row floor). `log_flux_aper_s`
  (88.5% coverage) and `log_flux_aper_h` (93.4%) were the two worst
  offenders -- this is the **exact same soft/hard-band selection-bias
  effect** `build_feature_matrix`'s own docstring already excludes soft/
  hard-band X-ray *ratios* for (requiring `flux_aper_s`/`flux_aper_h`
  strictly positive discards real non-detections). The extended matrix's
  X-ray log-fluxes were restricted to b/m to match that same rationale.
  This is itself a small piece of evidence *for* the original design: the
  same selection-bias trap reappears the moment raw per-band fluxes are
  reintroduced, even in log form.
- Same StandardScaler -> UMAP (`n_neighbors=15, min_dist=0.1,
  n_components=2, random_state=42`) -> HDBSCAN grid -> LightGBM ->
  TreeSHAP pipeline as v_final, unmodified (`src/clustering.py`,
  `src/interpretation.py` reused as-is).

## HDBSCAN grid summary

| configuration | n_clusters | n_model_clusters | noise_fraction | silhouette |
|---|---:|---:|---:|---:|
| mcs_20_ms_5_eom | 6 | 4 | 46.63% | 0.1202 |
| mcs_20_ms_5_leaf | 9 | 4 | **97.63%** | 0.3248 |
| mcs_30_ms_5_eom | 5 | 4 | 46.72% | 0.1172 |
| mcs_30_ms_5_leaf | 7 | 4 | **97.03%** | 0.2993 |
| mcs_50_ms_5_eom | 4 | 4 | 44.40% | 0.4161 |
| mcs_50_ms_5_leaf | 4 | 4 | 44.40% | 0.4161 |

**Selected: `mcs_50_ms_5_eom`** (4 clusters, met >=4-cluster target).

Notable: in v_final, `cluster_selection_method="leaf"` was what *unlocked*
the useful 7-cluster result (all 3 `leaf` configs identical, `eom` stuck at
2 clusters). Here it's reversed and worse: `leaf` at mcs 20/30 collapses to
**97% noise** -- almost the entire sample is discarded as noise, with only
a handful of points surviving into 4 tiny modeled clusters. Only at
`mcs_50` do `eom` and `leaf` converge to an identical, usable 4-cluster
result. The extended feature space is more brittle here: v_final had 3
independent configs agreeing on its answer; the extended run's usable
answer comes from only 1 of 6 grid cells (with a 2nd, `mcs_50_leaf`,
landing on it by coincidence of degenerating to the same partition).

## Global SHAP top-15 comparison

| feature | v_final rank (value) | extended rank (value) |
|---|---|---|
| `optical_r_over_w1` | #1 (0.1414) | #3 (0.1737) |
| `color_r_over_z` | #2 (0.0879) | #8 (0.0598) |
| `color_g_over_r` | #3 (0.0770) | #11 (0.0328) |
| `xray_b_over_r` | #4 (0.0622) | #9 (0.0370) |
| `WISE_w1snr` | #5 (0.0513) | #13 (0.0281) |
| `xray_b_over_w1` | #6 (0.0445) | **#4 (0.1629)** |
| `significance` | #7 (0.0423) | #14 (0.0255) |
| `color_w2_over_w3` | #8 (0.0357) | **#2 (0.2022)** |
| `WISE_w2snr` | #9 (0.0265) | not in top 15 |
| `color_w1_over_w2` | #10 (0.0257) | #7 (0.0710) |
| `xray_b_over_w2` | #11 (0.0241) | #10 (0.0356) |
| `hard_ms_hilim` | #12 (0.0184) | not in top 15 |
| `xray_m_over_r` | #13 (0.0164) | not in top 15 |
| `WISE_w3snr` | #14 (0.0161) | not in top 15 |
| `hard_hm` | #15 (0.0133) | not in top 15 |
| `log_flux_combo_z` | *(zero raw flux cols exist)* | **#1 (0.2313)** |
| `log_flux_wise_w1` | *(zero raw flux cols exist)* | **#5 (0.1214)** |
| `var_intra_index_m` | *(not included)* | #6 (0.0793) |
| `var_intra_index_b` | *(not included)* | #12 (0.0298) |
| `log_flux_combo_r` | *(zero raw flux cols exist)* | #15 (0.0237) |

**A raw brightness feature (`log_flux_combo_z`) is the #1 global feature
in the extended run**, ahead of every ratio and color -- though at 1.15x
the #2 feature, not the 5-15x dominance seen in v1's full brightness
collapse. `log_flux_wise_w1` also enters at #5. Both variability columns
appear (#6, #12) with real but modest weight -- present, not dominant.

## Cluster count comparison

v_final: **7 real clusters** (sizes 12,284 / 136 / 98 / 94 / 86 / 54 / 54)
+ 50.4% noise.

Extended: **4 real clusters** (sizes 11,436 / 834 / 118 / 75) + 44.4% noise.

The six small v_final clusters (0, 1, 2, 4, 5, 6; sizes 54-136) do **not**
map cleanly onto the extended run's three small clusters. Extended
Cluster 1 alone (n=834) is 6-15x larger than any individual v_final small
cluster -- this reads as **merged/coarsened**, not split or preserved.
Structure got coarser (7 -> 4 clusters), the opposite of what a
fine-discrimination hypothesis (variability separating compact source
subclasses) would predict.

## Cluster 1 equivalent

Extended **Cluster 2** (n=75) is the closest analog: highest
confirmed-non-AGN concentration among extended's non-dominant real
clusters.

| | v_final Cluster 1 (n=136) | Extended Cluster 2 (n=75) |
|---|---:|---:|
| % confirmed AGN | 58.8% | 64.0% |
| % confirmed non-AGN | **27.2%** | **25.3%** |
| % unlabeled | 14.0% | 10.7% |

Composition is strikingly similar (25.3% vs. 27.2% non-AGN, within ~2
points) despite a smaller member count and a different 32-column feature
space. Top SHAP drivers for extended Cluster 2: `optical_r_over_w1`
(0.386, dominant), `color_w1_over_w2` (0.265), `log_flux_combo_z` (0.193),
`color_r_over_z` (0.090), `xray_b_over_w1` (0.077) -- **the same top
physical driver** (`optical_r_over_w1`) as v_final Cluster 1 (there:
0.476), and the same secondary driver (`color_w1_over_w2`, proportionally
even more prominent here). This is the one genuinely reassuring robustness
signal in this comparison: the scientifically interesting non-AGN-enriched
cluster and its physical explanation both survive a materially different
feature space. It's also notable that `log_flux_combo_z` still ranks #3
even inside this specific cluster's profile -- brightness isn't absent
here either, just not dominant.

## Clusters 4/5 (mostly-unlabeled) identifiability

v_final Cluster 4 (n=85, 85.9% unlabeled, `hard_hm`/`hard_hs`-driven) and
Cluster 5 (n=94, 58.5% unlabeled, `optical_r_over_w1`/`significance`-driven)
have no clean 1:1 counterpart in extended's 4-cluster result. The closest
analog is extended **Cluster 1** (n=834): 69.1% unlabeled, 30.9% AGN, 0%
confirmed non-AGN -- same general flavor (mostly-unlabeled, low-moderate
AGN fraction, zero confirmed non-AGN) as v_final's 4/5/6 combined, but as
one large, coarse cluster rather than three smaller, distinct ones.

**This does not support the variability hypothesis as tested.** Rather
than splitting into more distinct, physically-identifiable sub-populations
(candidate XRBs/CVs/pulsars), the mostly-unlabeled population got
absorbed into a single larger, still-ambiguous bucket. `var_intra_index_b`
and `var_intra_index_m` do carry real SHAP weight (#6, #12 of 32
features), so variability isn't contributing nothing -- but with only 2
columns clearing the coverage bar, and combined with the flux-driven shift
in overall structure, this run cannot confirm or rule out the variability
hypothesis; it's a weak negative result, not a strong one.

## Figures

`results/figures/extended_32feat/umap_cluster.png`,
`results/figures/extended_32feat/umap_label.png`,
`results/figures/extended_32feat/shap_global.png`,
`results/figures/extended_32feat/shap_cluster_2.png` (Cluster 2, the
v_final-Cluster-1 analog) -- same plotting functions and color conventions
as the v_final figures, which now live in `results/figures/reduced_22feat/`.

## Honest read and recommendation

The extended feature space tells a **partially compatible, partially
worse** story, not a straightforwardly better one. On the flux question,
the evidence favors the original choice: the moment raw (log) fluxes are
added back, one of them (`log_flux_combo_z`) reclaims the #1 global SHAP
slot -- a smaller-scale echo of v1's brightness-dominance failure mode,
not a full relapse, but a real regression from v_final's flatter, more
physical top-5. On the variability question, the evidence is weak and
points the wrong way for the stated hypothesis: rather than resolving
finer sub-structure, the small clusters merged into fewer, coarser ones
(7 -> 4), and the `leaf` HDBSCAN method that gave v_final its resolution
nearly collapsed entirely (86-97% noise) in this feature space. The one
clear positive: the scientifically interesting non-AGN-enriched small
cluster (v_final's Cluster 1) reappears with near-identical composition
(25-27% non-AGN) and the same top physical driver in both feature spaces
-- a genuine, if narrow, robustness result.

**Recommendation: keep v_final as canonical for the email.** Reference
this extended run as a robustness check that (a) supports the original
flux-dropping decision rather than undermining it, and (b) shows
preliminary, inconclusive evidence on the variability question -- worth
mentioning as a direction for follow-up (better-covered variability
columns, or dedicated HDBSCAN tuning on a flux-free + variability matrix)
but not as a finding to lead with. Cluster 1's cross-feature-space
persistence is worth a sentence in the email as an added confidence point
for that specific result.
