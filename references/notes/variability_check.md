# Variability check -- is WISE variability a better AGN feature than optical variability?

Time-boxed to 25 minutes. Analysis only -- no changes to either variant's
feature matrix (`X`) in `src/conversions.py`. Figure saved to
`results/figures/reduced_22feat/variability_class_distributions.png`.

## Step 1: Inventory

Searched every column in the general df (170,196 rows, 208 cols, after
`load_catalog`/`merge_csc_spectral_fits`/`merge_desi`/`build_labels`) and
all three raw source CSVs directly (`Multiwave_UMAP_6_26_26.csv`,
`csc_spec_fits.csv`, `Multiwave_7_8_26.csv`) for variability-related
columns:

| Category | Pattern searched | Found |
|---|---|---:|
| WISE variability/quality (`*var*`, `*_mjd*`, `WISE_pm*`, `*moon_lev*`, `*qual*`, `*_cc*`) | regex across all columns | **0** |
| CSC X-ray variability (`var_intra_*`, `var_inter_*`, all bands) | exact prefix match | **18** |
| Optical variability (`*variab*` outside WISE/CSC namespace) | regex | **0** |
| Gaia `phot_variable_flag` | exact/substring match | **0** |

**This catalog contains no WISE variability columns and no optical
variability columns at all.** The actual WISE columns present are just
`WISE_ra/dec/glon/glat`, `WISE_w1mpro/w2mpro/w3mpro` (photometry, not
variability), `WISE_w1snr/w2snr/w3snr`, `Separation_WISE_CSC`,
`WISE_match_flag`, and (from the DESI merge) `WISE_W1_lum/W2_lum/W3_lum`
-- no AllWISE/CatWISE multi-epoch variability, proper-motion, moon-level,
quality-grade, or contamination-flag columns were ever merged into this
project's catalog. Similarly, no Gaia `phot_variable_flag`, no PS1/SDSS
epoch-to-epoch variability indicator (`SDSS_SN_MEDIAN_*` is signal-to-noise,
not variability; `SDSS_VDISP` is velocity dispersion, unrelated) exist
either. The only genuine variability measurements anywhere in this project
are the 18 CSC `var_intra_*`/`var_inter_*` X-ray columns.

## Coverage table (CSC `var_intra_*`/`var_inter_*`, the only candidates)

Gold sample: 39,173 rows (33,136 `agn_positive` + 6,037 `agn_negative`).

| feature | coverage (overall) | coverage (agn_positive) | coverage (agn_negative) | coverage (gold) |
|---|---:|---:|---:|---:|
| `var_intra_index_b` | 92.7% | 98.1% | 37.1% | 88.7% |
| `var_intra_index_m` | 73.5% | 85.9% | 24.3% | 76.4% |
| `var_intra_index_s` | 53.7% | 74.8% | 19.8% | 66.4% |
| `var_intra_index_h` | 63.3% | 74.2% | 23.2% | 66.3% |
| `var_inter_index_b` | 47.3% | 38.9% | 22.3% | 36.3% |
| `var_inter_index_m` | 45.7% | 38.4% | 20.9% | 35.7% |
| `var_inter_index_h` | 45.6% | 37.9% | 21.3% | 35.4% |
| `var_inter_index_s` | 41.5% | 37.0% | 19.5% | 34.3% |
| `var_inter_hard_prob_hm` | 41.9% | 34.8% | 19.7% | 32.4% |
| `var_inter_hard_prob_ms` | 41.6% | 34.7% | 19.3% | 32.4% |
| `var_inter_hard_sigma_hm` | 41.4% | 34.5% | 19.3% | 32.1% |
| `var_inter_hard_sigma_ms` | 41.0% | 34.4% | 19.0% | 32.0% |
| `var_inter_hard_prob_hs` | 40.6% | 34.2% | 19.0% | 31.9% |
| `var_inter_hard_sigma_hs` | 40.4% | 34.1% | 18.9% | 31.8% |
| `var_inter_index_u` | 21.8% | 22.4% | 9.8% | 20.5% |
| `var_intra_index_u` | 6.0% | 13.8% | 1.6% | 11.9% |
| `var_intra_index_w` | 1.4% | 0.8% | 0.1% | 0.7% |
| `var_inter_index_w` | 0.7% | 0.2% | 0.0% | 0.2% |

A consistent pattern across nearly every row: `agn_positive` coverage is
2-4x higher than `agn_negative` coverage for every column -- confirmed AGN
are more likely to have any variability measurement at all, independent of
what that measurement says. Worth keeping in mind when reading the AUC
values below (coverage itself, not just the value, differs by class).

## Step 2: Single-feature ROC-AUC (gold sample, no dropna, ≥50% gold coverage)

Only **4 of the 18** CSC candidates clear the 50% gold-coverage bar (not
5 -- there simply aren't 5 qualifying columns; no WISE/optical candidates
exist to fill out a 5th). AUC computed via `roc_auc_score(y, x)` directly
on the raw feature value for the subset of gold rows where that feature is
non-null (boolean/categorical features would reduce to the same rank-based
2x2-table AUC under this formula, but none of the 4 qualifying columns are
boolean/categorical -- all four are continuous indices).

| feature | source | gold coverage | n scored | AUC | \|AUC − 0.5\| |
|---|---|---:|---:|---:|---:|
| `var_intra_index_b` | CSC | 88.7% | 34,764 | 0.4567 | **0.0433** |
| `var_intra_index_h` | CSC | 66.3% | 25,973 | 0.4776 | 0.0224 |
| `var_intra_index_s` | CSC | 66.4% | 25,993 | 0.4899 | 0.0101 |
| `var_intra_index_m` | CSC | 76.4% | 29,943 | 0.4988 | 0.0012 |

**Every single-feature AUC is within 0.05 of pure chance (0.5).**
`var_intra_index_b` is the "best" of the four, and it's still nearly
uninformative alone (AUC=0.457 -- notably *below* 0.5, meaning higher
`var_intra_index_b` weakly predicts *non-AGN*, not AGN, though the effect
is small).

## Step 3: Figure

`results/figures/reduced_22feat/variability_class_distributions.png` --
4 panels (all 4 qualifying features, not 5; see above), KDE of AGN vs.
non-AGN within the gold sample for each. No panel needed a log x-axis
(none of the four showed the required combination of near-all-positive
values and >100x dynamic range to trigger the log-scale heuristic used).
Visually, all four panels show heavily overlapping AGN/non-AGN
distributions, consistent with the near-chance AUCs above.

## Step 4: Verdict

**The meeting claim -- "WISE variability is a better AGN feature than
optical variability" -- cannot be evaluated in this catalog, because
neither WISE variability nor optical variability data exists in it at
all.** This isn't a coverage problem (case 2 in the original framing) or a
case where optical wins instead (case 3); it's a more basic issue: **the
premise of the comparison has no data to test** for either side.

The only variability signal available anywhere in this project's data
comes from CSC X-ray variability (`var_intra_*`/`var_inter_*`), and even
that is not usefully discriminative: all 4 columns clearing the 50%
gold-coverage bar for a single-feature test have AUC within 0.05 of
chance. This is consistent with, and now triangulated by, two independent
prior findings: `EXTENDED_FEATURES_COMPARISON.md`'s Variant B (which
includes `var_intra_index_b`/`m` in a 32-feature multivariate model)
found them present in cluster-SHAP with real-but-modest weight (ranks
#6/#12 of 32), and `results/LABEL_SHAP.md`'s gold-label SHAP found neither
variability column cracked the top-20 for either feature-space variant.
All three independent checks (multivariate cluster-SHAP, multivariate
label-SHAP, and this univariate AUC check) agree: CSC X-ray variability
carries little to no standalone AGN/non-AGN signal in this catalog.

If the meeting claim is meant to be revisited, the actionable next step is
not a methodology change here -- it's acquiring WISE (AllWISE/CatWISE
multi-epoch photometry, moon-level, quality/contamination flags) or
optical (Gaia `phot_variable_flag`, ZTF/PS1 lightcurve-derived) variability
data and merging it into the catalog, since none currently exists to
compare.

## Recommendation: columns to add to `X`

**0 of 18 candidate columns recommended for addition**, based on this
univariate check alone. Reasoning: all 4 columns that clear a usable
coverage threshold have single-feature AUC indistinguishable from chance
(≤0.043 deviation), and this null result is corroborated by two prior
independent multivariate analyses (see Step 4). Adding a feature with no
demonstrated univariate signal and only marginal multivariate SHAP weight
is not well-motivated by the evidence gathered here. If one were to be
tried despite this, `var_intra_index_b` is the least-bad candidate (best
coverage at 88.7% gold, largest -- if still small -- AUC deviation at
0.043), but this write-up does not recommend it.

This is a univariate, time-boxed check and does not rule out variability
mattering in combination with other features or under a different
encoding (e.g. missingness itself as a signal, given the striking
AGN-vs-non-AGN coverage gap noted in the coverage table above) -- that
would need a dedicated follow-up, not a recommendation made here.
