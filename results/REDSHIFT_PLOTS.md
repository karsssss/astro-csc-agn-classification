# Redshift-coded UMAP plots

Driver: `notebooks/redshift_plots.ipynb` (executed, outputs stored). Module:
`src/plots.py` (`plot_embedding_by_value`, new/additive -- does not modify
or replace any of `interpretation.py`'s existing plotting functions).
Reuses the Variant A (22-feat, `build_feature_matrix`) UMAP embedding from
`analysis.ipynb` -- same pipeline (`build_feature_matrix` ->
`StandardScaler` -> `fit_umap` with defaults `n_neighbors=15, min_dist=0.1,
n_components=2, random_state=42`), re-executed here since the embedding
was not persisted anywhere in the repo before this task (checked: no
`.npy`/`np.save`/`joblib.dump` for it anywhere). Not refit with different
parameters or a different seed. The embedding is now persisted to
`data/processed/umap_embedding_reduced_22feat.npy` (+ a matching
`_index.npy` of the `name` index, same row order) so future tasks can
reuse it directly instead of recomputing.

Variant A specifically (not Variant B) because `analysis.ipynb` only ever
computes a UMAP embedding for Variant A -- there is no Variant B embedding
anywhere in the repo to reuse or reproduce, consistent with
`STATE.md` §3's finding that Variant B has no committed driver before the
label-SHAP task added one (and that one didn't compute a UMAP embedding
either).

## Headline finding: gold non-AGN have zero spectroscopic redshift coverage, catalog-wide

Before interpreting any of the three figures below: **zero gold non-AGN
rows have a valid, high-quality `combo_Z`** -- not just within the 25,801-row
Variant A sample, but across the **entire 170,196-row catalog**. This was
verified directly, not assumed:

| | count |
|---|---:|
| Rows with valid `combo_Z` + `high_quality_z` (general catalog, 170,196 rows) | 27,335 |
| ...of which `agn_positive` | **27,335 (100%)** |
| ...of which `agn_negative` | **0** |
| ...of which `silver_agn` | **0** |
| ...of which unlabeled | **0** |
| `agn_negative` rows with `combo_Z` merely *present* (not even requiring `>0` or high-quality) | **0** |

The last row is the sharpest version of the finding: this isn't a
data-quality gate filtering out otherwise-present non-AGN redshifts --
`combo_Z` is **entirely absent** (native NaN) for every single confirmed
non-AGN row in the catalog, not merely low-quality or masked.

**Interpretation:** `combo_Z` is, in this catalog, a *de facto AGN-only*
column. The most likely explanation is that the DESI/SDSS/AGN-database
spectroscopic cross-match this catalog draws `combo_Z` from
(`merge_desi`, `z_source` values `desi`/`sdss`/`agndb`) preferentially
targets AGN candidates for spectroscopic follow-up in the first place --
these surveys are not a blind, complete spectroscopic census of the X-ray
source population, so a non-AGN X-ray source (e.g. a galaxy or stellar
source) is simply much less likely to have been selected for spectroscopy
at all. This is a property of the input catalogs, not a bug introduced
anywhere in this project's pipeline.

Within the 25,801-row Variant A sample specifically:

| | total | with valid `combo_Z` |
|---|---:|---:|
| Gold (`label_3class` in `{0,2}`) | 13,115 | 9,992 |
| Gold AGN (`label_3class`==2) | 12,272 | 9,992 (81.4%) |
| Gold non-AGN (`label_3class`==0) | 843 | **0** |

So "Gold with valid z" and "Gold AGN with valid z" are numerically
identical (9,992) -- every z-having gold row is an AGN.

## Figures

All saved to `results/figures/reduced_22feat/`, matching this variant's
existing figure directory.

### 1. `umap_redshift_gold_only.png`

**Subset:** gold (`label_3class` in `{0,2}`) **with** valid `combo_Z` --
9,992 points. Per the headline finding above, this subset is **AGN-only in
practice** despite being specified as "gold" (both classes) -- the
non-AGN half of "gold" never survives the `has_z` filter. Color = `combo_Z`
on a viridis colormap with a colorbar. Caption: *"2D UMAP preserves
topology, not metric distance."*

**How to read it:** this figure shows where redshift-confirmed AGN sit in
UMAP space and how redshift varies across that population -- it is not a
gold-AGN-vs-gold-non-AGN comparison, despite the "gold" in its name.

### 2. `umap_redshift_agn_only_size.png`

**Subset:** gold AGN only (`label_3class`==2) with valid `combo_Z` -- the
same 9,992 points as Figure 1 (identical subset, since among gold rows
only AGN ever have `combo_Z`). Both marker **size and color** encode
`combo_Z`, size scaled into the 10-100pt range after clipping to the
1st-99th percentile (so a handful of extreme-z outliers don't compress
the rest of the size range to indistinguishable dots). Size legend shows
3 representative values at the 5th/50th/95th percentile of `combo_Z`
within this subset.

**How to read it:** redundant size+color encoding of the same variable --
useful for spotting high-z points at a glance even in dense regions where
color alone might be hard to distinguish, and for print/grayscale
contexts where color may not render.

### 3. `umap_redshift_agn_vs_nonagn.png`

**Subset:** all gold rows (`label_3class` in `{0,2}`), **regardless of
whether they have a redshift** -- 13,115 points (12,272 AGN + 843
non-AGN). Shape: circle = AGN, triangle = non-AGN. Color: `combo_Z` where
available; **gray where not**, handled as a first-class case in
`plot_embedding_by_value` (any NaN `value_col` entry is drawn as a fixed
neutral gray point with its own legend entry, rather than being dropped
from the plot or mis-rendered by the colormap).

This figure was originally going to use the same `has_z`-only subset as
Figures 1-2, which -- per the headline finding -- would have silently
rendered as "AGN only," visually resembling an AGN-vs-non-AGN comparison
without actually being one. Instead, it plots the **full gold population**
and makes the coverage gap itself visible:

- **Colored circles (n=9,992):** AGN with valid redshift.
- **Gray circles (n=2,280):** AGN *without* a valid redshift (12,272 total
  AGN minus the 9,992 with one) -- these exist and are drawn, just
  uncolored.
- **Gray triangles (n=843):** non-AGN, **all** of which lack a valid
  redshift -- there is no "colored triangle" category in this figure at
  all, and that absence is the point.

**How to read it:** the gray triangles are non-AGN plotted *without*
color because none of them have a redshift to color by -- they are not
filtered out, thinned, or hidden. This figure visualizes the
AGN/non-AGN spatial distribution in UMAP space *and* the redshift-coverage
asymmetry between the two classes simultaneously, which is more honest
than either dropping non-AGN entirely (Figures 1-2's approach) or
silently mislabeling an AGN-only plot as an AGN-vs-non-AGN one.

## Redshift structure diagnostics

Quantitative checks behind a visual read of the figures above, computed
on the 9,992-point AGN-with-z subset (Pearson correlations) or the same
subset's top decile (spatial concentration check).

| Diagnostic | Value | Plain-English read |
|---|---:|---|
| Pearson r(UMAP1, `combo_Z`) | **-0.578** (p≈0) | A moderately strong negative correlation -- redshift decreases fairly consistently as UMAP1 increases. This is a real, non-trivial gradient, not noise (p-value is effectively zero at n=9,992). |
| Pearson r(UMAP2, `combo_Z`) | **-0.303** (p=9.7e-211) | A weaker but still highly significant negative correlation along the second axis -- redshift varies with position along both UMAP dimensions, more strongly along UMAP1. |
| AGN `combo_Z` median (IQR) | **0.766** (0.293, 1.419) | Typical confirmed AGN in this sample sits at z≈0.77, with the middle 50% between z≈0.29 and z≈1.42 -- consistent with a broad, moderate-to-high-redshift AGN population (not a narrow, all-local sample). |
| Top-decile-by-z AGN spatial concentration | **0.0th percentile** | The 999 highest-redshift AGN are dramatically more tightly clustered in UMAP space than a random same-size draw of AGN (mean pairwise distance 3.336 vs. a random-draw distribution of mean 4.718, std 0.078 across 200 draws) -- the true value falls below every one of the 200 random draws. This is a strong, unambiguous result: high-z AGN are not scattered evenly through AGN-occupied UMAP space, they occupy a distinct, spatially compact sub-region. |

Non-AGN `combo_Z` median/IQR: **not computable** -- zero non-AGN rows have
a valid redshift (see headline finding).

## Implications for future work

Any downstream analysis that wants to compare AGN and non-AGN redshift
distributions, or use redshift as a non-AGN feature/covariate, is blocked
by this catalog's data availability, not by anything in this pipeline --
obtaining non-AGN redshifts would require a new spectroscopic cross-match
or survey (this catalog's existing `combo_Z` sources will not produce
them no matter how the pipeline code is changed).
