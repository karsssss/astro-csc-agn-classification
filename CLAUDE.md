# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is an astronomy/ML research project (not the Astro web framework, despite the directory name). It classifies X-ray sources from the Chandra Source Catalog (CSC) as AGN (Active Galactic Nuclei) or non-AGN/galaxies using multiwavelength cross-matched data (X-ray, optical, infrared). The work is entirely notebook-based: there is no build system, linter, or test suite. `src/` is a thin local package that notebooks import stable, reusable logic from (starting with `src/features.py` — magnitude-to-flux conversion, see "Photometric units and zero-points" below); `data/processed/` and `output/models/` still exist but are empty, landing spots for future processed data/saved models.

All real work happens in `notebooks/*.ipynb`, reading from `data/raw/` and writing figures to `output/figures/`.

## Environment

There is no `requirements.txt`/`environment.yml`/`pyproject.toml` in this repo — dependencies must be inferred from notebook imports.

Packages live in a mamba/miniforge environment named **`quant`** (not `base`) — `base` has no ML/data packages installed. Any command that runs Python for this project (scripts, `jupyter nbconvert`, package checks) must use the `quant` env. Since shell state does not persist between separate tool invocations, activation must happen within the same command rather than once up front:

```bash
source /c/Users/I/miniforge3/etc/profile.d/conda.sh && conda activate quant && python ...
```

or invoke the interpreter directly, which is simpler for one-off commands:

```bash
/c/Users/I/miniforge3/envs/quant/python.exe -c "..."
# Windows-path form, e.g. from PowerShell:
C:\Users\I\miniforge3\envs\quant\python.exe -c "..."
```

Key libraries used across notebooks:

- Data/IO: `pandas`, `numpy`, `astropy` (`astropy.io.fits`, `astropy.table.Table`) for reading `.fits` catalog files
- Dimensionality reduction: `umap-learn`, `pacmap`, `scikit-learn` (`PCA`, `StandardScaler`, `SimpleImputer`)
- Clustering: `hdbscan`, `sklearn.metrics.silhouette_score`
- Classification: `lightgbm`, `catboost`, `sklearn.ensemble.RandomForestClassifier`
- Model explainability: `shap`
- Plotting: `matplotlib`

When adding setup instructions or dependency files, base them on the actual imports in `notebooks/*.ipynb` rather than assuming a particular environment manager.

## Running notebooks

Notebooks assume the working directory is `notebooks/` and reference data with relative paths like `../data/raw/<file>`. Run them from within `notebooks/` (or with Jupyter's default per-notebook cwd).

There are no automated tests, lint configs, or CI — correctness is judged by inspecting notebook output (printed shapes/value_counts, plots) inline.

## Data files (`data/raw/`)

- `Multiwave_UMAP_6_26_26.csv` — the current/canonical merged multiwavelength catalog (~174 columns), indexed by `name` (CSC source designation, e.g. `2CXO_...`). This is the file most notebooks load as the primary `df`.
- `Multiwave_UMAP_6_22_26.csv`, `Multiwave_6_12_26.fits` — older snapshots of the same catalog, kept for comparison in `data_comparison.ipynb`.
- `csc_spec_fits.csv` — CSC X-ray power-law spectral fit parameters (e.g. `powlaw_gamma`), cross-matched on `name` after replacing `"2CXO "` with `"2CXO_"`.
- `Multiwave_7_8_26.csv` — DESI spectroscopic cross-match (`DESI_TARGETID`, `DESI_Z`, `DESI_SPECTYPE`, `combo_Z`, `high_quality_z`, etc.), merged in to supplement redshifts and AGN labels.

## Data conventions

### Column naming conventions

Columns are prefixed by survey/catalog of origin, so the prefix tells you the data source:
- No prefix / CSC-native: `ra`, `dec`, `significance`, `flux_aper_{b,h,m,s,u,w}` (broad/hard/medium/soft/ultrasoft/wide band), `hard_*` (hardness ratios), `var_intra_*`/`var_inter_*` (variability indices), `CSC_b_lum` (X-ray luminosity)
- `PS21P_*` — Pan-STARRS photometry
- `LEG21P_*` — DESI Legacy Survey photometry
- `GAIA21P_*` — Gaia
- `2MASS21P_*` — 2MASS infrared photometry
- `WISE_*` — WISE infrared photometry (`w1mpro`/`w2mpro`/`w3mpro` magnitudes used to build IR colors)
- `SDSS_*` (and `_sdss` suffixed duplicates from a second merge) — SDSS spectroscopy/photometry
- `AGNDB_*` — external AGN database cross-match/classifications
- `DESI_*`, `combo_*` — DESI spectroscopic cross-match

### Label taxonomy

AGN status is built up from several independent signals, then combined into a gold/silver/unlabeled scheme:

- Raw boolean labels: `xray_agn`, `optical_agn`, `ir_agn` (renamed from `IR_agn`), `non_agn` (renamed conflicts are resolved by giving positive AGN labels priority over `non_agn`)
- `agn_class` / `agn_positive` — OR of `xray_agn`, `optical_agn`, `ir_agn` (confirmed/"gold" AGN)
- `agn_negative` — confirmed non-AGN, with positive labels taking priority when both are set (`label_conflict` flags the overlap)
- `silver_score` / `silver_agn` — a weighted heuristic score (WISE Mateos-wedge membership, CSC power-law spectral fit presence, high X-ray-to-optical flux ratio, high-quality redshift) used to flag likely-but-unconfirmed AGN among otherwise-unlabeled sources
- `label_3class` — final encoding used for UMAP plots: `-1` unlabeled, `0` confirmed non-AGN/galaxy, `1` silver AGN, `2` confirmed AGN
- The dataset is split ~30K labeled (gold) vs ~140K unlabeled sources; the ML approach is explicitly two-pronged: (1) supervised training on the gold sample, (2) unsupervised/self-training using the unlabeled sample to expand training signal

### Derived features

Common feature engineering seen across notebooks, useful to recognize when reading further analysis:
- `WISE_W1_W2`, `WISE_W2_W3` — IR colors from WISE magnitudes; `mateos_wedge` is a boolean AGN-selection region defined by piecewise-linear boundaries in this color-color space
- `PS_g_r`, `PS_r_i`, `LEG_G_R`, `LEG_R_Z` — optical colors; `PS_r_W1`, `PS_i_W2` — optical-to-IR colors
- `log_flux_aper_{b,h,s}` — log10 of X-ray fluxes; `xray_opt_r_relation_legacy`/`xray_opt_g_relation_legacy` — the original ad hoc X-ray-to-optical flux ratio proxies, superseded by the systematic `xray_ratio_{b,h,s}_<band>` set (see Photometric units and zero-points below) but kept so existing downstream cells don't silently break
- `combo_g`/`combo_r`/`combo_z_mag` — PS1 photometry with Legacy Survey used as fallback via `combine_first`

### Missing-value sentinels

Different surveys encode non-detections with different magic values in the raw CSVs: WISE (`*mpro`) and Pan-STARRS (`PS21P_*`) use exactly `-999`; DESI Legacy Survey (`LEG21P_*`) uses exactly `-9`. Project-wide rule: **any magnitude ≤ 0 is treated as NaN**. Real point-source magnitudes in these catalogs are always positive, so this rule catches all known sentinels uniformly without having to track a different magic number per survey. Feeding a sentinel into `10^(-0.4 × mag)` overflows to `inf` and silently corrupts downstream features — always mask sentinels before any magnitude-to-flux conversion.

2MASS is a special case worth knowing about: only ~10% of rows (17,277–18,380 of 170,196) have a 2MASS crossmatch at all, because 2MASS is far shallower than PS1/WISE/Legacy Survey — the other ~90% are true non-detections (native NaN in the raw CSV), not sentinel-coded. Within that already bright-selected ~10%, only a handful of rows (7–8 per band) carry implausible negative magnitudes (likely saturated-source artifacts, not a systematic sentinel). Net effect: any feature derived from `2MASS21P_*` reflects the bright tail of the population, not the full catalog — keep that in mind before reading too much into 2MASS-based feature importance.

**This rule retroactively matters for the "Reduced Dataset Case" branch too** (see "Parallel research configurations" below), not just the general case. That branch's `dropna(subset=[..., "WISE_w1mpro", "WISE_w2mpro", "WISE_w3mpro", ...])` filter does **not** catch a literal `-999` — it's a real float, not NaN — so before the sentinel fix landed, ~1,100 sentinel-contaminated rows were silently surviving into the "clean" reduced dataset (28,632 → 27,537 rows once the fix correctly excludes them). Any HDBSCAN clustering, classifier, or SHAP results produced on the reduced case before this fix should be treated as having run on data with real `-999` values mixed into genuine WISE magnitudes, and ideally re-run.

### Photometric units and zero-points

Raw photometric columns are a mix of already-calibrated fluxes and magnitudes that need converting.
- Already fluxes (skip conversion): `SDSS_CALIBFLUX_{u,g,r,i,z}` (nanomaggies), `log_flux_aper_{b,h,s}` (X-ray, already log10).
- Need conversion: `PS21P_*`, `LEG21P_*`, `WISE_*mpro`, `2MASS21P_*`, `GAIA21P_g`.

Two zero-point systems are in play: AB (3631 Jy universal, used by PS1/Legacy Survey/SDSS) and Vega (per-band, used by WISE/2MASS/Gaia). Conversion: `flux_Jy = zp_Jy × 10^(-0.4 × mag)`, then `log10(flux_Jy)` to match the X-ray log-flux scale. Existing color columns (`PS_g_r`, `WISE_W1_W2`, etc.) are same-system magnitude differences and are already valid flux ratios in log space — do not re-transform them. Zero-point sources: WISE Cutri/Jarrett 2011; 2MASS Cohen et al. 2003; AB by definition; Gaia DR3 VEGAMAG (~0.05 mag residual color-term uncertainty — treat `GAIA21P_g` flux as good to ~5%, not exact). The zero-point table, conversion function, and cross-check utility live in `src/features.py` (`ZERO_POINT_JY`, `mag_to_flux_jy`, `apply_mag_to_flux_conversion`, `cross_check_mag_zero_point`).

Systematic X-ray/multiwavelength flux-ratio features (`xray_ratio_{b,h,s}_<band>` = `log10(F_x) − log10(F_band)`, one set per X-ray band against every converted MW band) are built directly in `unsupervised.ipynb` from these converted fluxes.

Columns *not* present in the current catalog release, so not in the zero-point table: `PS21P_y`, Gaia `BP`/`RP`, WISE `W4`. If a future catalog version adds any of these, `ZERO_POINT_JY` needs a corresponding entry before converting them — it will silently skip anything not listed.

### Verification discipline for physical-units work

Unit conversions and log transforms are error-prone in ways that look fine numerically but destroy downstream ML performance. Before applying any conversion to all bands: (1) show median magnitude and implied median flux for one AB-system band and one Vega-system band, (2) check the median falls within the physically-plausible depth of the source survey (e.g. Pan-STARRS 3π 5σ ≈ r=22; WISE W1 completeness ≈ 17), (3) where possible, cross-check against an independent survey's already-calibrated flux on overlapping rows (see `cross_check_mag_zero_point` in `src/features.py`, demonstrated in `unsupervised.ipynb`). Only after these pass, apply to all bands.

### Parallel research configurations

`unsupervised.ipynb` contains parallel "Reduced Dataset Case" and "General UMAP case" code paths (~cell 29 vs. the exclude_cols/PCA cells that follow). These are two independent, first-class research configurations — a curated ~28K-row subset with no imputation needed vs. the complete ~170K-row catalog with heavier imputation — not a fallback pair where one supersedes the other; both are used for comparing clean-but-biased vs. complete-but-noisier results. Verify any new pipeline stage against **both** branches, not just whichever one you happened to execute last: type/shape mismatches between them (e.g. one producing a DataFrame, the other a bare ndarray, for a variable shared cells downstream both rely on) are exactly how an untested path silently diverges.

As of this writing: the reduced-case pipeline runs end to end (imputation → StandardScaler → HDBSCAN → LightGBM classifier → SHAP) with no known errors. The general-case pipeline is verified through StandardScaler/PCA/a single UMAP fit at full scale; HDBSCAN and everything downstream of it on the full 170K-row dataset remain unverified — not due to a known code issue, but because a single HDBSCAN config didn't complete within 20 minutes at that scale (the notebook's own grid sweeps 8 configs, so a full run there is substantially more expensive still).

## Notebook pipeline (conceptual order)

Several notebooks (`data_explore.ipynb`, `new.ipynb`, `unsupervised.ipynb`) are variants/iterations of the same overall pipeline rather than strictly sequential steps. The pipeline, in the order it's meant to be reasoned about:

1. **Load & merge** — read the main CSV, cross-match with CSC spectral fits and DESI, normalize `name` keys, coerce object columns/`" "` to NaN, fix boolean columns
2. **Feature engineering & labeling** — build WISE/optical/IR colors and X-ray-optical ratios, compute gold labels (`agn_positive`/`agn_negative`) and silver score/labels, produce `label_3class`
3. **Dimensionality reduction** — drop label/metadata columns from the feature matrix, keep numeric columns, drop columns with too much missingness (typical cutoff 90-100%), add missing-value indicator columns, impute (median), scale (`StandardScaler`), reduce via PCA before running UMAP (and separately PaCMAP) with multiple `n_neighbors` sweeps
4. **Clustering** — HDBSCAN over the UMAP/PCA embedding, evaluated with silhouette score, to find candidate substructure (e.g. AGN vs galaxy vs silver-sample separation)
5. **Classification pipeline** — "UMAP -> Clustering (HDBSCAN/KMeans) -> cluster labels -> classifier (LightGBM/CatBoost/RandomForest) -> SHAP" — cluster assignments or gold labels are used as training targets; models are evaluated with `StratifiedKFold` + ROC-AUC/average precision/Brier score, and SHAP is used for feature-importance explainability
6. **Visualization** — scatter plots of UMAP embeddings colored by `label_3class` (unlabeled/gray, non-AGN/blue, silver AGN/orange, confirmed AGN/red) and WISE color-color (Mateos wedge) diagrams, saved to `output/figures/`

When modifying or extracting logic from these notebooks (e.g. into `src/`), preserve the gold/silver/unlabeled distinction and the "priority to positive AGN labels over non_agn" conflict-resolution rule — both are load-bearing for downstream model training.

## References

Research papers live in `references/`. See `references/README.md` for an
index of what each covers. When implementing new features or making
methodological choices, consult the relevant paper first and cite the
section in code comments (e.g. `# following Smith 2024, §3.2`).