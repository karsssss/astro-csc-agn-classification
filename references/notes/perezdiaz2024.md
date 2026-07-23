# Pérez-Díaz et al. 2024 — Unsupervised ML for the Classification of Astrophysical X-ray Sources

**Citation:** Pérez-Díaz, V. S., Martínez-Galarza, J. R., Caicedo, A., & D'Abrusco, R. 2024, MNRAS, 528, 4852. https://doi.org/10.1093/mnras/stae260
**File:** `references/perezdiaz2024_csc_unsupervised_gmm.pdf` (20 pp.)
**One-line summary:** Classifies CSC2 sources into probabilistic classes using **only X-ray properties** (no MW/optical/IR features at all) via a Gaussian Mixture Model (GMM) clustering step followed by a Mahalanobis-distance "cluster-then-label" step, cross-checked against sparse SIMBAD labels; aggregates 10 SIMBAD classes into 4 physically-motivated superclasses (AGN, Seyfert, XB, YSO). A genuinely different methodological path from the supervised MW pipelines in [[yang2022]] and [[kumaran2023]] — same underlying CSC problem, opposite starting assumption (no training set required, no MW data used at all).

## 1. CSC features used — hardness ratios + variability + spectral fits, no MW at all

**12 features total** (Table 1), all purely X-ray, extracted per-observation (detection level, not master-source level):

- **3 hardness ratios**: `hard_hm` (hard−medium), `hard_hs` (hard−soft), `hard_ms` (medium−soft) — same HR concept as Yang's `HR_ms`/`HR_hm`/`HR_h(ms)` and Kumaran's `var_*` hardness features, but computed only in the broad/hard/medium/soft bands (no ultrasoft distinction).
- **2 spectral-fit parameters**: `bb_kt` (blackbody temperature of best-fit absorbed blackbody model) and `powlaw_gamma` (photon index of best-fit absorbed power-law model) — **CSC-provided spectral fits used as direct ML features**, not just a filtering criterion. Neither Yang nor Kumaran use spectral-fit parameters as classifier inputs (Yang only uses them adjacent, e.g. our own `powlaw_gamma`/`has_csc_spectral_fit` in `unsupervised.ipynb` is closer to this paper's approach than to Yang's).
- **7 variability features**, all per-band (b/h/s = broad/hard/soft): `var_prob_*` (intra-observation Gregory–Loredo variability probability, 3 features), `var_ratio_*` (ratio of flux variability mean to its standard deviation, 3 features), and `var_newq_b` (broad-band only: proportion of the average min/max count rate relative to the mean count rate).
- **Preprocessing** (Table 2): `bb_kt`, `powlaw_gamma`, `var_ratio_*`, `var_newq_b` are log-transformed (offsetting zeros by 1/10 the minimum non-zero value) then MinMax-scaled to [0,1]; hardness ratios and `var_prob_*` are MinMax-scaled only (already reasonably well-behaved distributions).
- **Data selection**: CSCview query requiring broadband flux significance > 5σ and non-null `powlaw_gamma`/`bb_kt` (i.e., a spectral fit had to converge) → 37,878 unique per-observation entries → dropping any row with a missing value in the 12 chosen features → **29,655 detections** used for GMM clustering. Unlike Yang/Kumaran, **no imputation of any kind is used** — rows with any missing feature are simply excluded rather than imputed or flagged, since the feature set is small and mostly complete for well-detected sources.
- **Deliberately excludes sky position** (`ra`/`dec`/`gal_l`/`gal_b`) as a feature, specifically to avoid CSC's non-uniform, skewed sky coverage introducing a training bias — the opposite choice from us and Yang, who use Galactic latitude/reddening maps as a feature/correction (see §6 below).

## 2. GMM component selection: BIC + elbow method

- GMM fit via scikit-learn's `GaussianMixture`, EM algorithm, **`covariance_type='full'`** (each Gaussian component gets its own full covariance matrix — appropriate since they don't know a priori how components' shapes/orientations should look), `random_state=42` for reproducibility.
- Number of components K is the key hyperparameter. **BIC alone doesn't give a clean answer**: BIC decreases roughly monotonically as K grows (more components always improve likelihood, at the cost of more parameters — no clear minimum before overfitting territory).
- They instead take the **gradient (elbow) of the BIC curve** vs. K (swept 2–20): the slope of BIC decline flattens sharply around **K=6** and converges to ~0 for K≥7, meaning additional components beyond 6 don't meaningfully improve the fit. **K=6 is chosen from this elbow, cross-checked against domain knowledge and manual inspection of cluster feature distributions**, not from BIC's minimum directly.
- Resulting clusters correlate with **Galactic latitude** even though position wasn't a feature: clusters 3/4 are dominated by Galactic-plane detections (34%/29% at |b|<5°), clusters 1/2 are dominantly extragalactic (12.6%/9.8% at |b|<5°) — i.e., the X-ray hardness/variability/spectral-shape features alone recover a Galactic-vs-extragalactic split similar to what position or reddening-based features would give directly.
- Each cluster is a **mix of multiple SIMBAD classes**, not a 1:1 mapping — e.g. cluster 2 is dominated by large/small accretors (QSO/AGN/Seyfert plus HMXB/XB), cluster 4 is >90% young stars. This many-classes-per-cluster reality is exactly why a further per-cluster classification step (§3) is needed on top of the clustering.

## 3. Mahalanobis-distance classification step (the "cluster-then-label" step)

This is the paper's core novel contribution — a **semi-supervised, cluster-then-label method** (in the sense of Zhu & Goldberg 2009): GMM does the unsupervised clustering; a distance-based step then assigns probabilistic classes **within each cluster independently**, using whatever labeled members that cluster happens to contain.

1. Cross-match all 29,655 clustered detections against **SIMBAD** (1″ radius, `otype` field) — only a minority end up labeled; unlabeled majority remain to be classified.
2. **Within each of the 6 clusters**, group the labeled (SIMBAD) members by class and compute each class's centroid (mean vector) and covariance in the 12-feature space, **restricted to that cluster's own members**.
3. For every unlabeled detection in that cluster, compute the **Mahalanobis distance** `D_P(x) = sqrt((x−μ)ᵀ Σ⁻¹ (x−μ))` (Eq. 6) to each class's centroid/covariance present in that cluster.
4. Convert the resulting vector of distances to a probability distribution over classes via a **softmin** (Eq. 7: `softmin(x_n) = softmax(−x_n)`, i.e. exponentiate the negative distances so smaller distance → higher probability, normalized to sum to 1).
5. Repeat per-detection for all clusters. Output: a full probability distribution over classes for every detection, not just a point label.

**Important caveat they flag explicitly**: probabilities are only meaningful **within a given detection's own cluster** — a 100% probability in one cluster's local reference frame is not comparable to a 100% probability from a different cluster, since each cluster has its own class centroids/covariances. This is conceptually different from Yang's CT (global, uncertainty-normalized across all 8 classes at once) and Kumaran's CMP (a single global LightGBM probability) — **Mahalanobis-based confidence here is inherently local/relative, not absolute**.

## 4. Soft/hard voting scheme (per-detection → per-source master class)

Since a CSC master source can have multiple detections (77.7% have exactly 1, 11.4% have 2, 10.1% have 3–10, 0.8% have >10), a **master class per source** is needed by combining its detections' individual (possibly disagreeing) classifications:

- **Hard voting**: majority vote — the class assigned to the most detections of that source (Eq., argmax of count).
- **Soft voting**: average each detection's full probability vector across all of a source's detections, then take the argmax of the mean probability (uses the full probabilistic output, not just the point label).
- **Agreement**: hard and soft voting agree for the large majority of sources — **Cohen's κ = 0.94** between the two methods; only **485/8756 sources (6%)** get different master classes from the two methods ("ambiguous" sources, tabulated separately from "uniquely classified" sources). Several example ambiguous cases are walked through in the paper (e.g. a pulsar/Orion-variable degeneracy, an AGN/YSO degeneracy) — generally traceable to genuinely intermediate hardness/variability properties rather than a pipeline bug.

## 5. Aggregated 4-class scheme

Of ~125 possible SIMBAD `otype` classes matched, only **10 have enough labeled members** to be usable for the per-cluster Mahalanobis training: QSO, AGN, Seyfert_1, Seyfert_2, HMXB, LMXB, XB, YSO, TTau*, Orion_V*. These are aggregated into **4 physically motivated superclasses**, directly analogous to accretion-power scale rather than catalog convention:

- **Large accretors (AGN)**: QSO + AGN (central object is a supermassive black hole)
- **Extended large accretors with optical lines (Seyfert)**: Seyfert_1 + Seyfert_2
- **Small accretors (XB)**: HMXB + LMXB + XB (compact-object binaries — neutron star or stellar-mass black hole)
- **Young stars (YSO)**: YSO + TTau* + Orion_V*

Final output: probabilistic classification for **8756 sources / 14,507 detections**.

## 6. Performance and confusion patterns

- **Detection-level (10-class) confusion** (Fig. 4): QSO correctly classified only 37% of the time, with **30% misclassified as Seyfert_1** — attributed to genuinely similar QSO/Seyfert-1 X-ray spectra when no redshift/distance is available to break the degeneracy. HMXB correct >50%, confused as LMXB ~8%; conversely **LMXB misclassified as HMXB 44% of the time** — small accretors are harder to separate from each other than large accretors are from small ones.
- **Aggregated (4-class) confusion** (Fig. 5): large accretors (AGN) correctly labeled ~75%, misclassified as XB (small accretor) only ~19–20% of the time; small accretors (XB) correct ~66%, misclassified as large accretor ~20%. **YSO is by far the most robust class — almost never confused with the other three groups** (>85–88% correct), the same "young stars are the easiest class" pattern both Yang and Kumaran also report.
- **Comparison against Yang's MUWCLASS catalogs and other references (Table 4)**, cross-matching their probabilistic labels against external catalogs:

  | Reference catalog | Class | Precision | Recall |
  |---|---|---|---|
  | Yang CCGCS (confident, CT≥2) | AGN+Seyfert | 0.94 | 0.62 |
  | Yang CCGCS | XB | **0.05** | 0.51 |
  | Yang CCGCS | YSO | 0.36 | 0.72 |
  | Tranin 4XMM-DR10 | AGN+Seyfert | 0.76 | 0.62 |
  | Tranin 4XMM-DR10 | XB | 0.42 | 0.45 |

  **AGN+Seyfert precision is high (0.76–0.94) when cross-checked against MW-based catalogs, but XB precision is very low (0.05–0.42)** — i.e., a large fraction of what this X-ray-only method calls "small accretor/XB" is actually AGN/Seyfert by MW-informed standards. This is the paper's clearest demonstrated limitation: **without MW colors, large-vs-small accretor separation is unreliable in the "small accretor" direction specifically**, even though the reverse (correctly recognizing true large accretors) works well.
- Astrophysical sanity checks hold up despite no position feature being used: YSO detections cluster tightly along the Galactic plane, AGN/Seyfert concentrate off-plane, XB is intermediate — recovering the expected Galactic/extragalactic split purely from clustering on spectral/variability properties (Fig. 8).
- Feature-distribution analysis (§5.2) explicitly frames results in terms of the **AGN unified model**: Seyfert 2 galaxies show harder/more absorbed soft-band spectra than Seyfert 1 (consistent with torus obscuration along the line of sight); YSOs show the highest broad+soft-band variability of any class (flares/coronal activity); XRBs show a wider range of hardness ratios than AGN/Seyferts, consistent with faster spectral-state transitions in accreting compact binaries.

## 7. Where this complements vs. contradicts the supervised approaches

**Complements:**
- Requires **no labeled training set and no MW cross-match at all** — directly addresses the exact limitation both Yang and Kumaran flag as a bottleneck (small, potentially-biased, hard-to-construct training sets; MW incompleteness for a large fraction of CSC sources). Useful as an **independent, assumption-light cross-check** on supervised classifications, and applicable to the majority of CSC sources that have no optical/IR counterpart at all (where Yang/Kumaran's MW features are simply unavailable).
- The **cluster-then-label structure is essentially the same idea already sketched in our own `unsupervised.ipynb`** notebook pipeline note — "UMAP → Clustering (HDBSCAN/KMeans) → cluster labels → classifier → SHAP" — except this paper substitutes a **Mahalanobis-distance nearest-centroid rule** for a trained classifier at the per-cluster step, which is much cheaper and doesn't require a second training pass. Worth considering as a lighter-weight alternative to the LightGBM/CatBoost-on-cluster-labels step we currently do.
- Full **probabilistic output + hard/soft voting for multi-detection sources** is a clean, reusable pattern for our own pipeline wherever a CSC master source has multiple observations — we don't currently have an explicit per-detection→per-source aggregation step.

**Contradicts / limitations relative to Yang & Kumaran:**
- **Confirms, via direct external validation, that MW information is specifically needed to separate AGN from small-accretor XRBs** (Table 4's very low XB precision) — this is the single strongest piece of independent evidence in favor of the MW-feature-heavy approach both Yang and Kumaran (and our own pipeline) take, rather than an argument against it.
- **Deliberately excludes position/reddening entirely** to avoid sky-coverage bias — the opposite choice from Yang's explicit SFD-based AGN dereddening (flagged as a gap in the Yang note) and from our own use of `gal_l`/`gal_b`. If we adopt Yang's reddening correction, we're taking a position this paper explicitly argues against for different (non-AGN-specific) reasons — worth being deliberate about that trade-off rather than importing both papers' choices uncritically.
- Confidence/probability is **only locally comparable within a cluster**, unlike Yang's CT or Kumaran's CMP — not directly portable as a global "confidence threshold" concept if we wanted to borrow this method's uncertainty framework.
- Uses **spectral-fit parameters (`bb_kt`, `powlaw_gamma`) directly as classifier features**, not just as a data-quality filter — our current notebooks only use `powlaw_gamma` to build `has_csc_spectral_fit`/silver-score evidence, not as a raw numeric feature; this paper suggests it's usable directly (with log+MinMax preprocessing) as an actual ML input.
