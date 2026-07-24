# Lab notebook

Chronological record of research decisions that aren't otherwise visible from
the code or `CLAUDE.md` — things that were tried, superseded, or deliberately
dropped, and why.

Entries are in reverse chronological order (newest at top). Undated context
and conventions live in this header.

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
