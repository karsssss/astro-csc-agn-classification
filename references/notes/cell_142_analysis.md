# Analysis: `unsupervised.ipynb` cell 142 ("Multi-iterative model")

Requested before this cell's logic is ported into `classification/semi_supervised.py`
(see `PLAN.md` open question 3). **Recommendation up front: do not port this
cell as-is** — it does not do what its comment/section title claims, and it
most likely raises an exception on its second iteration. Details below.

## The cell, verbatim (for reference)

```python
# Multi-iterative model

final_lgbm = models["lightgbm"]
final_cat = models["catboost"]

for i in range(0, 10):
    # LightGBM
    final_lgbm.fit(X_labeled, y_labeled)
    unknown_score_lgbm = final_lgbm.predict_proba(X_unknown)[:, 1]

    # CatBoost
    final_cat.fit(X_labeled, y_labeled)
    unknown_score_cat = final_cat.predict_proba(X_unknown)[:, 1]


    # Defragmentate memory
    X_labeled = df.copy()

    X_labeled.loc[df_unknown.index, "score_supervised_lgbm"] = unknown_score_lgbm
    X_labeled.loc[df_unknown.index, "score_supervised_cat"] = unknown_score_cat
    X_labeled.loc[df_unknown.index, "score_supervised_mean"] = (unknown_score_lgbm + unknown_score_cat) / 2
```

## State entering the loop (from earlier cells, still in scope)

- `models` (cell 139): `{"lightgbm": LGBMClassifier(...), "catboost": CatBoostClassifier(...)}`, both binary, `class_weight`/`auto_class_weights="Balanced"`.
- `X_labeled` (cell 139): `X_aug.loc[df_labeled.index].copy()` — the **encoded numeric feature matrix** (imputed/augmented with `_missing` indicator columns), restricted to the ~gold-labeled rows only. Call its row count `n_labeled`.
- `y_labeled` (defined earlier, e.g. cell 121/126): an `int8` numpy array of length `n_labeled`, the binary AGN/non-AGN target for those same rows, in the same row order as the `X_labeled` that existed *before this cell ran*.
- `df_unknown` (cell 140): `df[df["label_3class"] == -1].copy()` — the ~140K unlabeled rows.
- `X_unknown` (cell 140): `X_aug.loc[df_unknown.index].copy()` — the encoded feature matrix for those unlabeled rows. Call its row count `n_unknown`.
- `df` (module-level, ~170K rows): the **original**, mostly-raw, mixed-dtype catalog DataFrame — object/string columns, label columns, coordinates, everything — not the encoded/imputed feature matrix.

## Iteration-by-iteration trace

**Iteration `i = 0`:**
- *Reads:* `X_labeled` (the correct `n_labeled`-row encoded feature matrix, from before the loop), `y_labeled` (length `n_labeled`).
- `final_lgbm.fit(X_labeled, y_labeled)` — valid: shapes match. This refits the exact same model object `final_lgbm` that cell 140 already fit on the exact same data (a harmless but redundant refit).
- `unknown_score_lgbm = final_lgbm.predict_proba(X_unknown)[:, 1]` — valid, `X_unknown` is still the correct encoded unlabeled-row matrix. Produces a length-`n_unknown` score array.
- Same for `final_cat` / `unknown_score_cat`.
- *Writes:* `X_labeled = df.copy()` — **`X_labeled` is now reassigned to the full, raw, ~170K-row original catalog** (not the encoded feature matrix, not filtered to labeled rows, and not even the same number of rows as before: `n_labeled` → ~170K).
- `X_labeled.loc[df_unknown.index, "score_supervised_lgbm"/"score_supervised_cat"/"score_supervised_mean"] = ...` — three new columns are added to this brand-new `X_labeled` (which, again, is just a copy of `df`), populated only for the unlabeled rows.
- *Net effect of iteration 0:* one legitimate (if redundant) model fit + a legitimate score computation, followed by a variable-name-conflict bug: the name `X_labeled` — which every earlier cell in this notebook uses to mean "encoded training feature matrix" — now points to something completely different ("raw catalog + 3 scratch score columns").

**Iteration `i = 1`:**
- *Reads:* `X_labeled` is now the object created at the end of iteration 0 — the ~170K-row raw `df` copy, **not** an encoded numeric matrix. `y_labeled` is unchanged: still the original length-`n_labeled` array from before the loop ever started.
- `final_lgbm.fit(X_labeled, y_labeled)` — **shape mismatch**: `X_labeled` has ~170K rows, `y_labeled` has `n_labeled` (~30K, per `CLAUDE.md`'s gold/silver split) rows. scikit-learn/LightGBM's `fit(X, y)` validates `n_samples_X == n_samples_y` and raises `ValueError: Found input variables with inconsistent numbers of samples: [...]` before any training happens. Separately, even if the row counts happened to match, `X_labeled` at this point contains non-numeric/object-dtype and string columns (`name`, label strings, etc.) that were never encoded — `LGBMClassifier.fit` would also reject this on dtype grounds.
- **The loop almost certainly stops here with an unhandled exception on its second pass.** (Whether it was ever run to completion, and what error message actually appeared, isn't recoverable from the notebook's saved outputs at cell 142, which are empty in `notebooks/unsupervised.ipynb`'s current saved state.)

**Iterations `i = 2..9`:** unreachable given the above; not analyzed further.

## What each iteration actually does (plain English)

- **Iteration 0:** "Refit the two models on the real labeled data (again — this was already done one cell earlier), score the unlabeled pool, then throw away the feature matrix I was just using and replace it with something unrelated."
- **Iteration 1 onward:** "Try to refit the same models using the *unrelated replacement* as if it were still the training feature matrix — this fails."

## What the code appears intended to do

Given the surrounding section is explicitly semi-supervised (cells 126-148:
stage-1 OOF ensembling → pseudo-positive candidate selection →
`pseudo_positive_bagging` → PNU bagging via `bagging_pnu_scores`), and given
the cell's own title, "Multi-iterative model," the intent looks like a
**self-training loop**: repeatedly (1) score the unlabeled pool with the
current model, (2) select some high-confidence unlabeled rows as pseudo-labels,
(3) fold those pseudo-labeled rows into the training set (typically with a
reduced sample weight), (4) refit on the expanded set, (5) repeat, so that
each iteration's training set — and therefore the model — gradually improves
using its own confident predictions on the unlabeled pool. `pseudo_positive_bagging`
(cell 132) and `bagging_pnu_scores` (cell 144) both implement a correct,
if different (bootstrap-ensemble rather than iterative-refit), version of this
same idea elsewhere in the same notebook.

## The gap

1. **No pseudo-label promotion mechanism at all.** The loop never thresholds
   `unknown_score_lgbm`/`unknown_score_cat`, never selects a confident subset
   of `X_unknown`, and never appends any unlabeled rows (or their inferred
   labels) into `y_labeled`. Every ingredient present in `pseudo_positive_bagging`
   (confidence cutoff, weighted inclusion of pseudo-positives) is absent here.
2. **`y_labeled` never changes across iterations**, so even if `X_labeled`
   weren't corrupted, there would be nothing "multi-iterative" about
   re-fitting on an unchanged `(X, y)` pair ten times — LightGBM/CatBoost are
   deterministic given the same data and `random_state`, so iterations 2-9
   (had they run) would have been no-ops.
3. **The line `X_labeled = df.copy()` looks like a copy-paste/variable-name
   mistake.** The comment directly above it — `# Defragmentate memory` — matches
   the pattern used correctly one cell earlier (cell 140: `df = df.copy()`,
   to defragment `df`'s block manager after many `.loc[...] = ...` column
   assignments). It looks like that idiom was copied into this loop but
   applied to the wrong variable (`X_labeled` instead of `df`), silently
   swapping out the loop's own training data.
4. **Nothing downstream reads any output of this loop.** Cell 143
   (`df.groupby("label_3class")["score_supervised_lgbm"].describe()`) reads
   `df["score_supervised_lgbm"]`, which was set in cell 140 *before* this loop
   ever ran and is never touched by the loop (the loop only ever mutates its
   local `X_labeled`, never `df` itself). So independent of whether the loop
   crashes on iteration 1, its output has no observable effect on the rest of
   the notebook — it is currently dead code even in the best case.

## Recommendation

Do not port this cell's logic into `classification/semi_supervised.py`.
If a self-training / iterative-refit stage is still wanted alongside
`pseudo_positive_bagging` and `bagging_pnu_scores`, it should be designed fresh
(explicit pseudo-label confidence threshold, explicit promotion of selected
unlabeled rows into the training set with an explicit reduced weight, explicit
re-fit on the *combined* labeled + promoted-pseudo-labeled set each round) —
not repaired in place, since the current version has no working self-training
mechanics to preserve, only the crash to fix.

Waiting on your decision (fix / rewrite / delete) before touching this in
`classification/semi_supervised.py` — see `PLAN.md` open question 3.
