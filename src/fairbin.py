"""
Fair Bin: split one numeric column into k bins so that the worst bin's error
rate is as low as possible.

The bins come from training rows only, with their own model:

1. A decision stump predicts every training row out of fold (5 folds), so each
   row's error is counted by a model that did not train on it.
2. Errors and rows are counted per distinct value of the column.
3. Every candidate bin, a run of consecutive values, is scored by the upper end
   of the Wilson interval on its error rate. z is Bonferroni-corrected for the
   n(n+1)/2 candidate bins over n values, so a bin with few rows gets a wide
   bound and the DP avoids tiny bins without a size floor.
4. A DP picks the k bins minimising the largest score. That pins down only the
   worst bin, so a second stage keeps its score as a ceiling and, among the
   partitions under it, takes the one whose smallest bin has the most rows.

Bins are [cuts[j], cuts[j+1]); values past either end fall in the first or
last bin. The bounded extension (fair_bin_bounded) adds limits on every bin's
share of the rows.

    cuts = fair_bin(train, target="income", column="age", k=10, seed=0)
    data["age"] = apply_bins(data["age"], cuts)

Code was re-organized using Claude for one cohesive py file.
"""
from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

FOLDS = 5
# this is the wilson scoring 
ALPHA = 0.05


def oof_predictions(X, y, seed, folds=FOLDS):
    """
    Out-of-fold decision-stump prediction for every row. Stratified folds when
    every class has at least `folds` rows, plain shuffled folds otherwise.

    Change Folds as need, paper uses 5 folds, probably can use 2-3 to lower time 
    with similar results.
    """
    y = np.asarray(y)
    # OOF
    splitter = StratifiedKFold if pd.Series(y).value_counts().min() >= folds else KFold
    # proxy model M hat
    stump = DecisionTreeClassifier(max_depth=1, random_state=seed)
    return cross_val_predict(stump, X, y, cv=splitter(n_splits=folds, shuffle=True, random_state=seed))


def error_counts(x, y, pred):
    """Per distinct value of x, ascending: (values, misclassified rows, rows).
    
    Get the raw error estimates from P
    """
    values, inv = np.unique(np.asarray(x, dtype=float), return_inverse=True)
    wrong = np.asarray(pred) != np.asarray(y)
    return values, np.bincount(inv, weights=wrong).astype(float), np.bincount(inv).astype(float)


def bonferroni_z(n_values, alpha=ALPHA):
    """One-sided z corrected for the n(n+1)/2 candidate bins: Phi^-1(1 - alpha / (n(n+1)/2))."""
    return NormalDist().inv_cdf(1 - alpha / (n_values * (n_values + 1) / 2)) # apply alpha value


def wilson_upper(e, n, z):
    """Upper end of the Wilson score interval for e errors in n rows; inf where n <= 0.
    
    Pre computation stage, modify raw estimates
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.clip(e / n, 0.0, 1.0)
        z2 = z * z # z correction
        # confidence bound
        upper = (p + z2 / (2 * n) + z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / (1 + z2 / n)
    return np.where(n > 0, upper, np.inf)


def _minimax(V, k):
    """
    Split values 0..n-1 into k contiguous bins minimising the largest V[a, b],
    where V[a, b] scores the bin of values a..b-1 (inf: not allowed). Ties go
    to the first minimum. Returns (that largest V, first value of bins 2..k).

    Basis for the DP algorithm + backtracking 
    """
    n = V.shape[0] - 1
    best = np.full(n + 1, np.inf)
    best[0] = -np.inf
    parents = []
    for _ in range(k):
        cand = np.maximum(best[:, None], V)
        parents.append(cand.argmin(axis=0))
        best = cand.min(axis=0)
    starts, b = [], n
    for p in reversed(parents):
        b = int(p[b])
        starts.append(b)
    return float(best[n]), starts[::-1][1:]


def bin_sums(a):
    """S[a, b] = a[a] + ... + a[b-1]: the total over the bin of values a..b-1."""
    c = np.concatenate([[0.0], np.cumsum(a)])
    return c[None, :] - c[:, None]


def two_stage(V, size, k):
    """
    k contiguous bins minimising the largest V; then, with that as a ceiling,
    the partition whose smallest bin (by size) is largest. V is modified: empty
    bins, and a last bin of a single value (as in the original DP), are ruled out.
    Shared with baselines.equal_freq, which scores a bin by its size.

    Returns (largest V, first value index of bins 2..k).
    """
    n = V.shape[0] - 1
    if k >= n:
        raise ValueError(f"Cannot create {k - 1} splits from {n} positions")
    V[np.tril_indices(n + 1)] = np.inf
    V[n - 1, n] = np.inf
    worst, _ = _minimax(V, k)
    if not np.isfinite(worst):
        raise ValueError("No partition fits the size limits")
    # stage 2: under the ceiling, maximise the smallest bin = minimise the largest -size
    _, starts = _minimax(np.where(V <= worst, -size, np.inf), k)
    return worst, starts


def choose_cuts(values, errors, rows, k, z, min_rows=None, max_rows=None):
    """
    The two-stage DP over per-value counts from error_counts.

    min_rows, max_rows: optional limits on every bin's rows; ValueError when no
    partition fits them.

    Returns (worst bin's Wilson bound, cuts), where cuts is
    [min value, first value of bins 2..k, max value].

    Main function (uses minimax, applies the wilson bounds)
    """
    size = bin_sums(rows)
    V = wilson_upper(bin_sums(errors), size, z)
    if min_rows is not None:
        V[size < min_rows] = np.inf
    if max_rows is not None:
        V[size > max_rows] = np.inf
    worst, starts = two_stage(V, size, k)
    return worst, [float(values[0])] + [float(values[s]) for s in starts] + [float(values[-1])]


def choose_cuts_bounded(values, errors, rows, k, z, min_share=None, max_share=None):
    """
    choose_cuts with every bin holding between min_share and max_share of the
    rows: e.g. the band 1/(2k)..2/k (no bin under half or over twice an
    equal-frequency bin), or min_share=0.05 alone.

    Ties can make the limits infeasible: one value holding more rows than the
    cap, or too few rows left to give every bin the floor. The cap is then
    raised and the floor lowered 5% a step until a partition fits.

    Returns (worst bin's Wilson bound, cuts, (min_rows, max_rows) used, relaxation steps).
    """
    n = rows.sum()
    lo = None if min_share is None else int(np.ceil(min_share * n))
    hi = None if max_share is None else int(np.floor(max_share * n))
    steps = 0
    while True:
        try:
            return (*choose_cuts(values, errors, rows, k, z, lo, hi), (lo, hi), steps)
        except ValueError:
            if not lo and (hi is None or hi >= n):
                raise
            hi = None if hi is None else int(np.ceil(hi * 1.05))
            lo = int(np.floor(lo / 1.05)) if lo else None
            steps += 1


def tally(train, target, column, seed):
    """error_counts of train[column] from out-of-fold stump predictions on the training rows."""
    pred = oof_predictions(train.drop(columns=[target]), train[target], seed)
    return error_counts(train[column], train[target], pred)


def fair_bin(train, target, column, k, seed, alpha=ALPHA):
    """Cuts for k bins of train[column], chosen from the training rows `train` alone."""
    values, errors, rows = tally(train, target, column, seed)
    return choose_cuts(values, errors, rows, k, bonferroni_z(len(values), alpha))[1]


def fair_bin_bounded(train, target, column, k, seed, min_share=None, max_share=None, alpha=ALPHA):
    """
    fair_bin with every bin between min_share and max_share of the training
    rows (choose_cuts_bounded). The band: min_share=1/(2k), max_share=2/k.
    """
    values, errors, rows = tally(train, target, column, seed)
    z = bonferroni_z(len(values), alpha)
    return choose_cuts_bounded(values, errors, rows, k, z, min_share, max_share)[1]


MIN_DISTINCT = 20  # columns with fewer distinct values are not binned
G = 2000  # columns with more distinct training values are pre-grouped before the DP


def pregroup(keys, num, den, rows, G):
    """Pool positions into <= G consecutive groups of about equal rows."""
    cum = np.cumsum(rows)
    targets = cum[-1] * np.arange(1, G) / G
    starts = np.unique(np.concatenate([[0], np.searchsorted(cum, targets, side="right")]))
    starts = starts[starts < len(keys)]
    return (keys[starts], np.add.reduceat(num, starts), np.add.reduceat(den, starts),
            np.add.reduceat(rows, starts))


def binnable(data, target, first=None, skip=()):
    """Numeric non-target columns with at least MIN_DISTINCT distinct values, `first` first, minus `skip`."""
    out = [first] if first else []
    for c in data.columns:
        if c in (target, first):
            continue
        if pd.api.types.is_numeric_dtype(data[c]) and data[c].nunique() >= MIN_DISTINCT:
            out.append(c)
    return [c for c in out if c not in skip]


def column_cuts(values, errors, rows, k, alpha=ALPHA, groups=G):
    """
    Unbounded Fair Bin cuts for one column's counts. Above `groups` distinct
    values they are pooled first (pregroup), z from the group count, and the
    last cut is put back at the column's maximum.
    """
    last = float(values[-1])
    if len(values) > groups:
        values, errors, _, rows = pregroup(values, errors, rows, rows, groups)
    cuts = choose_cuts(values, errors, rows, k, bonferroni_z(len(values), alpha))[1]
    cuts[-1] = last
    return cuts


def fair_bin_columns(train, target, columns, k, seed, alpha=ALPHA, groups=G):
    """
    Multi-column Fair Bin (allcols_pregroup.py's fairbin_zbonf): one
    out-of-fold stump prediction on the raw training rows gives every column
    its error counts, and each column gets its own cuts (column_cuts). A
    column whose cuts fail (e.g. no more than k distinct values) stays raw.

    Returns ({column: cuts}, [failed columns]).
    """
    pred = oof_predictions(train.drop(columns=[target]), train[target], seed)
    out, failed = {}, []
    for c in columns:
        try:
            out[c] = column_cuts(*error_counts(train[c], train[target], pred), k, alpha, groups)
        except Exception as e:  # noqa: BLE001 -- column stays raw, as in the experiments
            failed.append(f"{c}: {type(e).__name__}")
    return out, failed


def bin_index(column, cuts, right=False):
    """
    Each value's bin, 0..len(cuts) - 2. Bins are [cuts[j], cuts[j+1]), or with
    right=True (cuts[j], cuts[j+1]] as pd.cut makes them; values past either end
    fall in the first or last bin.
    """
    inner = np.asarray(cuts, dtype=float)[1:-1]
    return np.searchsorted(inner, np.asarray(column, dtype=float), side="left" if right else "right")


def apply_bins(column, cuts, right=False):
    """Each value replaced by its bin's lower cut (a Series if given one)."""
    labels = np.asarray(cuts, dtype=float)[:-1][bin_index(column, cuts, right)]
    if isinstance(column, pd.Series):
        return pd.Series(labels, index=column.index, name=column.name)
    return labels
