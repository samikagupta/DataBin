"""
Everything recorded for one (binning, eval model, seed).

Rates are fractions in [0, 1] (MRE 0.0847 is 8.47 %). Unless noted, they are
measured on the eval model's held-out test rows: 30 % by default, the same
split as gen_bins.train_test (see split).

    mre           worst bin's misclassification rate, over bins with test rows
                  (the max of bin_metrics.frac_equal_bin_error_full, / 100)
    max_fpr       worst bin's FPR, over bins with >= 1 actual negative  [binary]
    max_fnr       worst bin's FNR, over bins with >= 1 actual positive  [binary]
    overall_fpr   false positives / actual negatives, whole test set    [binary]
    overall_fnr   false negatives / actual positives, whole test set    [binary]
    accuracy
    precision, recall, f1   of class 1 for binary targets; macro average otherwise
    sd            std (ddof=1) of the per-bin misclassification rates
    per_bin_*     per-bin rate, one entry per bin in bin_cuts order, null if
                  the bin has no qualifying test rows

Full-dataset bin layout (bin_layout):
    n_bins              bins holding at least one row (can be below k)
    smallest_bin_share  rows in the smallest non-empty bin / all rows
    bin_sizes           rows per bin, in bin_cuts order

Support-filtered worst bins. A bin's "support" is that rate's denominator --
the test rows it is actually measured over: all rows for MRE, actual negatives
for FPR, actual positives for FNR. A bin can hold 500 test rows and still have
an FNR support of 1, because only actual positives count; its FNR is then 0 or
1 and sets max_fnr on one row. These columns make that visible:

    min_support         the threshold below, recorded so the row is self-describing
    mre_support         support of the bin that set mre (likewise max_fpr/fnr_support)
    mre_supported       worst bin with support >= min_support, null if no bin
                        qualifies (likewise max_fpr/fnr_supported)
    per_bin_rows        test rows per bin, the MRE support
    per_bin_fpr_support actual negatives per bin, the FPR denominator   [binary]
    per_bin_fnr_support actual positives per bin, the FNR denominator   [binary]

The per-bin supports are recorded so any other threshold can be applied to
saved results without re-running. Not to be confused with min_share, which is
Fair Bin's minimum bin size and applies when the bins are chosen, not scored.

[binary] metrics are blank for multiclass targets.
"""
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Support floor for the *_supported columns. 30 keeps a rate's standard error
# at or below ~0.09, so gaps of ~0.1 between methods mean something. Changing
# it only changes those columns; per_bin_*_support lets any threshold be
# applied to already-recorded results.
MIN_SUPPORT = 30

# Settings from the existing notebooks. n_jobs only parallelises the forest;
# its predictions are identical to n_jobs=1.
MODELS = {
    "rf": lambda seed: RandomForestClassifier(random_state=seed, n_jobs=-1),
    "nn": lambda seed: MLPClassifier(random_state=seed, max_iter=500),
    # The paper's MLP trains on unscaled features; on several datasets it barely
    # beats predicting the majority class. These two scale first.
    "nn_scaled": lambda seed: make_pipeline(StandardScaler(),
                                            MLPClassifier(random_state=seed, max_iter=500)),
    "nn_early": lambda seed: make_pipeline(StandardScaler(),
                                           MLPClassifier(random_state=seed, max_iter=500,
                                                         early_stopping=True, n_iter_no_change=5)),
    "xgb": lambda seed: xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                          random_state=seed),
}


def bin_layout(binned, bounds):
    """
    Bin index of every row, as a position in bounds so per-bin lists line up
    with bin_cuts, plus the full-dataset layout fields.
    """
    edges = np.asarray(bounds, dtype=float)
    idx = np.clip(np.searchsorted(edges, binned, side="right") - 1, 0, len(edges) - 2)
    sizes = np.bincount(idx, minlength=len(edges) - 1)
    nonempty = sizes[sizes > 0]
    return idx, {
        "n_bins": int(len(nonempty)),
        "smallest_bin_share": float(nonempty.min() / len(binned)),
        "bin_sizes": sizes.tolist(),
    }


def _per_bin(rates, n_slots):
    return [round(float(rates[i]), 6) if i in rates.index else None for i in range(n_slots)]


def _per_bin_support(support, n_slots):
    """Like _per_bin, but an absent bin has a support of 0, not an unknown rate."""
    return [int(support[i]) if i in support.index else 0 for i in range(n_slots)]


def _worst(rates, support, prefix, min_support=MIN_SUPPORT):
    """
    Worst bin by `rates`, reported twice: over every bin, and over bins whose
    support reaches min_support.

    Args:
        rates: per-bin rate, indexed by bin position
        support: per-bin denominator of that rate, same index
        prefix: column name of the unfiltered maximum ("mre", "max_fpr", ...)
    """
    if not len(rates):
        return {prefix: np.nan, f"{prefix}_support": 0, f"{prefix}_supported": np.nan}
    worst = rates.idxmax()
    eligible = rates[support.reindex(rates.index).fillna(0) >= min_support]
    return {
        prefix: rates.max(),
        f"{prefix}_support": int(support[worst]),
        f"{prefix}_supported": eligible.max() if len(eligible) else np.nan,
    }


def split(data, target, seed, train_size=0.7):
    """
    gen_bins.train_test with the training fraction as a parameter: the same
    unstratified train_test_split call, so at train_size=0.7 the split is
    identical (check_robust.py verifies this).

    Returns train_x, train_y, test_x, test_y.
    """
    cols = [c for c in data.columns if c != target]
    train_x, test_x, train_y, test_y = train_test_split(
        pd.DataFrame(data, columns=cols), data[target], train_size=train_size, random_state=seed)
    return train_x, train_y, test_x, test_y


def evaluate(data, bin_col, target, binned, bin_idx, n_slots, model, seed, binary, train_size=0.7):
    """
    Train the eval model on the binned data and measure it.

    Args:
        data: loaded dataset; copied, not modified
        binned: bin label per row, replaces data[bin_col]
        bin_idx, n_slots: from bin_layout; bin index per row, number of bins
        model: key of MODELS
        seed: split and model seed
        binary: whether the target has two classes (FPR/FNR are computed only then)
        train_size: training fraction; the rest is the test set (default 70/30)
    """
    d = data.copy()
    d[bin_col] = binned
    train_x, train_y, test_x, test_y = split(d, target, seed, train_size)

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf = MODELS[model](seed).fit(train_x, train_y)
    pred = np.asarray(clf.predict(test_x))
    eval_time = time.perf_counter() - t0

    y = test_y.to_numpy()
    bins = bin_idx[test_x.index.to_numpy()]  # load() gives a RangeIndex
    wrong = pd.Series(pred != y).groupby(bins)
    err, err_n = wrong.mean(), wrong.size()

    avg = "binary" if binary else "macro"
    out = {
        **_worst(err, err_n, "mre"),
        "min_support": MIN_SUPPORT,
        "per_bin_rows": _per_bin_support(err_n, n_slots),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, average=avg, zero_division=0),
        "recall": recall_score(y, pred, average=avg, zero_division=0),
        "f1": f1_score(y, pred, average=avg, zero_division=0),
        "sd": err.std(ddof=1) if len(err) > 1 else 0.0,
        "per_bin_error": _per_bin(err, n_slots),
        "eval_time_s": eval_time,
    }
    if binary:
        neg, pos = y == 0, y == 1
        # fp: an actual negative (y == 0) predicted otherwise
        # fn: an actual positive (y == 1) predicted otherwise
        fp, fn = pred[neg] != 0, pred[pos] != 1
        # grouped by the bins of the qualifying rows only, so .size() is the
        # rate's denominator: actual negatives per bin, actual positives per bin
        g_fp, g_fn = pd.Series(fp).groupby(bins[neg]), pd.Series(fn).groupby(bins[pos])
        fpr_bins, fpr_n = g_fp.mean(), g_fp.size()
        fnr_bins, fnr_n = g_fn.mean(), g_fn.size()
        out.update(
            **_worst(fpr_bins, fpr_n, "max_fpr"),
            **_worst(fnr_bins, fnr_n, "max_fnr"),
            overall_fpr=fp.mean(), overall_fnr=fn.mean(),
            per_bin_fpr=_per_bin(fpr_bins, n_slots), per_bin_fnr=_per_bin(fnr_bins, n_slots),
            per_bin_fpr_support=_per_bin_support(fpr_n, n_slots),
            per_bin_fnr_support=_per_bin_support(fnr_n, n_slots))
    return out
