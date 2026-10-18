"""

Built with the assistance of Claude for the sake of better formatting for the paper 
results. 

The A1 figure: per-bin test misclassification under each binning.

binning_panels and the style are copied from New_Experiments/motivating/figures.py;
the per-bin tables (bin_rows, bin_table) follow motivating/analyze.py, keeping only
the misclassification columns A1 uses. Rates are means over the eval seeds,
whiskers +-1 sd across seeds.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
BAR, HOT, BLUE, ORANGE = "#86b6ef", "#eb6834", "#2a78d6", "#eb6834"
NAMES = {"equal_width": "Equal width", "equal_freq": "Equal frequency", "kmeans": "K-means", "jenks": "Jenks",
         "entropy": "Entropy (MDLP)", "fairbin_error": "Fair Bin (paper)", "fairbin_fpr": "Fair Bin FPR (paper)",
         "fairbin_zbonf": "Fair Bin (robust)", "fairbin_zbonf_fpr": "Fair Bin FPR (robust)",
         "fairbin_zbonf_cap": "Fair Bin (robust, cap)", "fairbin_zbonf_band": "Fair Bin (robust, band)"}
NAMES["fairbin"] = "Fair Bin"
PANELS = ["equal_width", "kmeans", "jenks", "entropy", "equal_freq", "fairbin"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 9.5,
    "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.dpi": 220, "legend.frameon": False,
})


def pct(v, d=1):
    return f"{100 * v:.{d}f}%"


def header(fig, fid, title, sub):
    fig.text(0.01, 0.985, fid, fontsize=11, fontweight="bold", color=HOT, va="top")
    fig.text(0.045, 0.985, title, fontsize=11, fontweight="bold", color=INK, va="top")
    fig.text(0.045, 0.945 if fig.get_figheight() > 4 else 0.915, sub, fontsize=8.5, color=INK2, va="top")


def save(fig, fname, out=None):
    out = out or FIG
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / fname, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("wrote", fname)


def range_label(v):
    return "empty" if len(v) == 0 else (f"{v.min():g}" if v.min() == v.max() else f"{v.min():g}–{v.max():g}")


def bin_rows(strategy, seed, x, edges, bidx, ev):
    """
    One row per bin for one (binning, seed), from metrics.evaluate's result ev:
    the bin's test misclassification and test rows, and the raw-value range over
    all rows in it (as analyze.analyze). Plus one overall row (bin -1) carrying
    the run's error and MRE.
    """
    rows = []
    for i in range(len(edges) - 1):
        err = ev["per_bin_error"][i]
        rows.append({"strategy": strategy, "seed": seed, "bin": i, "label": range_label(x[bidx == i]),
                     "n_test": ev["per_bin_rows"][i], "err": np.nan if err is None else err})
    rows.append({"strategy": strategy, "seed": seed, "bin": -1, "label": "",
                 "n_test": sum(ev["per_bin_rows"]), "err": 1 - ev["accuracy"], "mre": ev["mre"]})
    return rows


def bin_table(rows, name):
    """
    (B, G) for binning_panels, as analyze.analyze aggregates them: mean and sd
    over seeds per bin; each bin labelled by its most common range, n_labels > 1
    when the range shifts across seeds (drawn with a leading ≈); G holds the
    overall rate (group -1).
    """
    df = pd.DataFrame(rows)
    per_bin, overall = df[df.bin >= 0], df[df.bin == -1]
    stats = per_bin.groupby(["strategy", "bin"])[["err", "n_test"]].agg(["mean", "std"])
    stats.columns = [a if b == "mean" else f"{a}_{b}" for a, b in stats.columns]
    labels = per_bin.groupby(["strategy", "bin"]).label.agg(lambda s: s.mode().iloc[0])
    n_labels = per_bin.groupby(["strategy", "bin"]).label.nunique().rename("n_labels")
    B = stats.join(labels).join(n_labels).reset_index()
    G = overall.groupby("strategy")[["err", "n_test"]].mean().reset_index().assign(group=-1)
    # MRE as the experiments report it: metrics.evaluate's mre (each seed's worst bin), averaged over seeds
    B = B.join(overall.groupby("strategy").mre.mean().rename("seed_mre"), on="strategy")
    for t in (B, G):
        t.insert(0, "dataset", name)
    return B, G


def binning_panels(B, G, name, metric, fid, title, sub, strategies=PANELS, count="n_test", min_n=30,
                   rate_label="misclassification", xlabel="Age bin", grid=(2, 3), bare=False, names=None, ytop=None, solid=False, scale=1.0):
    """Small multiples: one binning per panel, a bar per bin, worst reliable bin highlighted."""
    fig, axes = plt.subplots(*grid, figsize=(3.5 * grid[1], 2.8 * grid[0]), sharey=True, squeeze=False)
    ymax, panel = 0, []
    for ax, s in zip(axes.flat, strategies):
        b = B[(B.dataset == name) & (B.strategy == s)].sort_values("bin")
        allrow = G[(G.dataset == name) & (G.strategy == s) & (G.group == -1)].iloc[0]
        labels = [("≈" if nl > 1 else "") + l for l, nl in zip(b.label, b.n_labels)]
        vals, sds, ns = b[metric].to_numpy(), b[f"{metric}_std"].to_numpy(), b[count].to_numpy()
        keep = ~np.isnan(vals)
        xs = np.arange(keep.sum())
        vals, sds, ns, labels = vals[keep], sds[keep], ns[keep], [l for l, k in zip(labels, keep) if k]
        few = (ns < min_n) & (not solid)  # solid: draw small bins like any other
        worst = int(np.nanargmax(vals))
        colors = [HOT if i == worst else BAR for i in range(len(vals))]
        bars = ax.bar(xs, vals, width=0.72, color=colors, zorder=2)
        for i, bar in enumerate(bars):
            if few[i]:
                bar.set_alpha(0.45)
                bar.set_hatch("////")
                bar.set_edgecolor(SURF)
        ax.errorbar(xs[~few], vals[~few], yerr=sds[~few], fmt="none", ecolor=INK2, elinewidth=0.7, capsize=0, zorder=3)
        ax.axhline(allrow[metric], color=INK, lw=1, zorder=4)
        right = worst < len(xs) / 2  # keep the overall label away from the worst-bin label
        ax.text(len(xs) - 0.55 if right else -0.45, allrow[metric], f"overall {pct(allrow[metric])}",
                ha="right" if right else "left", va="bottom", fontsize=7.5 * scale, color=INK, zorder=5,
                bbox=dict(boxstyle="square,pad=0.15", fc=SURF, ec="none", alpha=0.85))
        panel.append((ax, bars, vals, sds, few, worst, ns))
        ax.set_xticks(xs, labels, rotation=55, ha="right", fontsize=6.8 * scale)
        ax.set_title(f"{(names or {}).get(s, NAMES[s])}   ·   accuracy {pct(1 - allrow.err)}", loc="left", fontsize=9.5 * scale, pad=4 + 10 * (scale - 1))
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", labelsize=9 * scale)
        ymax = max(ymax, np.nanmax(np.where(few, vals, vals + sds)))
    top = ytop or min(1.0, ymax * 1.25)
    cut = top * 0.85 if ytop else top  # with a fixed scale, overshooting bars stop here and carry their value
    for ax, bars, vals, sds, few, worst, ns in panel:
        # tiny bins can overshoot the shared scale: clip, and say so on the bar
        for i, v in enumerate(vals):
            if v > top and ytop:
                bars[i].set_height(cut)
                if i != worst:
                    ax.text(i, cut, f"{pct(v, 0)}↑", ha="center", va="bottom", fontsize=7 * scale, color=INK2)
            elif v > top:
                ax.text(i, top * 0.97, f"{pct(v, 0)}↑", ha="center", va="top", fontsize=7 * scale, color=INK2)
        over = ytop and vals[worst] > top
        v = cut if over else min(vals[worst] + (0 if few[worst] else sds[worst]), top if ytop else top * 0.8)
        ax.annotate(f"{pct(vals[worst])}\nn≈{ns[worst]:.0f}", (worst, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5 * scale, color=INK, fontweight="bold")
    used = axes.flat[:len(strategies)]
    for ax in axes.flat[len(strategies):]:  # odd panel count: leave the spare cell blank
        ax.set_visible(False)
    for ax in used:
        ax.set_ylim(0, top)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    for ax in axes[:, 0]:
        ax.set_ylabel(rate_label, fontsize=9 * scale)
    for ax in used[-grid[1]:]:
        ax.set_xlabel(xlabel, fontsize=8 * scale)
    if bare:  # no header or footnote, for placing in the paper with its own caption
        fig.tight_layout()
        return fig
    header(fig, fid, title, sub)
    fig.text(0.045, -0.02, f"Orange: worst bin (the value MRE / max rate reports). Hatched: fewer than {min_n} "
             f"{'test rows' if count == 'n_test' else 'qualifying test rows'} per seed. "
             "Black line: rate over all test rows. Whiskers: ±1 sd over 9 seeds. rf, k = 10.\n"
             "Fair Bin (robust): out-of-fold RF counts + Wilson bound, bins chosen per seed on its training rows; "
             "≈ marks a bin whose range shifts across seeds (most common range shown). Baselines: bins from all rows.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig



# Ablation bar charts: plot_ablation.py's draw, copied (names prefixed ABL_ so they do not clash with A1's).
# d: one row per (dataset, arm) with mean mre and median tally_time_s / dp_time_s; MRE is averaged over
# datasets, time is the median over datasets (log axis).
ABL_ORDER = ["original", "minsize_5pct", "z0", "pregroup_G100", "proxy_rf", "proxy_lr"]
ABL_NAMES = {"original": "Fair Bin", "minsize_5pct": "min size 5%", "z0": "z = 0", "pregroup_G100": "G = 100",
         "proxy_rf": r"$\hat{\mathcal{M}}$ = RF", "proxy_lr": r"$\hat{\mathcal{M}}$ = LR"}
ABL_GREY, ABL_BLUE, ABL_ORANGE, ABL_INK, ABL_BG = "#888780", "#2a78d6", "#ed6a33", "#444441", "#fcfcfa"
TICK_FS, LAB_FS, VAL_FS = 15, 16, 15


def draw_ablation(d, out, kind):
    x = np.arange(len(ABL_ORDER))
    fig, ax = plt.subplots(figsize=(6.47, 4.42), dpi=100, facecolor=ABL_BG)
    ax.set_facecolor("white")
    cols = [ABL_GREY if l == "original" else (ABL_BLUE if kind == "mre" else ABL_ORANGE) for l in ABL_ORDER]
    if kind == "mre":
        val = np.array([d[d.label == l].mre.mean() for l in ABL_ORDER])
        ax.bar(x, val, 0.65, color=cols)
        for xi, m in zip(x, val):
            ax.text(xi, 0.008, f"{m:.3f}", rotation=90, ha="center", va="bottom", color="white", fontsize=VAL_FS,
                    fontweight="bold")
        ax.set_ylabel("MRE", fontsize=LAB_FS, color=ABL_INK)
    else:
        t = d.tally_time_s + d.dp_time_s
        val = np.array([np.median(t[d.label == l]) for l in ABL_ORDER])
        ax.set_yscale("log")
        ax.bar(x, val, 0.65, color=cols)
        ylo = val.min() * 0.12
        ax.set_ylim(ylo, val.max() * 1.6)
        for xi, m in zip(x, val):
            ax.text(xi, ylo * 1.15, f"{m:.2f}s", rotation=90, ha="center", va="bottom", color="white", fontsize=VAL_FS, fontweight="bold")
        ax.set_ylabel("time (s, log)", fontsize=LAB_FS, color=ABL_INK)
    ax.set_xticks(x)
    ax.set_xticklabels([ABL_NAMES[l] for l in ABL_ORDER], rotation=40, ha="right", fontsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS, colors=ABL_INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color="#d9d9d6", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", out)


from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402


# Z1 (z sweep): zsweep_figures.py's z1, z1_panel and style, copied; z1 takes summary s (one row per dataset, k,
# z_label with mre, regret_pct, smallest_bin_share) and the output folder.
K_STYLE = {5: ("#2a78d6", "o"), 10: ("#eb6834", "s"), 14: ("#1baf7a", "^")}  # categorical slots 1-3


def z_style():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial"],
                         "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.spines.top": False,
                         "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.labelcolor": INK2, "savefig.dpi": 220, "legend.frameon": False})


def z_save(fig, name, out):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", **({} if fig.get_figwidth() < 5 else {"bbox_inches": "tight", "pad_inches": 0.15}))
    plt.close(fig)


def z1_panel(ax, num, bonf, col, scale, ylabel, title, with_band_label=False, big=False):
    ax.axvspan(bonf.z.min(), bonf.z.max(), color=GRID, alpha=0.7, lw=0, zorder=0)
    for k, (color, marker) in K_STYLE.items():
        g = num[num.k == k].groupby("z")[col]
        med = g.median() * scale
        if k == 10:
            ax.fill_between(med.index, g.quantile(0.25) * scale, g.quantile(0.75) * scale,
                            color=color, alpha=0.13, lw=0, zorder=1)
        ax.plot(med.index, med, color=color, lw=1.6 if big else 2, marker=marker, ms=3.5 if big else 6, label=f"k = {k}", zorder=3)
        b = bonf[bonf.k == k]
        ax.scatter(b.z.median(), b[col].median() * scale, marker="D", s=22 if big else 60, color=color,
                   edgecolor=SURF, linewidth=1 if big else 1.5, zorder=4)
    ax.set_xlabel("z (Wilson bound)")
    ax.set_ylabel(ylabel)
    if big:
        ax.set_xlabel("Z (Wilson Bound)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.tick_params(labelsize=8)
    else:
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlim(-0.4, 12.4)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    if big:  # whole-percent ticks only, so labels are never rounded (2.5 -> "2%")
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True, steps=[2, 5, 10]))
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.text((bonf.z.min() + bonf.z.max()) / 2, ax.get_ylim()[1] * 0.98, "Bonferroni z\n(per dataset)",
            ha="center", va="top", fontsize=7 if big else 7.5, color=MUTED)
    if with_band_label:
        ax.text(0.98, 0.98 if big else 0.58, "band: k = 10 IQR\nover 15 datasets" if big else "band: k = 10 interquartile range\nover the 15 datasets",
                 transform=ax.transAxes, ha="right", va="top", fontsize=7 if big else 7.5, color=MUTED)
    handles = ax.get_legend_handles_labels()[0] + [
        Line2D([], [], ls="", marker="D", ms=7, color=MUTED, markeredgecolor=SURF, label="Bonferroni z" if big else "Bonferroni z (median)")]
    if big and col == "smallest_bin_share":
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.01, 0.84), fontsize=7, handlelength=1.4,
                  labelspacing=0.3)
    else:
        ax.legend(handles=handles, loc="upper right", fontsize=7.5 if big else 8,
                  bbox_to_anchor=(1.0, 0.84) if big else None)


def z1(s, out):
    num = s[s.z_label != "bonf"]
    bonf = s[s.z_label == "bonf"]

    fig, ax = plt.subplots(figsize=(4.2, 2.25))
    z1_panel(ax, num, bonf, "regret_pct", 1, "MRE Above Best Z (Median)", "Worst-bin error vs z",
             with_band_label=True, big=True)
    fig.tight_layout()
    z_save(fig, "Z1_regret_vs_z", out)

    fig, ax = plt.subplots(figsize=(4.2, 2.25))
    z1_panel(ax, num, bonf, "smallest_bin_share", 100, "Smallest Bin (% of Rows)", "Why: z stops tiny bins",
             big=True)
    fig.tight_layout()
    z_save(fig, "Z1_smallest_bin_vs_z", out)


# Folds: paper_figures.py's folds, copied; df = runs (dataset, eval_seed, folds, mre, tally_time_s).
PF_INK, PF_MUTED, PF_GRID, PF_SURFACE, PF_FAINT = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#c3c2b7"
PF_BLUE, PF_ORANGE, PF_AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3


def pf_style():
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": PF_MUTED, "axes.labelcolor": PF_INK, "xtick.color": PF_MUTED,
                         "ytick.color": PF_MUTED, "axes.titlesize": 10, "axes.titlelocation": "left",
                         "figure.facecolor": PF_SURFACE, "axes.facecolor": PF_SURFACE, "savefig.facecolor": PF_SURFACE})


def tidy(ax):
    ax.grid(True, axis="y", color=PF_GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def seed_band(piv):
    """piv: index (dataset, seed), columns x -> mean over datasets per seed; returns mean and 95% CI half-width."""
    per_seed = piv.groupby(level="eval_seed").mean()
    m = per_seed.mean()
    ci = 1.96 * per_seed.std(ddof=1) / np.sqrt(len(per_seed))
    return m, ci


def folds(df, out):
    df = df[df.status == "ok"]
    piv = 100 * df.pivot_table(index=["dataset", "eval_seed"], columns="folds", values="mre")
    x = list(piv.columns)
    names = ["1\n(holdout)" if v == 1 else str(v) for v in x]

    fig, a = plt.subplots(figsize=(4.2, 2.5))
    m, ci = seed_band(piv)
    a.fill_between(x, m - ci, m + ci, color=PF_BLUE, alpha=0.15, lw=0)
    a.plot(x, m, color=PF_BLUE, lw=2.5, marker="o", ms=4.5, mec=PF_SURFACE, label="Mean of 15 datasets (95% CI)")
    for v in x:
        a.text(v, m[v] + ci[v] + 0.15, f"{m[v]:.1f}", ha="center", va="bottom", fontsize=7, color=PF_INK)
    a.axvline(5, ymin=0.12, color=PF_MUTED, lw=0.8, ls="--")
    a.text(5.1, m.max() + 2.4, "default (5)", color=PF_MUTED, fontsize=7)
    a.set_ylim(m.min() - 3, m.max() + 3)
    a.set_xticks(x, names)
    a.set_xlabel("Folds in the Stump Tally (1 = Single 50/50 Holdout)", fontsize=8.5)
    a.tick_params(labelsize=8)
    a.set_ylabel("MRE (%)", fontsize=9)
    a.legend(frameon=False, fontsize=7, loc="lower right")
    tidy(a)
    fig.tight_layout()
    path = out / "folds_mre.png"
    fig.savefig(path, dpi=200)
    print(path)

    t = df.groupby(["dataset", "folds"]).tally_time_s.median().unstack()
    fig, b = plt.subplots(figsize=(4.2, 2.5))
    b.plot(x, t.sum()[x], color=PF_BLUE, lw=2, marker="o", ms=4, label="all 15 datasets (sum)")
    b.plot(x, t.loc["uci117_kdd", x], color=PF_AQUA, lw=2, marker="s", ms=4, label="KDD (209k rows)")
    b.set_xticks(x, names)
    b.set_xlabel("Folds in the Stump Tally (1 = Single 50/50 Holdout)", fontsize=8.5)
    b.tick_params(labelsize=8)
    b.set_ylabel("Tally Time (s)", fontsize=9)
    b.set_ylim(0, None)
    b.legend(frameon=False, fontsize=7, loc="upper left")
    tidy(b)
    fig.tight_layout()
    path = out / "folds_time.png"
    fig.savefig(path, dpi=200)
    print(path)



# k sweep: paper_figures.py's ksweep figures, copied; d = runs (dataset, method, k, eval_model, eval_seed, mre),
# methods as K_NAMES, eval models as EVAL_LABELS.
K_NAMES = {"fairbin_stump_nobound": "Fair Bin", "eqfreq_k": "Equal freq. (b bins)", "equal_width": "Equal width",
           "kmeans": "k-means", "jenks": "Jenks", "entropy": "Entropy"}
K_COLORS = dict(zip(K_NAMES, [PF_BLUE, PF_ORANGE, PF_AQUA, "#eda100", "#e87ba4", "#008300"]))  # categorical slots 1-6
K_PANELS = (5, 7, 10, 14)
EVAL_LABELS = {"rf": "Random Forest", "xgb": "XGBoost", "nn_early": "Neural Net (MLP)"}
EVAL_COLORS = {"rf": PF_BLUE, "xgb": PF_ORANGE, "nn_early": PF_AQUA}


def ksweep_avg(d):
    """(eval_model, method, k) -> mean over seeds of the per-seed mean over datasets, with its 95% CI."""
    per_seed = 100 * d.groupby(["eval_model", "method", "k", "eval_seed"]).mre.mean()
    avg = per_seed.groupby(level=["eval_model", "method", "k"]).agg(["mean", "std", "size"])
    avg["ci"] = 1.96 * avg["std"] / np.sqrt(avg["size"])
    order = avg.xs(("rf", 10), level=("eval_model", "k"))["mean"].sort_values().index.tolist()
    return avg, order


def ksweep_lines(avg, order, models=None, show_title=True, fontsize_scale=1.0):
    models = [m for m in EVAL_LABELS if m in avg.index.unique("eval_model") and (models is None or m in models)]
    fs = fontsize_scale
    fig, axes = plt.subplots(1, len(models), figsize=(max(5 * len(models), 7), 4.2), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, model in zip(axes, models):
        for m in order:
            a = avg.loc[(model, m)]
            ax.fill_between(a.index, a["mean"] - a["ci"], a["mean"] + a["ci"], color=K_COLORS[m], alpha=0.15, lw=0)
            ax.plot(a.index, a["mean"], color=K_COLORS[m], lw=2, marker="o", ms=4, label=K_NAMES[m])
        ax.set_xticks(range(5, 15))
        ax.tick_params(labelsize=9 * fs)
        ax.set_xlabel("number of bins b", fontsize=9 * fs)
        if show_title:
            ax.text(0.0, 1.02, EVAL_LABELS[model], transform=ax.transAxes, color=PF_INK, fontsize=10 * fs)
        tidy(ax)
    axes[0].set_ylabel("MRE (%), mean of 15 datasets\n(95% CI over seeds)", fontsize=9 * fs)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8 * fs, loc="lower center", ncol=len(order) if len(models) > 1 else 3,
               bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=(0, 0.07 if len(models) > 1 else 0.13, 1, 1))
    return fig


def ksweep_dots(avg, order):
    fig, axes = plt.subplots(1, len(K_PANELS), figsize=(12, 3.6), sharex=True, sharey=True)
    models = [m for m in EVAL_LABELS if m in avg.index.unique("eval_model")]
    for ax, k in zip(axes, K_PANELS):
        for off, model in zip(np.linspace(-0.22, 0.22, len(models)), models):
            a = avg.xs((model, k), level=("eval_model", "k")).reindex(order)
            ax.errorbar(a["mean"], np.arange(len(order)) + off, xerr=a["ci"], fmt="o", ms=5, color=EVAL_COLORS[model],
                        elinewidth=1.2, capsize=0, label=EVAL_LABELS[model])
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([K_NAMES[m] for m in order])
        ax.text(0.0, 1.02, f"b = {k}", transform=ax.transAxes, color=PF_INK, fontsize=10)
        ax.grid(True, axis="x", color=PF_GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].invert_yaxis()
    fig.supxlabel("MRE (%), mean of 15 datasets (95% CI over seeds)", fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="upper right", ncol=len(h))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def ksweep_bars(avg, order):
    fig, axes = plt.subplots(1, len(K_PANELS), figsize=(13, 3.8), sharey=True)
    models = [m for m in EVAL_LABELS if m in avg.index.unique("eval_model")]
    width = 0.8 / len(models)
    for ax, k in zip(axes, K_PANELS):
        for i, model in enumerate(models):
            a = avg.xs((model, k), level=("eval_model", "k")).reindex(order)
            ax.bar(np.arange(len(order)) + (i - (len(models) - 1) / 2) * width, a["mean"], width=width,
                   color=EVAL_COLORS[model],
                   edgecolor=PF_SURFACE, linewidth=1.2, label=EVAL_LABELS[model], yerr=a["ci"],
                   error_kw={"ecolor": PF_INK, "elinewidth": 1, "capsize": 0})
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([K_NAMES[m] for m in order], rotation=35, ha="right")
        ax.text(0.0, 1.02, f"b = {k}", transform=ax.transAxes, color=PF_INK, fontsize=10)
        tidy(ax)
    axes[0].set_ylabel("MRE (%), mean of 15 datasets\n(95% CI over seeds)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="upper right", ncol=len(h))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def ksweep(d, out):
    avg, order = ksweep_avg(d)
    for name, fn in [("lines", ksweep_lines),
                     ("lines_nn", lambda a, o: ksweep_lines(a, o, ["nn_early"], show_title=False, fontsize_scale=1.4)),
                     ("lines_rf", lambda a, o: ksweep_lines(a, o, ["rf"], show_title=False, fontsize_scale=1.4)),
                     ("lines_xgb", lambda a, o: ksweep_lines(a, o, ["xgb"], show_title=False, fontsize_scale=1.4)),
                     ("dot", ksweep_dots), ("bars", ksweep_bars)]:
        fig = fn(avg, order)
        path = out / f"ksweep_mre_{name}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(path)
    print((avg["mean"].unstack("k")).round(1).to_string())
