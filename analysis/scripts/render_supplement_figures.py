"""Render supplementary figures S1–S6 from summary CSVs.

These are descriptive displays of the analysis results.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "inputs"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, GREEN, PURPLE, DARK = "#1769AA", "#C65D16", "#267A58", "#7254A3", "#24282D"
FIG_WIDTH = 180 / 25.4  # Wiley full-width upper bound: 180 mm.
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)


def read(name):
    return pd.read_csv(IN / name)


def save(fig, stem, sources, alt):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=450, facecolor="white")
    plt.close(fig)


def grid(ax, axis="x"):
    ax.grid(axis=axis, color="#E2E7EA", lw=0.5)


def s1():
    d = read("xshift_levine32_population.csv").sort_values("support", ascending=True)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 6.1), layout="constrained")
    y = np.arange(len(d))
    for metric, off, color, marker in [
        ("precision", -0.18, BLUE, "o"),
        ("recall", 0, ORANGE, "s"),
        ("f1", 0.18, GREEN, "^"),
    ]:
        ax.scatter(d[metric], y + off, s=21, color=color, marker=marker, label=metric.title())
    ax.set_yticks(y, [x.replace("_", " ") for x in d.population])
    ax.set(xlim=(-0.025, 1.025), xlabel="Per-reference-population metric")
    ax.legend(ncol=3, frameon=False, loc="lower right")
    grid(ax)
    ax.set_title(
        "Figure S1  Levine 32dim: all 14 reference populations", loc="left", fontweight="bold"
    )
    save(
        fig,
        "FigureS1_population_metrics",
        ["xshift_levine32_population.csv"],
        "Precision, recall and F1 for every manually referenced Levine 32dim population, sorted by population size. One fixed official X-shift K20 partition was evaluated post hoc.",
    )


def s2():
    d = read("flowsom_pareto.csv")
    fig, axs = plt.subplots(1, 2, figsize=(FIG_WIDTH, 3.5), layout="constrained")
    for ax, (ds, sub) in zip(axs, d.groupby("dataset", sort=False), strict=True):
        for grid_side, color in [(10, BLUE), (15, ORANGE), (20, GREEN)]:
            rows = sub[sub.grid_side.eq(grid_side)]
            for rlen, marker in [(10, "o"), (20, "s"), (30, "^")]:
                one = rows[rows.rlen.eq(rlen)]
                ax.scatter(
                    one.ari_mean,
                    one.macro_f1_mean,
                    color=color,
                    s=42,
                    marker=marker,
                    label=f"{grid_side}×{grid_side}" if rlen == 10 else None,
                )
        frontier = sub[sub.pareto_front_mean.astype(str).str.lower().eq("true")]
        ax.scatter(
            frontier.ari_mean,
            frontier.macro_f1_mean,
            s=85,
            facecolors="none",
            edgecolors=DARK,
            linewidth=1.1,
            label="3-objective Pareto",
        )
        ax.set(xlabel="Mean ARI", ylabel="Mean Macro F1", xlim=(0.45, 0.90), ylim=(0.58, 0.79))
        ax.set_title(ds.replace("_", " "), loc="left")
        grid(ax, "both")
    grid_handles = [
        Line2D([0], [0], color=c, marker="o", ls="none", label=f"{n}×{n}")
        for n, c in [(10, BLUE), (15, ORANGE), (20, GREEN)]
    ]
    rlen_handles = [
        Line2D([0], [0], color=DARK, marker=m, ls="none", label=f"rlen {r}")
        for r, m in [(10, "o"), (20, "s"), (30, "^")]
    ]
    pareto_handle = Line2D(
        [0],
        [0],
        color=DARK,
        marker="o",
        markerfacecolor="none",
        ls="none",
        label="3-objective Pareto",
    )
    axs[1].legend(
        handles=grid_handles + rlen_handles + [pareto_handle],
        frameon=False,
        loc="lower left",
        fontsize=5.8,
        ncol=2,
    )
    fig.suptitle(
        "Figure S2  FlowSOM grid × rlen: external metrics and Pareto status",
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    save(
        fig,
        "FigureS2_flowsom_pareto",
        ["flowsom_pareto.csv"],
        "Two datasets each have nine FlowSOM grid and rlen configurations. Points locate mean ARI and macro F1 within a common zoomed range, color encodes grid side, marker shape encodes rlen, and rings identify the three-objective Pareto set which also includes negative BMU distance.",
    )


def s3():
    x = read("xshift_k_sensitivity.csv")
    f = read("flowsom_auto_multiclass.csv")
    g = read("gmm_rare_regimes.csv")
    fig = plt.figure(figsize=(FIG_WIDTH, 7.2), layout="constrained")
    gs = fig.add_gridspec(3, 3)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(3)]
    datasets = ["Levine_13dim", "Levine_32dim", "Samusik_01", "Nilsson_rare", "Mosmann_rare"]
    for ax, ds in zip(axes[:5], datasets, strict=True):
        sub = x[x.dataset.eq(ds)].sort_values("knn_k")
        metric = "target_f1" if "rare" in ds else "macro_f1"
        ax.plot(sub.knn_k, sub[metric], "o-", color=BLUE, lw=1.4, ms=4)
        ax.set(
            xlabel="X-shift neighbor K",
            ylabel="Target F1" if "rare" in ds else "Macro F1",
            ylim=(0, 1),
        )
        ax.set_xticks(sub.knn_k)
        ax.set_title(ds.replace("_", " "), loc="left")
        grid(ax, "y")
    for ax, ds in zip(axes[5:7], ["Levine_32dim", "Samusik_01"], strict=True):
        sub = f[(f.dataset.eq(ds)) & f.metric.eq("macro_f1")]
        if sub.empty:
            raise ValueError((ds, "macro_f1"))
        regimes = ["official_auto_max40", "official_fixed_ktrue", "official_fixed_k40"]
        means = [float(sub[sub.regime.eq(r)]["mean"].iloc[0]) for r in regimes]
        ax.scatter(range(3), means, s=40, color=[BLUE, PURPLE, ORANGE], marker="D")
        ax.set_xticks(range(3), ["Auto", "K=true*", "K=40"])
        ax.set(ylabel="Macro F1", ylim=(0, 1))
        ax.set_title("FlowSOM | " + ds.replace("_", " "), loc="left")
        grid(ax, "y")
    for ax, ds in zip(axes[7:9], ["Nilsson_rare", "Mosmann_rare"], strict=True):
        sub = g[(g.dataset.eq(ds)) & g.metric.eq("target_f1")].sort_values("k")
        ax.errorbar(
            sub.k,
            sub["mean"],
            yerr=[sub["mean"] - sub["min"], sub["max"] - sub["mean"]],
            fmt="o-",
            capsize=2,
            color=GREEN,
            lw=1.2,
            ms=4,
        )
        ax.set_xticks(sub.k)
        ax.set(xlabel="GMM components K", ylabel="Target F1", ylim=(0, 1))
        ax.set_title("GMM | " + ds.replace("_", " "), loc="left")
        grid(ax, "y")
    fig.suptitle(
        "Figure S3  Specified parameter regimes (different K semantics)",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    save(
        fig,
        "FigureS3_parameter_regimes",
        ["xshift_k_sensitivity.csv", "flowsom_auto_multiclass.csv", "gmm_rare_regimes.csv"],
        "Full X-shift fixed-neighbor K scan on five datasets, FlowSOM automatic versus fixed metaclustering on two multiclass datasets, and GMM component-count sensitivity on two rare datasets. K has different meanings across methods and the panels do not support a common ranking.",
    )


def s4():
    d = read("stability_40cells.csv")
    t = read("stability_40tests.csv")
    p = read("multimodality_power.csv")
    fig = plt.figure(figsize=(FIG_WIDTH, 7.1), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])
    ax = fig.add_subplot(gs[0, :])
    axb = fig.add_subplot(gs[1, 0])
    axc = fig.add_subplot(gs[1, 1])
    d = d.sort_values(["algorithm", "dataset"])
    y = np.arange(len(d))[::-1]
    for i, (_, r) in enumerate(d.iterrows()):
        ax.plot([r.ari_min, r.ari_max], [y[i], y[i]], color=BLUE, lw=1.7)
        ax.scatter(r.ari_mean, y[i], color=BLUE, marker="D", s=20)
    method_names = {
        "Leiden_fixed_PhenoGraph_graph": "PhenoGraph–Leiden",
        "PhenoGraph_default_Louvain": "PhenoGraph–Louvain",
    }
    ax.set_yticks(
        y,
        [
            f"{method_names.get(str(r.algorithm), str(r.algorithm).replace('_', ' '))} | {str(r.dataset).replace('_', ' ')}"
            for _, r in d.iterrows()
        ],
    )
    ax.set(xlabel="Run ARI: mean diamond, min–max line", xlim=(-0.04, 1.04))
    ax.set_title("A  Earlier stability family: 10 fixed regimes", loc="left", fontweight="bold")
    grid(ax)
    rejected = int(t.reject_single_gaussian_global_0_05.astype(str).str.lower().eq("true").sum())
    axb.barh([1, 0], [rejected, len(t) - rejected], color=[ORANGE, BLUE], height=0.6)
    axb.set_yticks([1, 0], ["Rejected", "No rejection"])
    axb.set(xlabel="Tests in retrospective family", xlim=(0, len(t) + 2))
    for yy, n in [(1, rejected), (0, len(t) - rejected)]:
        axb.text(n + 0.5, yy, str(n), va="center", fontsize=8)
    axb.set_title("B  Global Holm: 40 endpoints", loc="left", fontweight="bold")
    grid(axb, "x")
    axc.plot(
        p.mean_separation_sd_units,
        p.independent_power,
        "o-",
        color=GREEN,
        label="Independent implementation",
    )
    axc.plot(
        p.mean_separation_sd_units,
        p.original_power,
        "s--",
        color=PURPLE,
        label="Original implementation",
    )
    axc.set(
        xlabel="Mixture mean separation (SD units)",
        ylabel="Estimated rejection probability",
        ylim=(0, 1),
    )
    axc.set_title("C  n=30 power simulation", loc="left", fontweight="bold")
    axc.legend(frameon=False, fontsize=6)
    grid(axc, "y")
    fig.suptitle(
        "Figure S4  Stability diagnostics and test power", x=0.01, ha="left", fontweight="bold"
    )
    save(
        fig,
        "FigureS4_stability_diagnostics",
        ["stability_40cells.csv", "stability_40tests.csv", "multimodality_power.csv"],
        "The earlier family of ten fixed-regime method-dataset conditions shows mean and range of ARI over algorithm runs. A count panel reports two rejections and 38 non-rejections after global Holm adjustment across 40 endpoints; exact adjusted p values are in Table S5. An n=30 simulation shows low rejection probability except at large mixture separation. These tests are not transferred to newer full-event runs.",
    )


def s5():
    d = read("label_perturbation.csv")
    d = d[d.kind.ne("unchanged")]
    fig, axs = plt.subplots(2, 2, figsize=(FIG_WIDTH, 5.6), layout="constrained")
    for row, ds in enumerate(["Levine_32dim", "Samusik_01"]):
        for col, (metric, title) in enumerate(
            [("delta_ari_from_baseline", "Δ ARI"), ("delta_macro_f1_from_baseline", "Δ Macro F1")]
        ):
            ax = axs[row, col]
            for method, color, marker in [("X-shift", BLUE, "o"), ("FlowSOM", ORANGE, "s")]:
                for kind, ls in [("boundary_flip", "-"), ("boundary_exclude", "--")]:
                    sub = d[
                        d.dataset.eq(ds) & d.algorithm.eq(method) & d.kind.eq(kind)
                    ].sort_values("nominal_fraction")
                    ax.plot(
                        sub.nominal_fraction * 100,
                        sub[metric],
                        color=color,
                        marker=marker,
                        ls=ls,
                        lw=1.2,
                        ms=4,
                    )
            ax.axhline(0, color="#7D858A", lw=0.7)
            ax.set(xlabel="Nominal boundary fraction (%)", ylabel=title, xticks=[1, 5, 10])
            ax.set_title(ds.replace("_", " ") + " | " + title, loc="left")
            grid(ax, "y")
    handles = [
        Line2D([0], [0], color=BLUE, marker="o", label="X-shift"),
        Line2D([0], [0], color=ORANGE, marker="s", label="FlowSOM"),
        Line2D([0], [0], color=DARK, ls="-", label="Flip"),
        Line2D([0], [0], color=DARK, ls="--", label="Exclude"),
    ]
    axs[0, 0].legend(handles=handles, frameon=False, ncol=2, fontsize=6.3)
    fig.suptitle(
        "Figure S5  Simulated boundary-label perturbations", x=0.01, ha="left", fontweight="bold"
    )
    save(
        fig,
        "FigureS5_label_perturbation",
        ["label_perturbation.csv"],
        "For two datasets and two fixed method outputs, simulated flipping or exclusion of one, five or ten percent of boundary labels changes post hoc ARI and macro F1. These are hypothetical sensitivity scenarios, not observed disagreement between human gaters.",
    )


def s6():
    d = read("marker_dimension.csv")
    fig, axs = plt.subplots(1, 2, figsize=(FIG_WIDTH, 3.3), layout="constrained")
    for ax, mean, sd, color, title in [
        (axs[0], "ari_mean_across_chain_mean", "ari_mean_across_chain_sd", BLUE, "ARI"),
        (
            axs[1],
            "macro_f1_mean_across_chain_mean",
            "macro_f1_mean_across_chain_sd",
            ORANGE,
            "Macro F1",
        ),
    ]:
        multi = d[d.n_marker_chains_contributing.gt(1)]
        ax.errorbar(
            multi.n_markers,
            multi[mean],
            yerr=multi[sd],
            fmt="o-",
            color=color,
            capsize=3,
            lw=1.3,
            ms=4,
        )
        single = d[d.n_marker_chains_contributing.eq(1)]
        ax.scatter(
            single.n_markers,
            single[mean],
            color=DARK,
            marker="D",
            s=34,
            label="Shared full-32 reference",
        )
        ax.set(
            xlabel="Markers retained in nested chain", ylabel=title, ylim=(0, 1), xticks=d.n_markers
        )
        grid(ax, "y")
        ax.set_title(title, loc="left", fontweight="bold")
    axs[1].legend(frameon=False, loc="lower right", fontsize=6.2)
    fig.suptitle(
        "Figure S6  Levine 32dim nested marker sensitivity", x=0.01, ha="left", fontweight="bold"
    )
    save(
        fig,
        "FigureS6_marker_dimension",
        ["marker_dimension.csv"],
        "For one dataset, ARI and macro F1 change along nested marker subsets. Means and SD across five marker chains are shown for 8 through 26 markers. At 32 markers there is only one shared full-panel reference, displayed as a diamond without an SD.",
    )


def main():
    for function in (s1, s2, s3, s4, s5, s6):
        function()
    print("Rendered Figures S1-S6 to", OUT)


if __name__ == "__main__":
    main()
