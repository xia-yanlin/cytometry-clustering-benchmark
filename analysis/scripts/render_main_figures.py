"""Render the three manuscript figures from panel CSVs.

Run with the project's FlowSOM Python environment containing Matplotlib:
  python scripts/render_main_figures.py
The figures describe fixed experimental regimes; they do not rank algorithms.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "figure_data"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

BLUE = "#1769AA"
ORANGE = "#C65D16"
GREEN = "#267A58"
PURPLE = "#7254A3"
DARK = "#24282D"
GRAY = "#747B82"
PALETTE = [BLUE, ORANGE, GREEN, PURPLE]
FIG_WIDTH = 180 / 25.4  # Wiley full-width upper bound: 180 mm.

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.65,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
})


def read(name):
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(f"Build the panel data first: {path}")
    return pd.read_csv(path)


def save(fig, stem, sources, alt):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=450, facecolor="white")
    plt.close(fig)


def label(ax, letter, title):
    ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold", pad=8)


def rare_panel(ax, data, dataset, letter):
    sub = data[data.dataset.eq(dataset)]
    metrics = ["ari", "precision", "recall", "f1"]
    labels = ["ARI", "Precision", "Recall", "F1"]
    x = np.arange(4)
    groups = [("GMM", "K=2", BLUE, "o"), ("GMM", "K=10", ORANGE, "s"),
              ("GMM", "K=40", GREEN, "^"), ("official_Xshift", "neighbor_K=20", DARK, "D")]
    for j, (method, regime, color, marker) in enumerate(groups):
        rows = sub[sub.method.eq(method) & sub.regime.eq(regime)]
        if rows.empty:
            raise ValueError((dataset, method, regime))
        xpos = x + (j - 1.5) * 0.17
        for i, metric in enumerate(metrics):
            values = rows[metric].to_numpy(float)
            if len(values) > 1:
                ax.vlines(xpos[i], values.min(), values.max(), color=color, alpha=.7, lw=1.3)
            ax.scatter(xpos[i], values.mean(), s=24 if j < 3 else 37, marker=marker,
                       color=color, edgecolor="white", linewidth=.35, zorder=4)
    ax.set_xticks(x, labels)
    ax.set_ylim(-.04, 1.05)
    ax.set_xlim(-.42, 3.42)
    ax.set_ylabel("Metric value")
    ax.grid(axis="y", color="#E1E5E8", linewidth=.5)
    label(ax, letter, dataset.replace("_", " "))


def figure1():
    a = read("figure1a_reference_simulation.csv")
    b = read("figure1b_rare_run_points.csv")
    c = read("figure1c_multiclass_population.csv")
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, 6.7), layout="constrained")
    ax = axes[0, 0]
    x = a.background_clusters.to_numpy(int)
    ax.plot(x, a.ari, "o-", color=BLUE, lw=1.5, ms=4, label="Raw partition ARI")
    ax.plot(x, a.target_f1, "s--", color=ORANGE, lw=1.4, ms=4, label="Target F1")
    ax.set_xscale("log", base=2)
    ax.set_xticks(x, [str(v) for v in x])
    ax.set(xlabel="Background clusters in prespecified simulation", ylabel="Metric value", ylim=(-.04, 1.05))
    ax.grid(axis="y", color="#E1E5E8", linewidth=.5)
    ax.legend(loc="center right", frameon=False)
    label(ax, "A", "Background splitting lowers ARI")

    rare_panel(axes[0, 1], b, "Nilsson_rare", "B")
    rare_panel(axes[1, 0], b, "Mosmann_rare", "C")
    handles = [Line2D([0], [0], color=col, marker=mark, linestyle="none", markersize=5, label=txt)
               for txt, col, mark in [("GMM K=2", BLUE, "o"), ("GMM K=10", ORANGE, "s"),
                                      ("GMM K=40", GREEN, "^"), ("X-shift K=20", DARK, "D")]]
    fig.legend(handles=handles, ncol=4, loc="outside lower center", frameon=False,
               fontsize=7, title="B–C: method and parameter regime")

    ax = axes[1, 1]
    ax.scatter(c.recall, c.precision, s=21, color=PURPLE, alpha=.85)
    for _, row in c.iterrows():
        if row.f1 < .3 or row.recall < .25:
            short = str(row.population).replace("_", " ")
            if len(short) > 17:
                short = short[:15] + "…"
            ax.annotate(short, (row.recall, row.precision), xytext=(3, 2),
                        textcoords="offset points", fontsize=6.5, color=DARK)
    ax.plot([0, 1], [0, 1], ls=":", color="#B8BFC4", lw=.7)
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(-.04, 1.04), ylim=(-.04, 1.04))
    ax.grid(color="#E8ECEF", linewidth=.45)
    label(ax, "D", "Levine 32dim: 14 reference populations")
    save(fig, "Figure1_reference_metrics",
         ["figure1a_reference_simulation.csv", "figure1b_rare_run_points.csv", "figure1c_multiclass_population.csv"],
         "A simulated split of the background makes raw ARI fall while target F1 stays one. In both rare datasets, GMM parameter regimes and a fixed X-shift regime give different precision, recall and F1 profiles, with ARI near zero. Levine 32dim populations vary widely in precision and recall.")


def figure2():
    a = read("figure2a_event_inclusion.csv")
    b = read("figure2b_flowsom_metacluster_regimes.csv")
    c = read("figure2c_xshift_nilsson_regimes.csv")
    fig = plt.figure(figsize=(FIG_WIDTH, 6.3), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
    axa, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axb1, axb2 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    for metric, color, marker, name in [("ari", BLUE, "o", "ARI"), ("macro_f1", ORANGE, "s", "Macro F1")]:
        axa.plot([0, 1], a[metric], color=color, marker=marker, lw=1.6, ms=5, label=name)
        for j, value in enumerate(a[metric]):
            axa.annotate(f"{value:.3f}", (j, value), xytext=(0, 6),
                         textcoords="offset points", ha="center", fontsize=6.3, color=color)
    axa.set_xticks([0, 1], ["Labeled fit\n81,747 events", "All-event fit\n167,044 events"])
    axa.set(ylabel="Metric on evaluable events", ylim=(0, 1), xlim=(-.15, 1.15))
    axa.legend(frameon=False, loc="lower left")
    axa.grid(axis="y", color="#E1E5E8", linewidth=.5)
    label(axa, "A", "Same parameters and seed")

    regime_order = ["fixed_neighbor_K20", "published_neighbor_K60", "official_automatic_elbow"]
    c = c.set_index("regime").loc[regime_order].reset_index()
    for metric, color, marker, name in [("precision", BLUE, "o", "Precision"),
                                        ("recall", ORANGE, "s", "Recall"), ("f1", GREEN, "^", "F1")]:
        axc.scatter(np.arange(3), c[metric], color=color, marker=marker, s=30, label=name, zorder=3)
    axc.set_xticks(range(3), ["Fixed\nK=20", "Published\nK=60", "Automatic\nelbow"])
    axc.set(ylabel="Nilsson target metric", ylim=(0, 1.04))
    axc.grid(axis="y", color="#E1E5E8", linewidth=.5)
    axc.legend(frameon=False, ncol=3, loc="upper center", fontsize=6.2)
    label(axc, "C", "X-shift parameter regimes")

    regime_map = {"official_auto_max40": "Automatic", "official_fixed_ktrue": "K=true*", "official_fixed_k40": "Fixed K=40"}
    for ax, ds in [(axb1, "Levine_32dim"), (axb2, "Samusik_01")]:
        sub = b[b.dataset.eq(ds)]
        for j, (regime, shown) in enumerate(regime_map.items()):
            rows = sub[sub.regime.eq(regime)]
            if len(rows) != 30:
                raise ValueError((ds, regime, len(rows)))
            for metric, off, color, marker in [("ari", -.11, BLUE, "o"), ("macro_f1", .11, ORANGE, "s")]:
                vals = rows[metric].to_numpy(float)
                xx = j + off + np.linspace(-.045, .045, len(vals))
                ax.scatter(xx, vals, color=color, alpha=.30, s=8, marker=marker, linewidth=0)
                ax.scatter(j + off, vals.mean(), color=color, s=35, marker=marker,
                           edgecolor="white", linewidth=.4, zorder=4)
        ax.set_xticks(range(3), list(regime_map.values()))
        ax.set(xlim=(-.42, 2.42), ylim=(0, 1), ylabel="ARI / Macro F1")
        ax.grid(axis="y", color="#E1E5E8", linewidth=.5)
        panel = "B1" if ds == "Levine_32dim" else "B2"
        label(ax, panel, ds.replace("_", " "))
    axb2.legend(handles=[Line2D([0], [0], marker="o", color=BLUE, ls="none", label="ARI"),
                         Line2D([0], [0], marker="s", color=ORANGE, ls="none", label="Macro F1")],
                frameon=False, loc="upper left", fontsize=6.3)
    save(fig, "Figure2_information_access",
         ["figure2a_event_inclusion.csv", "figure2b_flowsom_metacluster_regimes.csv", "figure2c_xshift_nilsson_regimes.csv"],
         "A one-seed FlowSOM event-inclusion contrast changes both ARI and macro F1. In two multiclass datasets, 30 paired SOM seeds show how automatic, label-informed true-count and fixed-40 metaclustering regimes differ. Three X-shift regimes on Nilsson show a precision-recall tradeoff.")


def run_strip(ax, data, letter, title):
    datasets = list(dict.fromkeys(data.dataset))
    for j, ds in enumerate(datasets):
        rows = data[data.dataset.eq(ds)].sort_values("repeat_id")
        for metric, off, color, marker in [("ari", -.10, BLUE, "o"), ("macro_f1", .10, ORANGE, "s")]:
            vals = rows[metric].to_numpy(float)
            xx = j + off + np.linspace(-.045, .045, len(vals))
            ax.scatter(xx, vals, s=10, alpha=.47, color=color, marker=marker, linewidth=0)
            ax.scatter(j + off, vals.mean(), s=42, color=color, marker=marker,
                       edgecolor="white", linewidth=.4, zorder=4)
    ax.set_xticks(range(len(datasets)), [x.replace("_", " ") for x in datasets])
    ax.set(xlim=(-.45, len(datasets)-.55), ylim=(.4, 1), ylabel="ARI / Macro F1")
    ax.grid(axis="y", color="#E1E5E8", linewidth=.5)
    label(ax, letter, title)


def figure3():
    a = read("figure3a_flowsom_r_run_points.csv")
    b = read("figure3b_phenograph_run_points.csv")
    c = read("figure3c_within_method_partition_similarity.csv")
    fig = plt.figure(figsize=(FIG_WIDTH, 7.5), layout="constrained")
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.65])
    axa, axb, axc = [fig.add_subplot(gs[i]) for i in range(3)]
    run_strip(axa, a, "A", "R FlowSOM: 30 controlled seeds, full-event fit")
    run_strip(axb, b, "B", "Fixed-graph Louvain: 30 native repeats")
    handles = [Line2D([0], [0], color=BLUE, marker="o", ls="none", label="ARI"),
               Line2D([0], [0], color=ORANGE, marker="s", ls="none", label="Macro F1")]
    axa.legend(handles=handles, frameon=False, ncol=2, loc="lower left")
    short = {"FlowSOM_R_2.18.0_full_event": "R FlowSOM", "PhenoGraph_v1.5.7_default_Louvain": "PhenoGraph",
             "official_Xshift_fixed_neighbor_K20": "X-shift"}
    c = c.copy()
    c["display"] = c.apply(lambda r: f"{short[r.method]} | {r.dataset.replace('_', ' ')}", axis=1)
    c = c.sort_values(["method", "dataset"])
    y = np.arange(len(c))[::-1]
    for i, (_, row) in enumerate(c.iterrows()):
        color = {"R FlowSOM": BLUE, "PhenoGraph": ORANGE, "X-shift": GREEN}[short[row.method]]
        axc.plot([row.min_pairwise_ari, row.max_pairwise_ari], [y[i], y[i]], color=color, lw=2)
        axc.scatter(row.mean_pairwise_ari, y[i], color=color, marker="D", s=22,
                    edgecolor="white", linewidth=.35, zorder=3)
    axc.set_yticks(y, c.display)
    axc.set(xlim=(.68, 1.015), xlabel="Within-method pairwise partition ARI", ylabel="")
    axc.grid(axis="x", color="#E1E5E8", linewidth=.5)
    label(axc, "C", "Mean diamond; line = min–max across 435 dependent pairs")
    save(fig, "Figure3_repeat_stability",
         ["figure3a_flowsom_r_run_points.csv", "figure3b_phenograph_run_points.csv", "figure3c_within_method_partition_similarity.csv"],
         "Thirty fixed-protocol runs are shown for R FlowSOM and PhenoGraph. A third panel gives mean and minimum-to-maximum within-method partition ARI for each dataset; five X-shift fixed-K series are empirically identical. Pairwise values share runs and are dependent.")


def main():
    figure1()
    figure2()
    figure3()
    print("Rendered Figures 1–3 to", OUT)


if __name__ == "__main__":
    main()
