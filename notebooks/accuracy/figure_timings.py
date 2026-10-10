import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLBACKEND", "Agg")
import pickle

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["svg.fonttype"] = "none"
from matplotlib import ticker

from paper.color_map import COLOR_MAP
from constants import MAKE_RESULTS_DIR, REPORTS

BUILD_DIR = os.path.join(REPORTS, "figs")
os.makedirs(BUILD_DIR, exist_ok=True)

SKIP = 8

FS = 7
LS = 6
LW = .8
GRID_LW = 0.4
GRID_ALPHA = 0.2

BOX_GAP = 1
BOX_WIDTH = 2
QLO = 0.025
QHI = 0.975

KEEP_RUN_IDS = [
    "eig-mcmc",
    "focused_eig_a-mcmc",
]


def plot_column(
    axes,
    *,
    metrics,
    response_len,
):
    src = os.path.join(
        MAKE_RESULTS_DIR,
        f"times_r{response_len:02d}.pkl"
    )
    with open(src, "rb") as f:
        out = pickle.load(f)

    out.keys()
    times = out["times"]
    idx = [out["run_ids"].index(u) for u in KEEP_RUN_IDS]
    times = times[:, idx]
    run_ids = KEEP_RUN_IDS
    idx = [out["metrics"].index(u) for u in metrics]
    times = times[..., idx]
    keep_space = [
        3,
        5,
        *range(8, 64 + 1, SKIP),
    ]
    idx = [out["ns_space"].index(u) for u in keep_space]
    times = times[..., idx, :]
    ns_space = keep_space
    times = times / 1e9

    num_run_ids = len(run_ids)
    offsets = (
        np.arange(num_run_ids)
        - (num_run_ids - 1) / 2
    ) * BOX_GAP

    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        for ri, run_id in enumerate(run_ids):
            if metric == "t_update_ns" and run_id != "eig-mcmc":
                continue
            color, label = COLOR_MAP[run_id]
            values = times[:, ri, :, mi]
            positions = np.asarray(ns_space, dtype=float) + offsets[ri]
            if metric == "t_update_ns":
                color = "black"
                label = "Update (MCMC)"
                values = times[:, :, :, mi]
                values = values.reshape(-1, *values.shape[2:])
                positions = ns_space
            ax.boxplot(
                values,
                positions=positions,
                widths=BOX_WIDTH,
                patch_artist=True,
                manage_ticks=False,
                showfliers=True,
                flierprops={
                    "marker": "o",
                    "markersize": 1.5,
                    "markerfacecolor": color,
                    "markeredgecolor": color,
                    "markeredgewidth": 0,
                    "alpha": 0.25,
                },
                whis=(QLO * 100, QHI * 100),
                medianprops={
                    "color": color,
                    "linewidth": LW,
                },
                boxprops={
                    "edgecolor": color,
                    "facecolor": "none",
                    "linewidth": LW,
                },
                whiskerprops={
                    "color": color,
                    "linewidth": LW,
                },
                capprops={
                    "color": color,
                    "linewidth": LW,
                },
            )

    for ax in axes:
        ax.set_xticks(ns_space)
        ax.set_xticklabels(ns_space)
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(
            axis="both",
            which="both",
            left=True,
            labelleft=True if response_len == 1 else False,
            bottom=True,
            labelbottom=True,
            labelsize=LS,
            pad=2,
        )
        ax.grid(
            True,
            axis="y",
            linewidth=GRID_LW,
            alpha=GRID_ALPHA,
        )

    return


def main():
    metrics = [
        "t_iter_ns",
        "t_update_ns",
        "t_propose_ns",
    ]

    nr, nc = 3, 2
    fig, axes = plt.subplots(
        nr,
        nc,
        figsize=(4.2, 5.5),
        constrained_layout=True,
        squeeze=False,
        gridspec_kw={
            "hspace": 0.0,
            "wspace": 0.0,
        },
    )

    for ax in axes.reshape(-1,):
        ax.sharex(axes[0, 0])

    for i in range(2):
        for j in range(2):
            axes[i, j].sharey(axes[0, 0])

    axes[-1, -1].sharey(axes[-1, 0])

    plot_column(
        axes[:, 0],
        metrics=metrics,
        response_len=1
    )

    plot_column(
        axes[:, 1],
        metrics=metrics,
        response_len=3
    )

    axes[0, 0].set_title(
        "One-muscle acquisition\n",
        fontsize=FS,
        fontweight="bold",
    )

    axes[0, 1].set_title(
        "Three-muscles acquisition\n(simultaneous)",
        fontsize=FS,
        fontweight="bold",
    )

    row_labels = [
        "Total time to acquire (s)",
        "Time to update (s)",
        "Time to propose (s)",
    ]
    for i in range(nr):
        ax = axes[i, 0]
        ax.set_ylabel(
            row_labels[i],
            fontsize=FS,
            labelpad=2,
        )
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))

    axes[-1, 0].set_xlabel(
        "Number of stimuli",
        fontsize=FS,
        labelpad=2,
    )

    ax = axes[0, 0]
    ax.set_ylim(top=6.4)
    ax = axes[-1, 0]
    ax.set_ylim(top=1.8)

    fig.align_labels()
    fig.align_xlabels()
    fig.align_ylabels()

    fname = "times"
    out = os.path.join(BUILD_DIR, f"{fname}.svg")
    fig.savefig(out, dpi=600)
    out = os.path.join(BUILD_DIR, f"{fname}.png")
    fig.savefig(out, dpi=600)
    print("Saved", out)
    return


if __name__ == "__main__":
    main()
