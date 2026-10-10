import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLBACKEND", "Agg")
import pickle
from random import randint

import hbmep as mep
import numpy as np
from numpyro.diagnostics import hpdi
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap, to_rgb, ListedColormap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns

from paper.color_map import COLOR_MAP

from constants import (
    REPORTS,
    MAKE_RESULTS_DIR,
    SAMPLES_DIR,
)

BUILD_DIR = os.path.join(REPORTS, "figs")
os.makedirs(BUILD_DIR, exist_ok=True)
# MAKE_RESULTS_DIR = "/home/vishu/reports/c2b2-Jun-2026/make_results"

SHOW_LEGEND = False
if not SHOW_LEGEND:
    plt.rcParams["svg.fonttype"] = "none"
else:
    plt.rcParams["svg.fonttype"] = "path"
    plt.rcParams["mathtext.fontset"] = "cm"

NUM_DRAWS = 2000

USE_RELATIVE = False

SKIP = 8
GRADATION_MAX = 0.65

FS = 7
LS = 6
LW = .8
MS = 1.5
IN_LW = .5
IN_MS = .8
GRID_LW = 0.4
GRID_ALPHA = 0.2
IN_GRID_LW = 0.3
IN_GRID_ALPHA = 0.2

SET_YSCALE_LOG = False
X_JITTER_STEP = .5

KEEP_RUN_IDS = [
    "uniform-mcmc",
    "pest-mle1",
    "focused_eig_a-mcmc",
    "eig-mcmc",
]

YLABLES = [
    "Threshold\nAbsolute error (% MSO)",
    "Predictive curve\nIntegrated absolute error (% MSO)",
]


def plot_accuracy_panel(
    ax,
    *,
    param,
    response_len,
):
    model_name = "rl_gamma"
    src = os.path.join(
        MAKE_RESULTS_DIR,
        f"{response_len:02d}_{model_name}_{param}_{NUM_DRAWS:04d}.pkl"
    )
    with open(src, "rb") as f:
        out = pickle.load(f)
    assert len(out["response"]) == response_len
    assert out["draws"].shape[0] == NUM_DRAWS

    keep_space = [
        3,
        5,
        *range(8, 64 + 1, SKIP),
    ]
    keep_run_ids = [u for u in KEEP_RUN_IDS]

    err = out["err"]
    run_ids = out["run_ids"]
    ns_space = out["ns_space"]

    idx = [ns_space.index(u) for u in keep_space]
    err = err[..., idx, :]
    ns_space = keep_space
    idx = [run_ids.index(u) for u in keep_run_ids]
    err = err[:, idx]
    run_ids = keep_run_ids

    err = np.mean(err, axis=-1)
    err_me = np.mean(err, axis=0)
    err_sem = stats.sem(err, axis=0)

    for run_id in run_ids:
        if run_id not in COLOR_MAP:
            COLOR_MAP[run_id] = (
                f"#{randint(0, 0xFFFFFF):06x}",
                run_id,
            )

    if SET_YSCALE_LOG:
        x_offsets = (
            np.arange(len(run_ids))
            - (len(run_ids) - 1) / 2
        ) * X_JITTER_STEP

    for ri, run_id in enumerate(run_ids):
        color, label = COLOR_MAP[run_id]
        yme = err_me[ri]
        yerr = err_sem[ri]
        ax.errorbar(
            x=(
                np.asarray(ns_space) + x_offsets[ri]
                if SET_YSCALE_LOG and "evr-mcmc" in KEEP_RUN_IDS
                else ns_space
            ),
            y=yme,
            yerr=yerr,
            linestyle="-",
            marker="o",
            color=color,
            ms=MS,
            linewidth=LW,
            label=label,
            zorder=2,
        )

    ax.set_xticks(ns_space)
    ax.set_xticklabels(ns_space)
    ax.tick_params(
        axis="both",
        which="both",
        left=True,
        labelleft=True,
        bottom=True,
        labelbottom=True,
        labelsize=LS,
        pad=2,
    )
    ax.spines[["top", "right"]].set_visible(False)
    return


def plot_samples_panel(ax, params, response_len):
    def load():
        src = os.path.join(
            SAMPLES_DIR,
            "make_samples",
            f"{response_len:02d}_samples.pkl"
        )
        with open(src, "rb") as f:
            out = pickle.load(f)
        a = out["a"]
        run_ids = out["run_ids"]
        draws = out["draws"]
        if "evr-mcmc" not in KEEP_RUN_IDS:
            assert sorted(run_ids) == sorted(KEEP_RUN_IDS)
        order = [
            "pest-mle1",
            "focused_eig_a-mcmc",
            "uniform-mcmc",
            "eig-mcmc"
        ]
        assert sorted(run_ids) == sorted(order)
        idx = [run_ids.index(u) for u in order]
        a = a[..., idx]
        run_ids = order
        return a, run_ids, draws,
    a, samples_run_ids, draws = load()
    assert draws.shape[0] in {499, 500}
    samples = np.median(a, axis=-2)
    ax.set_axisbelow(True)
    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(
        axis="both",
        which="both",
        left=True,
        labelleft=True,
        bottom=True,
        labelbottom=True,
        labelsize=LS,
        pad=2,
    )
    a = np.median(a, axis=-2)

    for i, samples_run_id in enumerate(samples_run_ids):
        color, label = COLOR_MAP[samples_run_id]
        x = samples[:, 0, i]
        y = samples[:, 1, i]
        rgb = np.array(to_rgb(color))
        light = rgb + (1 - rgb) * GRADATION_MAX
        cmap = LinearSegmentedColormap.from_list(
            f"cmap_{i}",
            [light, rgb],
        )
        levels = np.linspace(0.1, 0.90, 10)
        num_levels = len(levels)
        colors = [cmap(v) for v in np.linspace(0, 1, num_levels)]
        bw = 1.4
        sns.kdeplot(
            x=x, y=y,
            levels=levels,
            thresh=0.05,
            fill=False, bw_adjust=bw, linewidths=.8,
            alpha=1, ax=ax, colors=colors, zorder=2 * i + 2,
        )
        xy = np.vstack([x, y])
        kde = stats.gaussian_kde(xy)
        kde.set_bandwidth(kde.factor * bw)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        ax.scatter(
            x_mean,
            y_mean,
            s=12,
            marker="o",
            color=color,
            edgecolor="white",
            linewidth=0.4,
            zorder=100,
        )
    return


def plot_column(
    axes,
    *,
    params,
    response_len,
):
    for row_idx, param in enumerate(params):
        plot_accuracy_panel(
            axes[row_idx],
            param=param,
            response_len=response_len,
        )
    plot_samples_panel(axes[-1], params, response_len)
    return


def main():
    params = [mep.site.a, "mu_int"]
    nr, nc = 3, 2
    fig, axes = plt.subplots(
        nr,
        nc,
        figsize=(4.2, 5.5),
        constrained_layout=True,
        squeeze=False,
        gridspec_kw={
            "hspace": 0.,
            "wspace": 0.,
        },
    )

    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            ax = axes[i, j]
            if i == axes.shape[0] - 1:
                ax.sharex(axes[-1, 0])
                ax.sharey(axes[-1, 0])
                continue
            ax.sharex(axes[0, 0])
            ax.sharey(axes[i, 0])

    plot_column(
        axes[:, 0],
        params=params,
        response_len=1,
    )

    plot_column(
        axes[:, 1],
        params=params,
        response_len=3,
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

    for ax in axes.reshape(-1):
        ax.set_xlabel("")
        ax.set_ylabel("")

    for row_idx, ylabel in enumerate(YLABLES):
        axes[row_idx, 0].set_ylabel(
            ylabel,
            fontsize=FS,
            labelpad=2,
        )

    for i in range(len(params)):
        ax = axes[i, 0]
        if SET_YSCALE_LOG:
            ax.set_yscale("log", base=2)
        else:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        for j in range(nc):
            ax = axes[i, j]
            ax.grid(
                True,
                axis="y",
                which="major",
                linewidth=GRID_LW,
                alpha=GRID_ALPHA,
            )

    for ax in axes[:, 1]:
        ax.tick_params(axis="y", labelleft=False)

    for ax in axes[-1, :]:
        ax.axline(
            (0, 0), slope=1, 
            color="r", 
            linestyle="--", 
            linewidth=GRID_LW * 2, 
            alpha=0.6, 
            zorder=int(1e6),
        )

    ax = axes[-1, 0]
    ax.set_xlabel("Average number of stimuli to reach\n2% relative threshold error", fontsize=FS, labelpad=2)
    ax.set_ylabel("Stimuli to reach\n10% relative predictive error", fontsize=FS, labelpad=2)
    ax.set_xlim(left=2.25, right=8.2)
    ax.set_ylim(bottom=3.6, top=12.7)
    xticks = [3, 4, 5, 6, 7, 8]
    xticklabels = [2 ** i for i in xticks]
    xticklabels = [f"$2^{{{i}}}$" for i in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    yticks = [4, 6, 8, 10, 12]
    yticklabels = [2 ** i for i in yticks]
    yticklabels = [f"$2^{{{i}}}$" for i in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax = axes[-1, 1]
    ax.tick_params(axis="y", labelleft=False)

    for ax in [axes[0, 0], axes[1, 0]]:
        ax.set_xlabel(
            "Number of stimuli",
            fontsize=FS,
            labelpad=2,
        )

    if SET_YSCALE_LOG:
        ax = axes[0, 0]
        ax.set_ylim(top=22)

    if SHOW_LEGEND or SET_YSCALE_LOG:
        axes[0, 0].legend(
            loc="upper left",
            bbox_to_anchor=(0.02, 1.0),
            ncol=1,
            frameon=True,
            fontsize=LS,
            columnspacing=0.6,
        )

    fig.align_labels()
    fig.align_xlabels()
    fig.align_ylabels()

    filename = "acc"
    out_svg = os.path.join(BUILD_DIR, f"{filename}.svg")
    fig.savefig(out_svg, dpi=600)
    out_png = os.path.join(BUILD_DIR, f"{filename}.png")
    fig.savefig(out_png, dpi=600)
    print("Saved", out_png)
    return


if __name__ == "__main__":
    main()
