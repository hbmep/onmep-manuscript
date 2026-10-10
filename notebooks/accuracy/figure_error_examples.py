import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLBACKEND", "Agg")
from random import randint

import hbmep as mep
import numpy as np
from numpyro.diagnostics import hpdi

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import ticker
import seaborn as sns

from paper.model.model_samples import HBestimation
from paper.color_map import COLOR_MAP

from constants import (
    SAMPLES_DIR,
    REPORTS,
)

plt.rcParams["svg.fonttype"] = "none"
BUILD_DIR = os.path.join(REPORTS, "errors")
os.makedirs(BUILD_DIR, exist_ok=True)
# MAKE_RESULTS_DIR = "/home/vishu/reports/c2b2-Jun-2026/make_results"
# SAMPLES_DIR = "/home/vishu/reports/c2b2-Jun-2026/out_samples"

NUM_POINTS = 2000

FS = 8
LS = 7
TICK_PAD = 1
XLABEL_PAD = 2
YLABEL_PAD = 2
KDE_LW = .9
GRID_LW = 0.5
BW_ADJUST = 2
SCATTER_MS = 10


def run_one(axes, draw, param, ax_group):
    nr, nc = axes.shape
    for i in range(nr):
        for j in range(nc):
            ax = axes[i, j]
            ax.sharey(axes[0, 0])

    num_subjects = 16
    model_dir = os.path.join(
        SAMPLES_DIR,
        "r01",
        param,
        f"d{draw:04d}",
        f"n{num_subjects:02d}",
    )
    model_state, post, df, *_ = mep.load(model_dir)
    model = HBestimation().load_state_dict(model_state)

    a = post[mep.site.a]

    pred_df = model.make_prediction_dataset(df, num_points=NUM_POINTS)
    ppd = model.predict(pred_df, posterior=post)

    mu = ppd[mep.site.mu]
    mu_mean = np.mean(mu, axis=0)

    obs = ppd[mep.site.obs]
    obs_hdi = hpdi(obs, axis=0, prob=.95)

    if param == mep.site.a:
        ref = np.log2(2)
        ref_sty = "--"
        param_name = "a"
        annotation_color = "#D55E00"   # vermillion
    else:
        ref = np.log2(10)
        ref_sty = "--"
        param_name = "y"
        annotation_color = "#007C83"   # dark teal

    mu_0 = post["a_fixed_loc"]
    mu_delta = post["a_delta_loc"]
    mu_combined = np.concatenate([mu_0[:, None], mu_0[:, None] + mu_delta], axis=-1)

    for i in range(nr):
        idx = df[model.features[0]] == i
        temp_df = df.iloc[idx].reset_index(drop=True).copy()
        idx = pred_df[model.features[0]] == i
        temp_pred_df = pred_df.iloc[idx].reset_index(drop=True).copy()
        temp_mu = mu_mean[idx]
        temp_obs_hdi = obs_hdi[:, idx]
        for j in range(nc):
            response = model.response[j]
            run_id = response.split("__")[-1]
            color, label = COLOR_MAP[run_id]
            ax = axes[i, j]
            ax.clear()
            ax.fill_between(
                temp_pred_df[model.intensity],
                temp_obs_hdi[0][:, j],
                temp_obs_hdi[1][:, j],
                color=color,
                alpha=.1,
            )
            sns.scatterplot(
                x=temp_df[model.intensity],
                y=temp_df[response],
                ax=ax,
                color=color,
                s=SCATTER_MS,
            )
            ax.plot(
                temp_pred_df[model.intensity],
                temp_mu[:, j],
                color="k",
                label=label,
            )
            ax.axhline(
                ref,
                linestyle=ref_sty,
                linewidth=0.8,
                color=annotation_color,
                zorder=4,
                xmax=.95
            )
            xlim = ax.get_xlim()
            ax2 = inset_axes(
                ax,
                width="100%",
                height="25%",
                loc="lower left",
                bbox_to_anchor=(0, 0, 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0,
            )
            ax2.sharex(ax)
            ax2.set_zorder(3)
            ax2.set_facecolor("none")
            ax2.spines[
                ["right", "left", "top", "bottom"]
            ].set_visible(False)
            ax2.tick_params(
                axis="both",
                which="both",
                left=False,
                right=False,
                top=False,
                bottom=False,
                labelleft=False,
                labelbottom=False,
            )
            ax2.set_yticks([])
            ax2.set_xlabel("")
            ax2.set_ylabel("")
            samples = a[:, i, j]
            sns.kdeplot(
                x=samples,
                ax=ax2,
                color=color,
                linewidth=KDE_LW,
                clip=xlim,
                cut=0,
                bw_adjust=BW_ADJUST,
            )
            ax2.set_ylim(bottom=0)
            ax.set_xlim(xlim)
            ax.text(
                0.90,
                0.90,
                fr"${param_name}_{{({i + 1},{j + 1})}}$",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=LS,
                color=annotation_color,
                zorder=100,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor=annotation_color,
                    linewidth=0.6,
                    alpha=1,
                ),
            )
            if not i:
                ax = ax_group
                samples = mu_combined[:, j]
                sns.kdeplot(
                    x=samples,
                    ax=ax,
                    color=color,
                    linewidth=KDE_LW,
                    bw_adjust=BW_ADJUST,
                    label=label,
                )
    return


def main(draw):
    nr, nc = 8, 8
    fig = plt.figure(
        figsize=(8, 9),
        constrained_layout=True,
    )
    gs = fig.add_gridspec(
        nrows=9,
        ncols=8,
        height_ratios=[1] * 8 + [1.35],
    )
    axes = np.empty((nr, nc), dtype=object)
    for i in range(nr):
        for j in range(nc):
            axes[i, j] = fig.add_subplot(gs[i, j])
    ax_group_a = fig.add_subplot(gs[8, :4])
    ax_group_y = fig.add_subplot(gs[8, 4:])

    for i in range(nr):
        for j in range(nc):
            ax = axes[i, j]
            ax.sharex(axes[0, 0])
            ax.spines[['right', 'top']].set_visible(False)

    run_one(axes[:, :4], draw, mep.site.a, ax_group_a)
    run_one(axes[:, 4:], draw, "mu_int", ax_group_y)

    for i in range(nr):
        for j in range(nc):
            ax = axes[i, j]
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(
                axis="both",
                bottom=True,
                labelbottom=True,
                left=True,
                labelleft=True if j in {0, 4} else False,
                labelsize=LS,
                pad=TICK_PAD,
            )

    for ax in [axes[0, 0], axes[0, 4], axes[-1, 0], axes[-1, 4]]:
        ax.set_xlabel(r"$\log_2(n)$", fontsize=FS, labelpad=XLABEL_PAD)

    titles = [
        "EIG",
        "EIG focused\non threshold",
        "ML-PEST",
        "Uniform\nsampling"
    ] * 2
    for j in range(nc):
        ax = axes[0, j]
        ax.set_title(titles[j], fontsize=FS, pad=0)

    for i in range(nr):
        for j in {0, 4}:
            ax = axes[i, j]
            if j == 0:
                ylab = "$\log_2(r_a)$"
            else:
                ylab = "\n\n$\log_2(r_y)$"
            ax.set_ylabel(ylab, fontsize=FS, labelpad=YLABEL_PAD)

    subset = [ax_group_a, ax_group_y]
    for ax in subset:
        ax.tick_params(
            axis="both",
            left=False,
            labelleft=False,
            labelsize=LS + 2,
            pad=TICK_PAD,
        )
        ax.set_ylabel("Prob. density", fontsize=FS, labelpad=YLABEL_PAD)
        ax.set_xlabel(r"$\log_2(n)$", fontsize=FS, labelpad=XLABEL_PAD)
        ax.spines[['right', 'top']].set_visible(False)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))

    ax = axes[0, 0]
    ax.set_xticks([2, 3, 4, 5, 6])

    ax = ax_group_a
    ax.set_xlim(left=1.8, right=7)
    ax.set_xticks([i for i in range(2, 7)])
    ax.legend(fontsize=FS)
    ax.set_title(
        "Threshold error summary for N=16 participants",
        pad=0,
        fontsize=FS
    )

    ax = ax_group_y
    ax.set_xlim(left=1.8, right=16.5)
    ax.set_xticks([2 * i for i in range(1, 9)])
    ax.set_title(
        "Predictive error summary for N=16 participants",
        pad=0,
        fontsize=FS
    )

    fig.align_labels()
    fig.align_xlabels()
    fig.align_ylabels()
    fname = f"error_{draw:04d}"
    for ext in ["svg", "png"]:
        output_path = os.path.join(
            BUILD_DIR,
            f"{fname}.{ext}"
        )
        fig.savefig(output_path, dpi=600)
    print(f"Saved to {output_path}")
    return


if __name__ == "__main__":
    num_draws = 200
    tasks = [(draw,) for draw in range(num_draws)]
    mep.run_batched(main, tasks=tasks, batch_size=1)
