import os
from pathlib import Path
import json
import logging

# import matplotlib as mpl
# os.environ.setdefault("MPLBACKEND", "Agg")
# mpl.use("Agg", force=True)

import hbmep as mep
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
import numpy as np
import pandas as pd

from core_job import K, N0, N1, N, GRID_N
from results import get_paths

plt.rcParams["svg.fonttype"] = "none"
logger = logging.getLogger(__name__)

K = 1
N0 = 8
N1 = 3
N = 64
FINAL_STEP = f"step_{N0 + N:03d}.pkl"

RESPONSE = [
    "PKPK_APB",
    "PKPK_ECR",
    "PKPK_FCR"
]

BUILD_DIR = "/home/vishu/reports/hbmep-adaptive/accuracy/out-time/400w_400s_4c_1t_10d_80a_fm"
RUN_ID = "eig-mcmc"

D01 = (
    "/home/vishu/reports/hbmep-adaptive/"
    "accuracy/out-time-saved/400w_400s_4c_1t_10d_80a_fm"
)
D02 = (
    "/home/vishu/reports/hbmep-adaptive/"
    "accuracy/out-time-sequential/400w_400s_4c_1t_10d_80a_fm"
)

RUN_IDS = [
    ("eig-mcmc", D01),
    # ("eig-sequential", D02)
]

FS = 8


def plot(axes, run_ids, response, num_metrics):
    labels_dict = {
        "random-mcmc": "Uniform (mcmc)",
        "eig-mcmc": "EIG (mcmc)",
        "eig-focused-mcmc": "Variance redn. (mcmc)",
        "eig-sequential": "EIG (sequential)",
    }
    draws, paths = get_paths(run_ids, response=response, final_step=FINAL_STEP)
    num_samples_space_2 = [3, 5, 8, 16, 24, 32, 40, 48, 56, 64]
    num_samples_space = list(range(3, 65))
    metrics = [
        't_propose_ns',
        't_update_ns',
        't_iter_ns',
        't_step_ns',
        't_ckpt_ns',
    ]
    metrics_mapping = {
        't_propose_ns': "Propose step",
        't_update_ns': "Update step",
        't_iter_ns': "Total iteration",
    }
    times = []
    for nd, draw in enumerate(draws):
        for nid, (run_id, _) in enumerate(run_ids):
            run_id_dir = paths[(draw, run_id)]
            records = []
            src = Path(run_id_dir) / "timings.ndjson"
            with open(src) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if record['step'] - N0 in num_samples_space:
                        for metric in metrics:
                            times.append(record[metric])
    times = np.array(times, dtype=float)
    times = times.reshape(len(draws), len(run_ids), -1, len(metrics))
    times[..., 2] = times[..., 0] + times[..., 1]
    times *= 1e-9
    times_mean = np.mean(times, axis=0)
    times_err = np.std(times, ddof=1, axis=0)
    num_draws, num_runs, num_steps, _ = times.shape
    xticklabels = [str(n) for n in num_samples_space]
    positions = np.array(num_samples_space, dtype=float)
    # yticks = [0, 2.5, 5, 7.5, 10]
    # yticks = [i for i in range(0, 11, 2)]
    # # yticklabels = [0, 2.5, 5.0, 7.5, 10.0]

    for nid, (run_id, _) in enumerate(run_ids):
        for nmet, metric in enumerate(metrics):
            if nmet >= num_metrics:
                break
            ax = axes[nid, nmet]
            ax.clear()

            arr = times[:, nid, :, nmet]
            med = np.median(arr, axis=0)
            q25, q75 = np.percentile(arr, [25, 75], axis=0)
            q10, q90 = np.percentile(arr, [10, 90], axis=0)

            ax.fill_between(positions, q25, q75, alpha=0.25)
            ax.plot(positions, med, marker='o', markersize=2, linewidth=1.0)
            ax.errorbar(
                positions,
                med,
                yerr=np.vstack([med - q10, q90 - med]),
                fmt='none',
                capsize=2,
                elinewidth=0.75,
                alpha=0.6,
                label=labels_dict.get(run_id, run_id),
            )

            if nid == 0:
                ax.set_title(
                    metrics_mapping[metric]
                    if metric in metrics_mapping else metric,
                    fontsize=FS
                )
            if nmet == 0:
                ax.set_ylabel(run_id, fontsize=FS)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(
                axis="both", left=True,
                # labelleft=True if nmet == 0 else False,
                labelleft=True,
                bottom=True, labelbottom=True, labelsize=7
            )
            # if nmet == 0:
            #     ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
            #     ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            # else:
            #     ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
            # ax.set_yticks(yticks)
            # ax.set_yticklabels(yticklabels)
            # if nmet == 0:
            #     ax.axhline(y=0.3, color="r", linestyle='--', linewidth=.25)
            ax.set_xticks(num_samples_space_2)
            ax.set_xticklabels(num_samples_space_2, rotation=0)
            ax.set_xlabel("")
    return


def main():
    mep.enable_logging(BUILD_DIR)
    run_ids = RUN_IDS

    plt.close("all")
    num_metrics = 3
    nr, nc = len(run_ids), num_metrics
    fig, axes = plt.subplots(
        nr, nc,
        figsize=(8, 4.0),
        constrained_layout=True,
        squeeze=False,
        sharex=True,
        sharey='col'
    )
    fig.show()
    response = RESPONSE
    plot(axes, run_ids, response=response, num_metrics=num_metrics)

    ax = axes[-1, 0]
    ax.set_xlabel("Number of stimuli", fontsize=FS)

    ax = axes[0, 0]
    ax.set_ylabel("Time taken (seconds)", fontsize=FS)

    ax = axes[1, 0]
    ax.set_ylabel("Time taken (seconds)", fontsize=FS)

    for i in range(nr):
        ax = axes[i, 0]
        ax.legend(frameon=False, fontsize=FS + 2)

    fig.align_labels()
    fig.align_xlabels()
    fig.align_ylabels()

    out = Path(BUILD_DIR) / "time.svg"
    fig.savefig(out, dpi=600)
    out = Path(BUILD_DIR) / "time.pdf"
    fig.savefig(out, dpi=600)
    logger.info(f"Saved to {out}")

    # response = ["PKPK_APB", "PKPK_ECR", "PKPK_FCR"]
    # plot(axes[1], run_ids, response=response, num_metrics=num_metrics)
    return


if __name__ == "__main__":
    main()
