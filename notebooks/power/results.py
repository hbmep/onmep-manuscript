import os
import sys
import logging
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib as mpl
os.environ.setdefault("MPLBACKEND", "Agg")
mpl.use("Agg", force=True)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
from joblib import Parallel, delayed

import hbmep as mep
import numpy as np
from rtmep import CheckpointManager, ObsArrays
from rtmep.model.simulate import Simulator

from rtmep.notebooks.util import load_model, get_response_colors

plt.rcParams["svg.fonttype"] = "none"
logger = logging.getLogger(__name__)

K = 2
N0 = 8
N1 = 3
N = 32
FINAL_STEP = f"step_{2 * (N0 + N):03d}.pkl"
TOTAL_SUBJECTS = 16

KEYS = (mep.site.a, mep.site.b, mep.site.g, mep.site.h, mep.site.v)
RESPONSE = [
    'PKPK_APB',
    # 'PKPK_ECR',
    # 'PKPK_FCR'
]

TP_MODEL_DIR = (
    "/home/vishu/reports/hbmep-adaptive/accuracy/"
    "learn_posterior/simulate_power_data_tp"
)
FP_MODEL_DIR = (
    "/home/vishu/reports/hbmep-adaptive/accuracy/"
    "learn_posterior/simulate_power_data_fp"
)

TP_EIG_MCMC = (
    "/home/vishu/reports/hbmep-adaptive/power/"
    "tp/400w_400s_4c_1t_10d_80a_fm/"
)
FP_EIG_MCMC = (
    "/home/vishu/reports/hbmep-adaptive/power/"
    "fp/400w_400s_4c_1t_10d_80a_fm/"
)

TP_RUN_IDS = [
    ("eig-mcmc", TP_EIG_MCMC),
    # ("random-mcmc", TP_EIG_MCMC),
]
FP_RUN_IDS = [
    ("eig-mcmc", FP_EIG_MCMC),
    # ("random-mcmc", FP_EIG_MCMC),
]

NUM_JOBS = 1
NUM_JOBS = -1
TIMES_SEM = 1.96

# MODEL_DIR = TP_MODEL_DIR
# RUN_IDS = TP_RUN_IDS

MODEL_DIR = FP_MODEL_DIR
RUN_IDS = FP_RUN_IDS

BUILD_DIR = "/home/vishu/reports/hbmep-adaptive/power"
SIGNIFICANCE_LEVEL = 0.05


def main():
    run_ids = RUN_IDS
    output_dir = BUILD_DIR
    # samples_space = [6, 10, 16, 24, 32, 40, 48, 56, 64]
    samples_space = [64]
    make_figure(output_dir, run_ids, samples_space)
    return


def load_true(draw, response):
    sim, _, post_one, *_ = load_model(MODEL_DIR, memmap_draw=draw)
    response_mapping = dict(zip(sim.response, range(sim.num_response)))
    true_response_idx = [response_mapping[u] for u in response]
    post_one = {u: post_one[u][0][..., true_response_idx] for u in KEYS}
    return post_one


@mep.timing
def make_figure(output_dir, run_ids, samples_space, n_jobs=NUM_JOBS):
    mep.enable_logging(os.path.join(output_dir, "make_figure.log"))
    labels_dict = {
        "random-mcmc": "Uniform (mcmc)",
        "eig-mcmc": "EIG (mcmc)",
        "eig-focused-mcmc": "Variance red. (mcmc)",
    }
    draws, paths = get_paths(run_ids, final_step=FINAL_STEP, response=RESPONSE)
    run_ids = [u[0] for u in run_ids]
    compare_params = [mep.site.a]

    def body_load_true_one(named_param, draw):
        post_one = load_true(draw, response=RESPONSE)
        return post_one[named_param]

    def body_load_pred_one(named_param, draw, run_id, pid, num_samples):
        path = paths[(draw, run_id)]
        ckpt = CheckpointManager(os.path.join(path, f"p{pid:02d}", "checkpoints"))
        payload = ckpt.load_step(num_samples)
        posterior = payload["agent_state"]["posterior"]
        return posterior[named_param]

    true_tasks = []
    for pi, named_param in enumerate(compare_params):
        for di, draw in enumerate(draws):
            true_tasks.append((pi, di, named_param, draw))

    pred_tasks = []
    for pi, named_param in enumerate(compare_params):
        for di, draw in enumerate(draws):
            for ri, run_id in enumerate(run_ids):
                for pid in range(TOTAL_SUBJECTS):
                    for si, num_samples in enumerate(samples_space):
                        pred_tasks.append((
                            pi, di, ri, si, named_param, draw, run_id, pid, num_samples
                        ))

    true_results = Parallel(n_jobs=n_jobs)(
        delayed(body_load_true_one)(named_param, draw)
        for (pi, di, named_param, draw) in true_tasks
    )
    t = true_results[0]; print(t.shape)

    pred_results = Parallel(n_jobs=n_jobs)(
        delayed(body_load_pred_one)(named_param, draw, run_id, pid, (K * N0) + num_samples)
        for (pi, di, ri, si, named_param, draw, run_id, pid, num_samples) in pred_tasks
    )
    t = pred_results[0]; print(t.shape)

    true = np.array(true_results)
    true = true[..., :TOTAL_SUBJECTS, :, :]
    true = true.reshape(len(compare_params), len(draws), *true.shape[1:])
    assert len(compare_params) == 1
    assert true.shape[-1] == 1
    true = true[0, ..., 0]

    pred = np.array(pred_results)
    pred = pred.reshape(len(compare_params), len(draws), len(run_ids), TOTAL_SUBJECTS, len(samples_space), *pred.shape[1:])
    assert pred.shape[-1] == 1
    pred = pred[0, ..., 0]
    assert len(samples_space) == 1
    pred = pred[:, :, :, 0]
    pred = pred.swapaxes(-2, -3)
    pred_point = np.mean(pred, axis=-3)
    assert true.shape == pred_point[:, 0].shape

    # Power analysis
    print(true.shape)       # (len(draws), TOTAL_SUBJECTS, 2)
    print(pred_point.shape) # (len(draws), len(run_ids), TOTAL_SUBJECTS, 2)

    reject = []
    num_subjects_space = [2, 4, 5, 6, 8, 9, 10, 11, 12, 14, 16]
    for ri, run_id in enumerate(run_ids):
        for si, num_subjects in enumerate(num_subjects_space):
            x = pred_point[:, ri, :num_subjects]
            # x = true[:, :num_subjects]
            diff = x[..., 0] - x[..., 1]
            pr = stats.wilcoxon(x=diff, alternative="two-sided", axis=1).pvalue
            decision = pr < SIGNIFICANCE_LEVEL
            reject.append(decision)
    reject = np.array(reject)
    reject = reject.reshape(len(run_ids), len(num_subjects_space), *reject.shape[1:])
    reject = reject.swapaxes(0, -1)
    reject_mean = np.mean(reject, axis=0)
    reject_err = TIMES_SEM * stats.sem(reject, axis=0)

    num_response = 1
    nr, nc = 1, num_response
    plt.close("all")
    fig, axes = plt.subplots(
        nr, nc,
        figsize=(4, 3.5),
        constrained_layout=True,
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    fig.show()
    x = num_subjects_space
    ax = axes[0, 0]
    ax.clear()
    for ri, run_id in enumerate(run_ids):
        yme = reject_mean[..., ri]
        yerr = reject_err[..., ri]
        ax.errorbar(
            x=x,
            y=yme,
            yerr=yerr,
            linestyle="--",
            marker="o",
            ms=4,
            label=labels_dict.get(run_id, run_id),
        )
        ax.set_xticks(num_subjects_space)

    ax = axes[0, 0]
    ax.grid(True, alpha=0.4)
    fig.align_labels()
    fig.align_xlabels()
    fig.align_ylabels()
    fig.suptitle(f"Total draws: {len(draws)}")
    out = os.path.join(BUILD_DIR, "out.png")
    fig.savefig(out, dpi=600)
    print(f"Saved to {out}")
 
    return


def get_paths(run_ids, final_step, response):
    space = list(range(1000))
    paths = {draw_idx: f"d{draw_idx:04d}" for draw_idx in space}
    def make_path(draw_name, run_id):
        response_dir = "__".join(response)
        return os.path.join(
            run_id[1], response_dir, run_id[0], draw_name
        )
    # eg = make_path(paths[0], run_ids[0]); print(eg)
    # os.path.exists(eg)
    paths = {
        draw: draw_name for draw, draw_name in paths.items()
        if all([
            os.path.exists(os.path.join(
                make_path(draw_name, run_id), f"p{pid:02d}", "checkpoints", final_step
            ))
            for run_id in run_ids
            for pid in range(TOTAL_SUBJECTS)
        ])
    }
    draws = sorted(paths.keys())
    missing_draws = [u for u in space if u not in draws]
    if missing_draws:
        print(f"min missing draw: {min(missing_draws)}")

    paths = {
        (draw, run_id[0]): make_path(draw_name, run_id)
        for draw, draw_name in paths.items()
        for run_id in run_ids
    }
    return draws, paths


if __name__ == "__main__":
    main()
