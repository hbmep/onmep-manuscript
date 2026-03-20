import os
import sys
import logging
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# import matplotlib as mpl
# os.environ.setdefault("MPLBACKEND", "Agg")
# mpl.use("Agg", force=True)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
from scipy.special import logsumexp
from joblib import Parallel, delayed

import hbmep as mep
from jax import random
import numpy as np
from rtmep import CheckpointManager, ObsArrays
from rtmep.model.simulate import Simulator

from rtmep.notebooks.util import load_model, get_response_colors
from simulate_data import BUILD_DIR as SIMULATION_DIR

plt.rcParams["svg.fonttype"] = "none"
logger = logging.getLogger(__name__)

K = 1
N0 = 8
N1 = 3
N = 64
FINAL_STEP = f"step_{N0 + N:06d}.pkl"
FINAL_STEP_NEW = f"step_{N0 + N:03d}.pkl"

KEYS = (mep.site.a, mep.site.b, mep.site.g, mep.site.h, mep.site.v)
RESPONSE = [
    'PKPK_APB',
    'PKPK_ECR',
    'PKPK_FCR'
]

BUILD_DIR = (
    "/home/vishu/reports/hbmep-adaptive/"
    "accuracy/out/400w_400s_4c_1t_10d_80a_fm"
)
D01 = (
    "/home/vishu/reports/hbmep-adaptive/accuracy/"
    "cumc01-out-nov24/400w_400s_4c_1t_10d_80a_fm"
)

RUN_IDS = [
    ("eig-focused-mcmc", D01),
    ("eig-mcmc", D01),
    ("random-mcmc", D01),
    ("eig-sequential-5", BUILD_DIR),
]

NUM_JOBS = 1
# NUM_JOBS = -1
TIMES_SEM = 1.96


def main():
    run_ids = RUN_IDS
    output_dir = BUILD_DIR
    samples_space = [3, 5, 8, 16, 24, 32, 40, 48, 56, 64]
    make_figure(output_dir, run_ids, samples_space)
    return


def load_true(draw, response):
    sim, _, post_one, *_ = load_model(SIMULATION_DIR, memmap_draw=draw)
    response_mapping = dict(zip(sim.response, range(sim.num_response)))
    true_response_idx = [response_mapping[u] for u in response]
    post_one = {u: post_one[u].reshape(-1,)[..., true_response_idx] for u in KEYS}
    post_one[mep.site.bth] = np.array(mep.functional.slope_at_threshold(
        post_one[mep.site.b], post_one[mep.site.h], post_one[mep.site.v]
    ))
    return post_one


@mep.timing
def make_figure(output_dir, run_ids, samples_space, n_jobs=NUM_JOBS):
    mep.enable_logging(os.path.join(output_dir, "make_figure.log"))
    labels_dict = {
        "random-mcmc": "Uniform (mcmc)",
        "eig-mcmc": "EIG (mcmc)",
        "eig-focused-mcmc": "Variance redn. (mcmc)",
        "eig-sequential": "EIG (sequential)",
    }
    draws, paths = get_paths(run_ids, response=RESPONSE)
    run_ids = [u[0] for u in run_ids]
    compare_params = [
        mep.site.a,
        # "bth"
    ]

    def body_load_true_one(named_param, draw):
        post_one = load_true(draw, response=RESPONSE)
        return post_one[named_param]

    def body_load_pred_one(named_param, draw, run_id, num_samples):
        path = paths[(draw, run_id)]
        ckpt = CheckpointManager(os.path.join(path, "checkpoints"))
        payload = ckpt.load_step(num_samples)

        posterior = payload["agent_state"]["posterior"]
        logw = payload["agent_state"].get("_logw", None)

        posterior["bth"] = np.array(mep.functional.slope_at_threshold(
            posterior[mep.site.b], posterior[mep.site.h], posterior[mep.site.v]
        ))

        if run_id.startswith("eig-sequential-"):
            if logw is None:
                raise ValueError("Missing _logw in checkpoint for eig-sequential.")
            logw = np.asarray(logw)
            w = np.exp(logw - logsumexp(logw))
            samp = posterior[named_param]
            S = samp.shape[0]
            key = random.key(draw + 1729)
            key, sub = random.split(key)
            idx = random.choice(key=sub, a=np.arange(S), shape=(S,), replace=True, p=w)
            idx = np.array(idx)
            out = samp[idx]
            return out
        else:
            posterior[named_param] = posterior[named_param][:, None]
            return posterior[named_param]

    true_tasks = []
    for pi, named_param in enumerate(compare_params):
        for di, draw in enumerate(draws):
            true_tasks.append((pi, di, named_param, draw))

    pred_tasks = []
    for pi, named_param in enumerate(compare_params):
        for di, draw in enumerate(draws):
            for ri, run_id in enumerate(run_ids):
                for si, num_samples in enumerate(samples_space):
                    pred_tasks.append((
                        pi, di, ri, si, named_param, draw, run_id, num_samples
                    ))

    true_results = Parallel(n_jobs=n_jobs)(
        delayed(body_load_true_one)(named_param, draw)
        for (_, _, named_param, draw) in true_tasks
    )
    t = true_results[0]; print(t.shape)

    pred_results = Parallel(n_jobs=n_jobs)(
        delayed(body_load_pred_one)(named_param, draw, run_id, N0 + num_samples)
        for (_, _, _, _, named_param, draw, run_id, num_samples) in pred_tasks
    )
    t = pred_results[0]; print(t.shape)

    true = np.array(true_results)
    true = true.reshape(len(compare_params), len(draws), *true.shape[1:])

    pred = np.array(pred_results)
    pred = pred.reshape(
        len(compare_params),
        len(draws),
        len(run_ids),
        len(samples_space),
        *pred.shape[1:]
    )
    pred_point = np.mean(pred, axis=-3)
    pred_point = pred_point[..., 0, :]
    assert true.shape[-1] == pred_point.shape[-1]

    # Make broadcastable
    true = true[:, :, None, None, ...]
 
    draws_dim = 1
    err = np.abs(pred_point - true)
    # err = np.abs(pred_point - true) / np.abs(true)
    num_draws = err.shape[draws_dim]
    err_mean = np.mean(err, axis=draws_dim) 
    err_err = TIMES_SEM * stats.sem(err, axis=draws_dim)

    num_response = len(RESPONSE)
    nr, nc = len(compare_params), num_response
    fig, axes = plt.subplots(
        *(nr, nc), figsize=(2.8 * nc, 3.2 * nr), constrained_layout=True,
        squeeze=False, sharex=True, sharey="row"
    )

    for i, named_param in enumerate(compare_params):
        for response_idx, response in enumerate(RESPONSE):
            ax = axes[i, response_idx]
            ax.clear()
            x = samples_space
            for ri, run_id in enumerate(run_ids):
                yme = err_mean[i, ri, :, response_idx]
                yerr = err_err[i, ri, :, response_idx]
                ax.errorbar(
                    x=x,
                    y=yme,
                    yerr=yerr,
                    linestyle="--",
                    marker="o",
                    ms=4,
                    label=labels_dict.get(run_id, run_id),
                )
            ax.spines[["top", "right"]].set_visible(False)
            if not i and compare_params[0] == mep.site.a:
                ax.axhline(
                    y = .5, color="k", linestyle="--",
                    linewidth=1, label="0.5 %MSO"
                )
                ax.axhline(
                    y = 1, color="k", linestyle="-",
                    linewidth=1, label="1 %MSO"
                )
                ax.set_title(
                    f"Muscle #{response_idx + 1} "
                    f"{RESPONSE[response_idx][5:]}"
                )
                if not response_idx:
                    ax.legend(frameon=False, reverse=False, fontsize=10)

    ax = axes[0, 0]
    ax.set_xlabel("Number of stimuli")
    ax.set_xticks(samples_space)
    xticklabels = [u for u in samples_space]
    ax.set_xticklabels(xticklabels)

    fig.align_xlabels()
    fig.align_ylabels()
    fig.suptitle(f"Total draws = {num_draws}\n")

    output_path = os.path.join(output_dir, "figure.pdf")
    mep.make_pdf([fig], output_path)
    output_path = os.path.join(output_dir, "figure.svg")
    fig.savefig(output_path, dpi=600)
    return


def get_paths(run_ids, response, final_step=None):
    space = list(range(1000))
    paths = {draw_idx: f"draw_{draw_idx:04d}" for draw_idx in space}
    def make_path(draw_name, run_id):
        response_dir = "__".join(response)
        return os.path.join(
            run_id[1], response_dir, run_id[0], draw_name
        )
    eg = make_path(paths[0], run_ids[0]); print(eg)
    os.path.exists(eg)
    paths = {
        draw: draw_name for draw, draw_name in paths.items()
        if all([
            os.path.exists(os.path.join(
                make_path(draw_name, run_id), "checkpoints",
                final_step if final_step is not None
                else (
                    FINAL_STEP
                    if not run_id[0].startswith("eig-sequential-")
                    else FINAL_STEP_NEW
                )
            ))
            for run_id in run_ids
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


def load_estimate(path, num_samples):
    subdir = "checkpoints"
    ckpt_dir = os.path.join(path, subdir)
    ckpt = CheckpointManager(ckpt_dir)
    payload = ckpt.load_step(num_samples)
    posterior = payload["agent_state"]["posterior"]
    model, *_ = load_model(path)
    data = ObsArrays.from_state(payload["agent_state"]["data"])
    df = data.to_df(**model.variables)
    posterior = {u: posterior[u] for u in KEYS}
    posterior["bth"] = np.array(mep.functional.slope_at_threshold(
        posterior[mep.site.b], posterior[mep.site.h], posterior[mep.site.v]
    ))
    return df, posterior, model, posterior


if __name__ == "__main__":
    main()
