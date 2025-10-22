import os
import logging

import matplotlib as mpl
os.environ.setdefault("MPLBACKEND", "Agg")
mpl.use("Agg", force=True)

import hbmep as mep
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy import stats
from joblib import Parallel, delayed
from hbmep import site, functional as F
from hbmep.util import generate_response_colors, make_pdf, timing
from rtmep import CheckpointManager, ObsArrays
from rtmep.model.simulate import Simulator

from rtmep.notebooks.util import enable_logger, load_model
from simulate_data import BUILD_DIR as SIMULATION_DIR
from core_job import BUILD_DIR

logger = logging.getLogger(__name__)
NUM_SAMPLES = 64
# NUM_SAMPLES = 24
FINAL_STEP = f"step_{NUM_SAMPLES:06d}.pkl"

RESPONSE = ['PKPK_APB', 'PKPK_ECR', 'PKPK_FCR']
# RESPONSE = ['PKPK_APB']
# RESPONSE = ['PKPK_ECR']
# RESPONSE = ['PKPK_FCR']

BUILD_DIR = "/home/vishu/reports/gpu/hbmep-adaptive/tests/time/"
# BUILD_DIR = "/home/vishu/reports/gpu/hbmep-adaptive/tests/time-automvn-100part-good/"
# BUILD_DIR = "/home/vishu/reports/hbmep-adaptive/accuracy/out-12Oct"

BUILD_DIR = os.path.join(BUILD_DIR, f"response_{'__'.join(RESPONSE)}")
KEYS = (site.a, site.b, site.g, site.h, site.v)

RUN_ID = "eig-mcmc"
# RUN_ID = "eig-svi"
# RUN_ID = "eig"
# RUN_ID = "focused-marginal"
# RUN_ID = "random"

LOAD_OFFLINE = False
# LOAD_OFFLINE = True

N_JOBS = -1
# N_JOBS = 1

MIN_OBS_AT_ZERO = 8


def get_paths(output_dir, run_ids, final_step=FINAL_STEP, load_offline=LOAD_OFFLINE):
    draws = list(range(200))
    paths = {draw_idx: f"draw_{draw_idx:04d}" for draw_idx in draws}
    if not load_offline:
        paths = {
            u: v for u, v in paths.items()
            if all([
                os.path.exists(os.path.join(output_dir, run_id, v, "checkpoints", final_step))
                for run_id in run_ids
            ])
        }
    else:
        paths = {
            u: v for u, v in paths.items()
            if all([
                os.path.exists(os.path.join(output_dir, run_id, v, "offline", final_step))
                for run_id in run_ids
            ])
        }

    draws = sorted(paths.keys())
    # draws = draws[:55]
    paths = {
        (draw, run_id): os.path.join(output_dir, run_id, paths[draw])
        for draw in paths
        for run_id in run_ids
    }
    return draws, paths


def load_estimate(path, num_samples=NUM_SAMPLES, load_offline=LOAD_OFFLINE):
    subdir = "checkpoints"
    ckpt_dir = os.path.join(path, subdir)
    ckpt = CheckpointManager(ckpt_dir)
    payload = ckpt.load_step(num_samples)
    posterior = payload["agent_state"]["posterior"]
    model, *_ = load_model(path)
    data = ObsArrays.from_state(payload["agent_state"]["data"])
    df = data.to_df(**model.variables)
    posterior = {u: posterior[u] for u in KEYS}
    if load_offline:
        ckpt_dir = os.path.join(path, subdir)
        ckpt = CheckpointManager(ckpt_dir)
        payload = ckpt.load_step(num_samples)
        posterior = payload["agent_state"]["posterior"]
    return df, posterior, model, posterior


def load_true(draw, response):
    sim, _, post_one, *_ = load_model(SIMULATION_DIR, memmap_draw=draw)
    response_mapping = dict(zip(sim.response, range(sim.num_response)))
    true_response_idx = [response_mapping[u] for u in response]
    post_one = {u: post_one[u].reshape(-1,)[true_response_idx] for u in KEYS}
    return post_one


@timing
def figure(output_dir, run_ids, samples_space, n_jobs=N_JOBS, load_offline=LOAD_OFFLINE):
    labels_dict = {
        "eig": "EIG", "random": "Uniform", "focused": "EIGFocused",
        "eig-svi": "EIG-SVI"
    }
    enable_logger(os.path.join(output_dir, "figure.log"))
    draws, paths = get_paths(output_dir, run_ids)

    compare_params = [site.a, site.b, site.h]
    # compare_params = [site.a, site.b]
    # compare_params = [site.a]

    def body_load_true_one(named_param, draw):
        post_one = load_true(draw, response=RESPONSE)
        return post_one[named_param]

    def body_load_pred_one(named_param, draw, run_id, num_samples, load_offline=LOAD_OFFLINE):
        path = paths[(draw, run_id)]
        ckpt = CheckpointManager(os.path.join(path, "checkpoints"))
        if load_offline:
            ckpt = CheckpointManager(os.path.join(path, "offline"))
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
                for si, num_samples in enumerate(samples_space):
                    pred_tasks.append((pi, di, ri, si, named_param, draw, run_id, num_samples))

    true_results = Parallel(n_jobs=n_jobs)(
        delayed(body_load_true_one)(named_param, draw)
        for (_, _, named_param, draw) in true_tasks
    )
    pred_results = Parallel(n_jobs=n_jobs)(
        delayed(body_load_pred_one)(named_param, draw, run_id, num_samples)
        for (_, _, _, _, named_param, draw, run_id, num_samples) in pred_tasks
    )

    true = np.array(true_results)
    true = true.reshape(len(compare_params), len(draws), 1, 1, *true.shape[1:])

    pred = np.array(pred_results)
    pred = pred.reshape(len(compare_params), len(draws), len(run_ids), len(samples_space), *pred.shape[1:])
    pred_point = np.median(pred, axis=-2)
    assert true.shape[-1] == pred_point.shape[-1]

    draws_dim = 1
    err = np.abs(pred_point - true)
    # err = (pred_point - true) ** 2
    num_draws = err.shape[draws_dim]
    err_mean = np.mean(err, axis=draws_dim) 
    # err_mean = np.median(err, axis=draws_dim) 
    err_err = stats.sem(err, axis=draws_dim)

    num_response = len(RESPONSE)
    nr, nc = len(compare_params), num_response
    fig, axes = plt.subplots(
        *(nr, nc), figsize=(3 * nc, 3 * nr), constrained_layout=True,
        squeeze=False, sharex=True, sharey="row"
    )

    for i, named_param in enumerate(compare_params):
        for response_idx, response in enumerate(RESPONSE):
            ax = axes[i, response_idx]
            x = samples_space
            for run_idx_idx, run_id in enumerate(run_ids):
                yme = err_mean[i, run_idx_idx, ..., response_idx]
                yerr = err_err[i, run_idx_idx, ..., response_idx]
                ax.errorbar(
                    x=x,
                    y=yme,
                    yerr=yerr,
                    linestyle="--",
                    marker="o",
                    ms=4,
                    label=labels_dict[run_id]
                )
        
    ax = axes[0, 0]
    ax.set_xticks(samples_space)
    ax.legend()

    for i in range(nr):
        ymax = float('-inf')
        for j in range(nc):
            ax = axes[i, j]
            cmin, cmax = ax.get_ylim()
            ymax = max(ymax, cmax)
        named_param = compare_params[i]
        if named_param == site.a:
            yticks = np.arange(0, int(ymax) + 1, 2)
        else:
            yticks = np.arange(0, int(ymax) + 1, .2)
        yticks = sorted(list(set(yticks)))
        # axes[i, 0].set_yticks(yticks)

    for i in range(nr):
        for j in range(nc):
            ax = axes[i, j]
            ax.spines[["top", "right"]].set_visible(False)
            ax.axhline(
                y = 1, color="r", linestyle="--", label="y = 1",
                linewidth=1
            )
            if not i:
                ax.axhline(
                    y = .5, color="r", linestyle="-", label="y = .5",
                    linewidth=1
                )
            if not j:
                ax.set_ylabel(f"Mean abs. error. param = {compare_params[i]}")
            if not i:
                ax.set_title(RESPONSE[j])
            ax.tick_params(
                axis="both", labelleft=True, labelsize=7,
                labelbottom=True
            )
        axes[i, 0].set_ylim(bottom=0)
        axes[i, 0].set_xlim(left=1)
    
    xticklabels = [u - MIN_OBS_AT_ZERO for u in samples_space]
    axes[0, 0].set_xticklabels(xticklabels)

    fig.align_xlabels()
    fig.align_ylabels()
    fig.suptitle(f"Total draws = {num_draws}")

    output_path = os.path.join(output_dir, "figure.pdf")
    make_pdf([fig], output_path)
    return


@timing
def plot_estimates(output_dir, run_id, n_jobs=N_JOBS):
    enable_logger(os.path.join(output_dir, run_id, "plot_analyze.log"))
    draws, paths = get_paths(output_dir, [run_id])

    model, *_ = load_model(paths[(draws[0], run_id)])
    num_response = model.num_response
    colors = generate_response_colors(num_response)

    def body_make(draws):
        print(f"Processing draws [{draws[0]}, {draws[-1]}]...")
        xgrid = np.arange(0, 100, .5)[:, None]
        nr, nc = len(draws), num_response
        kr, kc = 2, 1
        height_ratios = [1, .5] * nr
        fig, axes = plt.subplots(
            *(nr * kr, nc * kc), figsize=(2.8 * nc, nr * 2), constrained_layout=True,
            squeeze=False, height_ratios=height_ratios
        )

        for i, draw in enumerate(draws):
            path = paths[(draw, run_id)]
            df, posterior, model, post_pred = load_estimate(path)
            a_pred = post_pred[site.a]
            post_one = load_true(draw, response=RESPONSE)
            mu_true = F.rectified_logistic(
                xgrid, **{u: v.reshape(1, -1) for u, v in post_one.items()}
            )
            a_true = post_one[site.a]
            prediction_df = model.make_prediction_dataset(df)
            predictive = model.predict(
                df=prediction_df, posterior=posterior, return_sites=[site.mu]
            )
            mu = predictive[site.mu].mean(axis=0)

            for j, response in enumerate(model.response):
                ax = axes[i * kr, j * kc]
                color = colors[j]
                sns.scatterplot(
                    x=df[model.intensity], y=df[response], ax=ax, color=color,
                    s=20
                )
                sns.lineplot(
                    x=prediction_df[model.intensity], y=mu[..., j], ax=ax,
                    color=color, zorder=500
                )
                sns.lineplot(
                    x=xgrid[..., 0], y=mu_true[..., j], ax=ax, color="k",
                    linestyle="--", zorder=1000
                )
                if not j:
                    ax.set_ylabel(f"draw:{draw}")
                else:
                    ax.set_ylabel("")
                if not i:
                    ax.set_title(response)
                ax.set_xlabel("")
                samples = a_pred[..., j]
                ax.axvline(x=samples.mean(), color=color, linestyle="--")
                ax.axvline(x=a_true[..., j], color="k", linestyle="-")
                ax.sharex(axes[0, 0])
                ax = axes[i * kr + 1, j * kc]
                sns.kdeplot(samples, ax=ax, color=color)
                ax.axvline(x=samples.mean(), color=color, linestyle="--")
                ax.axvline(x=a_true[..., j], color="k", linestyle="-")
                ax.set_xlabel("")
                ax.set_ylabel("")

        for i in range(nr * kr):
            for j in range(nc * kc):
                ax = axes[i, j]
                ax.spines[['right', 'top']].set_visible(False)
                ax.tick_params(axis="both", labelsize=6)

        axes[0, 0].set_xlim(0, 100)
        return fig
    
    num_rows = 10
    pages = [draws[i:i + num_rows] for i in range(0, len(draws), num_rows)]
    figures = Parallel(n_jobs=n_jobs)(
        delayed(body_make)(page) for page in pages
    )
    output_path = os.path.join(output_dir, run_id, "curves.pdf")
    make_pdf(figures, output_path)
    return


def plot_iterations(output_dir, run_id):
    enable_logger(os.path.join(output_dir, run_id, "plot_iterations.log"))
    draws, paths = get_paths(output_dir, [run_id])

    model, *_ = load_model(paths[(draws[0], run_id)])
    num_response = model.num_response
    colors = generate_response_colors(num_response)
    xgrid = np.arange(0, 100, .5)[:, None, None]

    def body_make(draw):
        nr, nc = 6, 10
        kr, kc = 2, 1
        height_ratios = [1, .5] * nr
        fig, axes = plt.subplots(
            *(nr * kr, nc * kc), figsize=(2 * nc, nr * 2), constrained_layout=True,
            squeeze=False, height_ratios=height_ratios, sharex=True
        )

        i, j = 0, 0
        for step in range(nr * nc):
            path = paths[(draw, run_id)]
            ckpt_dir = os.path.join(path, "checkpoints")
            ckpt = CheckpointManager(ckpt_dir)
            num_samples = step + 3
            payload = ckpt.load_step(num_samples)
            posterior = payload["agent_state"]["posterior"]
            a_pred = posterior[site.a]
            mu = F.rectified_logistic(xgrid, **{u: posterior[u][None] for u in KEYS})
            mu = np.array(mu).mean(axis=1)
            post_one = load_true(draw, response=RESPONSE)
            mu_true = F.rectified_logistic(
                xgrid, **{u: v.reshape(1, -1) for u, v in post_one.items()}
            )
            a_true = post_one[site.a]
            x = payload["agent_state"]["data"]["_X"]
            y = payload["agent_state"]["data"]["_Y"]
            if num_samples == 3:
                print(x)

            for response_idx, response in enumerate(model.response):
                ax = axes[i * kr, j]
                color = colors[response_idx]
                sns.scatterplot(x=x[..., 0], y=y[..., response_idx], ax=ax, color=color, s=20)
                if num_samples > 3:
                    sns.scatterplot(
                        x=x[-1:, 0], y=y[-1:, response_idx], ax=ax,
                        color="cyan", s=20, zorder=1000
                    )
                sns.lineplot(x=xgrid[..., 0, 0], y=mu[..., response_idx], ax=ax, color=color)
                sns.lineplot(
                    x=xgrid[..., 0, 0], y=mu_true[..., 0, response_idx], ax=ax, color="k",
                    linestyle="--"
                )
                ax.text(
                    0, .98, f"#{num_samples}", fontsize=8,
                    va="top", ha="left", transform=ax.transAxes
                )
                # if not j:
                #     ax.set_ylabel(f"draw:{draw}")
                # else:
                #     ax.set_ylabel("")
                # if not i:
                #     ax.set_title(response)
                # ax.set_xlabel("")
                samples = a_pred[..., response_idx]
                # ax.axvline(x=samples.mean(), color=color, linestyle="--")
                # ax.axvline(x=a_true[..., j], color="k", linestyle="-")
                # ax.sharex(axes[0, 0])
                ax = axes[i * kr + 1, j]
                sns.kdeplot(samples, ax=ax, color=color)
                ax.axvline(x=samples.mean(), color=color, linestyle="--")
                ax.axvline(x=a_true[..., response_idx], color="k", linestyle="-")
                ax.set_xlabel("")
                ax.set_ylabel("")
            
            j += 1
            j %= nc
            if not j:
                i += 1

        for i in range(nr * kr):
            for j in range(nc * kc):
                ax = axes[i, j]
                ax.spines[['right', 'top']].set_visible(False)
                ax.tick_params(axis="both", labelsize=6)
                ax.tick_params(axis="x", labelbottom=True)
        
        # for i in range(nr):
        #     ax = axes[i * kr, 0]
        #     ax.set_xlim(0, 100)

        axes[0, 0].set_xlim(0, 100)
        fig.suptitle(f"\nDraw {draw + 1}", fontsize=14)
        return fig
    
    # draws = draws[:5]
    num_pages = len(draws)
    figures = []
    page = 0
    for draw in draws:
        print(f"Page {page + 1}/{num_pages}...")
        fig = body_make(draw=draw)
        figures.append(fig)
        page += 1

    output_path = os.path.join(output_dir, run_id, "iterations.pdf")
    make_pdf(figures, output_path)
    return


def main():
    output_dir = BUILD_DIR
    run_id = RUN_ID

    plot_estimates(output_dir, run_id)
    # # plot_iterations(output_dir, run_id)
    # # error_analysis(output_dir, run_id)

    # # run_ids = ["eig", "random"]
    # # run_ids = ["eig", "random", "focused"]
    # # run_ids = ["eig", "focused"]
    # run_ids = ["eig-svi"]
    # samples_space = [3, 5, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64]
    # samples_space = [9, 10, 11, 12, 16, 20, 24, 32, 40, 48, 56, 64]
    # figure(output_dir, run_ids, samples_space)


if __name__ == "__main__":
    main()
