import os
import logging

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from hbmep import site, functional as F
from hbmep.util import generate_response_colors, make_pdf
from rtmep import CheckpointManager
from rtmep.model.simulate import Simulator

from rtmep.notebooks.util import enable_logger, load_model
from simulate_data import BUILD_DIR as SIMULATION_DIR
from core_job import BUILD_DIR

logger = logging.getLogger(__name__)

RESPONSE = ['PKPK_APB', 'PKPK_ECR', 'PKPK_FCR']
RESPONSE = ['PKPK_APB']
# RESPONSE = ['PKPK_ECR']
BUILD_DIR = os.path.join(BUILD_DIR, f"response_{'__'.join(RESPONSE)}")

KEYS = (site.a, site.b, site.g, site.h, site.v)

RUN_ID = "eig"
# RUN_ID = "random"


def get_paths(output_dir, run_ids, final_step="step_000064.pkl"):
    draws = list(range(1000))
    paths = {draw_idx: f"draw_{draw_idx:04d}" for draw_idx in draws}
    paths = {
        u: v for u, v in paths.items()
        if all([
            os.path.exists(os.path.join(output_dir, run_id, v, "checkpoints", final_step))
            for run_id in run_ids
        ])
    }
    draws = sorted(paths.keys())
    # draws = draws[:5]
    paths = {
        (draw, run_id): os.path.join(output_dir, run_id, paths[draw])
        for draw in paths
        for run_id in run_ids
    }
    return draws, paths


def load_estimate(output_dir):
    model, df, posterior, _, _ = load_model(output_dir)
    posterior = {u: posterior[u] for u in KEYS}
    return df, posterior, model, posterior


def load_true(draw):
    _, _, post_one, *_ = load_model(SIMULATION_DIR, memmap_draw=draw)
    post_one = {u: post_one[u].reshape(-1,) for u in KEYS}
    return post_one


def figure(output_dir, run_ids, samples_space):
    sim, *_ = load_model(SIMULATION_DIR)
    labels_dict = {"eig": "EIG", "random": "Uniform"}
    enable_logger(os.path.join(output_dir, "figure.log"))
    draws, paths = get_paths(output_dir, run_ids)

    compare_params = [site.a, site.b, site.h]
    # compare_params = [site.a, site.b]
    # compare_params = [site.a]

    true, pred = [], []
    for named_param in compare_params:
        for run_id in run_ids:
            for draw in draws:
                path = paths[(draw, run_id)]
                ckpt_dir = os.path.join(path, "checkpoints")
                ckpt = CheckpointManager(ckpt_dir)
                post_one = load_true(draw)
                a_true = post_one[named_param]
                true.append(a_true)
                for num_samples in samples_space:
                    payload = ckpt.load_step(num_samples)
                    posterior = payload["agent_state"]["posterior"]
                    a_pred = posterior[named_param]
                    pred.append(a_pred)
    
    true = np.array(true)
    response_mapping = dict(zip(sim.response, range(sim.num_response)))
    true_response_idx = [response_mapping[u] for u in RESPONSE]
    true = true[..., true_response_idx]
    true = true.reshape(len(compare_params), len(run_ids), len(draws), 1, *true.shape[1:])
    pred = np.array(pred)
    pred = pred.reshape(len(compare_params), len(run_ids), len(draws), len(samples_space), *pred.shape[1:])
    pred_point = np.mean(pred, axis=-2)

    draws_dim = 2
    err = np.abs(pred_point - true)
    num_draws = err.shape[draws_dim]
    err_mean = np.mean(err, axis=draws_dim) 
    # err_mean = np.median(err, axis=draws_dim) 
    err_err = stats.sem(err, axis=draws_dim)
    # err_err = np.std(err, axis=draws_dim)

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
        axes[i, 0].set_yticks(yticks)

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

    fig.align_xlabels()
    fig.align_ylabels()
    fig.suptitle(f"Total draws = {num_draws}")

    output_path = os.path.join(output_dir, "figure.pdf")
    make_pdf([fig], output_path)
    return


def error_analysis(output_dir, run_id):
    run_dir = os.path.join(output_dir, run_id)
    enable_logger(os.path.join(run_dir, "error_analysis.log"))
    draws, paths = get_paths(run_dir)

    true, pred = [], []
    for draw in draws:
        path = paths[(draw, run_id)]
        *_, post_pred = load_estimate(path)
        a_pred = post_pred[site.a]
        post_one = load_true(draw)
        a_true = post_one[site.a]
        true.append(a_true)
        pred.append(a_pred)
    
    true = np.array(true)
    pred = np.array(pred)

    pred_point = np.mean(pred, axis=1)
    err = np.abs(pred_point - true)
    num_draws = err.shape[0]
    err_mean = err.mean(axis=0)
    err_std = err.std(axis=0)
    print(f"{num_draws}:", run_id, err_mean, err_std)
    return
    

def plot_estimates(output_dir, run_id):
    enable_logger(os.path.join(output_dir, run_id, "plot_analyze.log"))
    draws, paths = get_paths(output_dir, [run_id])

    model, *_ = load_model(paths[(draws[0], run_id)])
    num_response = model.num_response
    colors = generate_response_colors(num_response)

    def body_make(draws):
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
            post_one = load_true(draw)
            xgrid = np.arange(0, 100, .5)[:, None]
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
    num_pages = len(draws) // num_rows
    if len(draws) % num_rows:
        num_pages += 1
    figures = []
    for page in range(num_pages):
        print(f"Page {page + 1}/{num_pages}...")
        fig = body_make(draws=draws[page * num_rows: (page + 1) * num_rows])
        figures.append(fig)

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
            post_one = load_true(draw)
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

    # plot_estimates(output_dir, run_id)
    # plot_iterations(output_dir, run_id)
    # error_analysis(output_dir, run_id)

    run_ids = ["eig", "random"]
    run_ids = ["eig"]
    samples_space = [3, 5, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64]
    figure(output_dir, run_ids, samples_space)


if __name__ == "__main__":
    main()
