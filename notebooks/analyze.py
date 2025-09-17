import os
import logging

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hbmep import site
from hbmep.util import generate_response_colors, make_pdf

from rtmep.notebooks.util import enable_logger, load_model
from core_parallel import BUILD_DIR

logger = logging.getLogger(__name__)

def get_paths(output_dir):
    draws = list(range(1000))
    paths = {draw_idx: f"draw_{draw_idx:04d}" for draw_idx in draws}
    paths = {u: os.path.join(output_dir, v) for u, v in paths.items()}
    paths = {u: v for u, v in paths.items() if os.path.exists(os.path.join(v, "a.npz"))}
    draws = [u for u in draws if u in paths.keys()]
    return draws, paths


def load_estimate(output_dir):
    df, _, posterior, model, _ = load_model(output_dir)
    a = np.load(os.path.join(output_dir, "a.npz"))
    a_true = a["a_true"]
    a_pred = a["a_pred"]
    return df, posterior, model, a_true, a_pred,


def error_analysis(output_dir, run_id):
    run_dir = os.path.join(output_dir, run_id)
    enable_logger(run_dir)
    draws, paths = get_paths(run_dir)

    true, pred = [], []
    for draw in draws:
        path = paths[draw]
        *_, a_true, a_pred = load_estimate(path)
        true.append(a_true)
        pred.append(a_pred)
    
    true = np.array(true)
    pred = np.array(pred)

    pred_point = np.mean(pred, axis=1, keepdims=True)
    err = np.abs(pred_point - true)
    err_mean = err.mean(axis=(0, 1))
    err_std = err.std(axis=(0, 1))
    print(run_id, err_mean, err_std)
    return
    

def plot_estimates(output_dir, run_id):
    run_dir = os.path.join(output_dir, run_id)
    enable_logger(run_dir)
    draws, paths = get_paths(run_dir)
    draws = draws[:25]

    *_, model, _ = load_model(paths[0])
    num_response = model.num_response
    colors = generate_response_colors(num_response)

    def body_make(draws):
        nr, nc = len(draws), num_response
        kr, kc = 2, 1
        height_ratios = [1, .5] * nr
        fig, axes = plt.subplots(
            *(nr * kr, nc * kc), figsize=(8, nr * 1.5), constrained_layout=True,
            squeeze=False, height_ratios=height_ratios
        )

        for i, draw in enumerate(draws):
            path = paths[draw]
            df, posterior, model, a_true, a_pred = load_estimate(path)
            prediction_df = model.make_prediction_dataset(df)
            predictive = model.predict(df=prediction_df, posterior=posterior, return_sites=[site.mu])
            mu = predictive[site.mu].mean(axis=0)

            for j, response in enumerate(model.response):
                ax = axes[i * kr, j * kc]
                color = colors[j]
                sns.scatterplot(x=df[model.intensity], y=df[response], ax=ax, color=color)
                sns.lineplot(x=prediction_df[model.intensity], y=mu[..., j], ax=ax, color="grey")
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
                ax.tick_params(axis="both", labelsize=8)

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

    output_path = os.path.join(run_dir, "curves.pdf")
    make_pdf(figures, output_path)
    return


def main():
    output_dir = BUILD_DIR
    run_id = "random"

    # plot_estimates(output_dir, run_id)
    error_analysis(output_dir, run_id)


if __name__ == "__main__":
    main()
