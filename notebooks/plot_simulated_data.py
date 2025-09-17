import os
import pickle
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from jax import random
from hbmep import functional as F
from hbmep.util import site, make_pdf, generate_response_colors
from rtmep.model.simulate import Simulator

from rtmep.notebooks.util import enable_logger, load_model, save_model
from simulate_data import BUILD_DIR

logger = logging.getLogger(__name__)


def main():
    df, _, posterior, model, _, = load_model(BUILD_DIR)

    def body_make_figure(left, right):
        num_draws = right - left
        nr, nc = num_draws, model.num_response
        fig, axes = plt.subplots(
            *(nr, nc), figsize=(8, 11), constrained_layout=True,
            # sharex=False, sharey="row"
        )
        colors = generate_response_colors(model.num_response)

        for draw_idx, draw in enumerate(range(left, right)):
            for response_idx, response in enumerate(model.response):
                ax = axes[draw_idx, response_idx]
                color = colors[response_idx]
                obs = posterior[site.obs][draw, :, response_idx]
                mu = posterior[site.mu][draw, :, response_idx]
                # idx = df["rep"] == 0
                # ccdf = df[idx].reset_index(drop=True).copy()
                x = df[model.intensity].values
                sns.scatterplot(x=x, y=obs, color=color, ax=ax)
                sns.lineplot(x=x, y=mu, color=color, ax=ax)
                samples = posterior[site.a][draw, 0, response_idx]
                ax.axvline(x=samples.mean(), linestyle="--", color="k")
                samples = posterior[site.h][draw, 0, response_idx]
                ax.axhline(y=samples.mean(), linestyle="--", color="k")
                if not draw_idx:
                    ax.set_title(response)
                if not response_idx:
                    ax.set_ylabel(f"Draw: {draw}")

        return fig

    num_draws = 500
    step = 10
    figures = []
    for left in range(0, num_draws, step):
        right = left + step
        fig = body_make_figure(left, right)
        figures.append(fig)
    output_path = os.path.join(BUILD_DIR, "simulated_data.pdf")
    make_pdf(figures=figures, output_path=output_path)


if __name__ == "__main__":
    enable_logger(BUILD_DIR)
    main()
