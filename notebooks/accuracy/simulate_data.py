import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import logging

import matplotlib as mpl
os.environ.setdefault("MPLBACKEND", "Agg")
mpl.use("Agg", force=True)

import hbmep as mep
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from jax import random
from numpyro import distributions as dist
from numpyro.diagnostics import hpdi
from rtmep.model.simulate import Simulator

from rtmep.notebooks.util import load_model, save_model
from rtmep.notebooks.constants import TMS_TOML
from learn_posterior import BUILD_DIR as LEARN_POSTERIOR_DIR

BUILD_DIR = os.path.join(LEARN_POSTERIOR_DIR, "simulate_data")

logger = logging.getLogger(__name__)
TOTAL_SUBJECTS = 1
TOTAL_PULSES = 101
TOTAL_REPS = 8

CHECK_POWER = False
# CHECK_POWER = True


def sim_accuracy_data(build_dir):
    model = Simulator(toml_path=TMS_TOML)
    model.simulate = True
    model.use_mixture = False
    model._model = model.hb_rl
    model.build_dir = build_dir
    mep.enable_logging(model.build_dir)

    # Create template dataframe for simulation
    df = pd.DataFrame(
        np.arange(0, TOTAL_SUBJECTS, 1),
        columns=model.features
    )
    df[model.intensity] = 0
    df = model.make_prediction_dataset(
        df=df,
        min_intensity=0,
        max_intensity=100,
        num_points=TOTAL_PULSES
    )
    df = (
        pd.concat([df] * TOTAL_REPS, ignore_index=True)
        .reset_index(drop=True)
        .copy()
    )

    arr = []
    for i in range(0, TOTAL_REPS):
        arr += [i] * TOTAL_SUBJECTS * TOTAL_PULSES
    df["rep"] = arr
    logger.info(f"Simulation dataframe: {df.shape}")

    # Load learnt posterior
    model_dir = LEARN_POSTERIOR_DIR
    _, _, posterior, *_ = load_model(model_dir)

    logger.info("Learn posterior shapes:")
    for u, v in posterior.items():
        logger.info(f"{u}: {v.shape}")
 
    # Exclude priors
    keep_sites = [
        mep.site.a.loc, mep.site.a.scale,
        mep.site.b.loc, mep.site.b.scale,
        mep.site.h.loc, mep.site.h.scale,
        mep.site.g.scale,
        mep.site.v.scale,
        mep.site.c1.scale,
        mep.site.c2.scale,
    ]
    posterior = {
        u: v for u, v in posterior.items()
        if u in keep_sites
    }
    remaining_sites = sorted(list(posterior.keys()))
    logger.info(f"Remaining sites: {remaining_sites}")
    for u, v in posterior.items():
        logger.info(f"{u}: {v.shape}")

    # Simulate
    logger.info(f"Simulating new subjects ...")
    ppd = model.predict(df=df, posterior=posterior)
    logger.info(f"simulation_ppd: {sorted(list(ppd.keys()))}")
    for u, v in ppd.items():
        logger.info(f"{u}: {v.shape}")

    named_params = [mep.site.a, mep.site.b, mep.site.g, mep.site.h, mep.site.v]
    ylo = mep.functional.rectified_logistic(
        x=5, **{u: v for u, v in ppd.items() if u in named_params}
    )
    yhi = mep.functional.rectified_logistic(
        x=95, **{u: v for u, v in ppd.items() if u in named_params}
    )
    y = yhi - ylo
    y = y[:, 0]

    # Filter draws
    a = ppd[mep.site.a]
    a = a[:, 0]
    idx = (
        np.all(y > .15, axis=(-1))
        & np.all((a > 10) & (a < 70), axis=(-1))
    )
    ppd = {u: v[idx, ...] for u, v in ppd.items()}
    logger.info(f"Num draws left: {ppd[mep.site.a].shape[0]}")
 
    # Shuffle draws
    logger.info(f"Shuffling draws ...")
    idx = np.arange(0, ppd[mep.site.a].shape[0], 1)
    _, sub = random.split(model.key)
    idx = np.array(random.permutation(sub, idx))
    ppd = {u: v[idx, ...] for u, v in ppd.items()}
    ppd = {u: v[:5000, ...] for u, v in ppd.items()}

    save_model(model, df, ppd)
    return


def sim_power_data(build_dir, a_delta_loc, a_delta_scale):
    model = Simulator(toml_path=TMS_TOML)
    model.features = ["participant", "intervention"]
    model.simulate = True
    model.use_mixture = False
    model._model = model.hb_rl
    model.build_dir = build_dir
    mep.enable_logging(model.build_dir)

    # Create template dataframe for simulation
    df = (
        pd.DataFrame(
            np.arange(32),
            columns=[model.features[0]]
        )
        .merge(
            pd.DataFrame(
                np.arange(2),
                columns=[model.features[1]]
            ),
            how="cross"
        )
    )
    df[model.intensity] = 0
    df = model.make_prediction_dataset(
        df=df,
        min_intensity=0,
        max_intensity=100,
        num_points=16
    )
    logger.info(f"Simulation dataframe: {df.shape}")

    # Load learnt posterior
    model_dir = LEARN_POSTERIOR_DIR
    _, _, posterior, *_ = load_model(model_dir)

    logger.info("Learn posterior shapes:")
    for u, v in posterior.items():
        logger.info(f"{u}: {v.shape}")
 
    # Exclude priors
    keep_sites = [
        mep.site.a.loc, mep.site.a.scale,
        mep.site.b.loc, mep.site.b.scale,
        mep.site.h.loc, mep.site.h.scale,
        mep.site.g.scale,
        mep.site.v.scale,
        mep.site.c1.scale,
        mep.site.c2.scale,
    ]
    posterior = {
        u: v for u, v in posterior.items()
        if u in keep_sites
    }
    remaining_sites = sorted(list(posterior.keys()))
    logger.info(f"Remaining sites: {remaining_sites}")
    for u, v in posterior.items():
        logger.info(f"{u}: {v.shape}")

    # Simulate
    logger.info(f"Simulating new subjects ...")
    ppd = model.predict(df=df, posterior=posterior)
    logger.info(f"simulation_ppd: {sorted(list(ppd.keys()))}")
    for u, v in ppd.items():
        logger.info(f"{u}: {v.shape}")
 
    a = ppd[mep.site.a]
    _, sub = random.split(model.key)
    a_delta = np.array((
        dist.Normal(loc=a_delta_loc, scale=a_delta_scale)
        .sample(sub, a.shape)
    ))

    a[..., 1, :] = a[..., 0, :] + a_delta[..., 1, :]
    np.testing.assert_almost_equal(
        a[..., 1, :] - a[..., 0, :],
        a_delta[..., 1, :]
    )
    ppd[mep.site.a] = a

    named_params = [mep.site.a, mep.site.b, mep.site.g, mep.site.h, mep.site.v]
    ylo = mep.functional.rectified_logistic(
        x=5, **{u: v for u, v in ppd.items() if u in named_params}
    )
    yhi = mep.functional.rectified_logistic(
        x=95, **{u: v for u, v in ppd.items() if u in named_params}
    )
    y = yhi - ylo
    logger.info(f"y: {y.shape}")

    # Filter draws
    a = ppd[mep.site.a]
    idx = (
        np.all(y > 0.15, axis=(1, 2))[..., 0]
        & np.all((a > 5) & (a < 70), axis=(1, 2, 3))
    )
    ppd = {u: v[idx, ...] for u, v in ppd.items()}
    logger.info(f"Num draws left: {ppd[mep.site.a].shape[0]}")

    if CHECK_POWER:
        from scipy import stats
        import matplotlib.pyplot as plt
        import seaborn as sns
        SIGNIFICANCE_LEVEL = 0.05
        true = ppd[mep.site.a]
        reject = []
        num_subjects_space = list(range(2, true.shape[1], 2))
        for si, num_subjects in enumerate(num_subjects_space):
            x = true[:, :num_subjects]
            diff = x[..., 0, :] - x[..., 1, :]
            pr = stats.wilcoxon(x=diff, alternative="two-sided", axis=-2).pvalue
            decision = pr < SIGNIFICANCE_LEVEL
            reject.append(decision)
        reject = np.array(reject)
        reject = reject.reshape(len(num_subjects_space), *reject.shape[1:])
        reject = reject.swapaxes(0, 1)
        reject_mean = np.mean(reject, axis=0)
        reject_err = stats.sem(reject, axis=0)

        num_response = reject.shape[-1]
        nr, nc = 1, num_response
        plt.close("all")
        fig, axes = plt.subplots(
            nr, nc,
            figsize=(5.0 * nc, 3.4 * nr),
            constrained_layout=True,
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        fig.show()
        x = num_subjects_space
        for ri in range(num_response):
            ax = axes[0, ri]
            ax.clear()
            yme = reject_mean[..., ri]
            yerr = reject_err[..., ri]
            ax.errorbar(
                x=x,
                y=yme,
                yerr=yerr,
                linestyle="--",
                marker="o",
                ms=4,
            )
            ax.set_xticks(num_subjects_space)
            ax.tick_params(axis="x", rotation=15)
            ax.grid(True, alpha=.4)
            ax.axhline(.8, color="r", linestyle="--")
            ax.set_ylim(top=1.1)
            ax.set_xticks(x)
            if "tp" in build_dir:
                ax.set_ylim(top=1.1)
                ax.axhline(.8, color="r", linestyle="--")
            else:
                ax.set_ylim(top=0.11)
                ax.axhline(SIGNIFICANCE_LEVEL, color="r", linestyle="--")
 
    # Shuffle draws
    logger.info(f"Shuffling draws ...")
    idx = np.arange(0, ppd[mep.site.a].shape[0], 1)
    _, sub = random.split(model.key)
    idx = np.array(random.permutation(sub, idx))
    ppd = {u: v[idx, ...] for u, v in ppd.items()}
    ppd = {u: v[:5000, ...] for u, v in ppd.items()}

    ppd.pop(mep.site.mu, None)
    ppd.pop(mep.site.obs, None)
    save_model(model, df, ppd)
    return


def main():
    # sim_accuracy_data(BUILD_DIR)

    # a_delta_loc, a_delta_scale = 0., 2.5
    # build_dir = os.path.join(LEARN_POSTERIOR_DIR, "simulate_power_data_fp")

    a_delta_loc, a_delta_scale = -3.5, 2.5
    build_dir = os.path.join(LEARN_POSTERIOR_DIR, "simulate_power_data_tp")

    # if CHECK_POWER:
    #     build_dir = os.path.join(build_dir, "check_power")

    os.makedirs(build_dir, exist_ok=True)
    sim_power_data(build_dir, a_delta_loc, a_delta_scale)

    return


def plot_data():
    model_dir = "/home/vishu/reports/hbmep-adaptive/accuracy/learn_posterior/simulate_power_data_tp"
    model, df, posterior, encoder, mcmc = load_model(model_dir)
    posterior.keys()
    named_params = [mep.site.a, mep.site.b, mep.site.g, mep.site.h, mep.site.v]

    for u in named_params:
        v = posterior[u]
        print(u, v.shape)
    
    posterior[mep.site.c1].mean()
    posterior[mep.site.c2].mean()
    # posterior[mep.site.v][..., 0, :] /= 10
    posterior[mep.site.c2][..., 0, :] *= 10

    num_rows = 10
    xgrid = np.linspace(0, 100, 200)
    pidx = 0

    def body_make(draws):
        nr, nc = num_rows, 2
        figsize = (8, 11)
        fig, axes = plt.subplots(
            *(nr, nc), figsize=figsize, constrained_layout=True
        )
        for di, draw in enumerate(draws):
            for j in range(2):
                ax = axes[di, j]
                ax.clear()
                params = [posterior[u][draw, pidx, j, 0] for u in named_params]
                y = mep.functional.rectified_logistic(xgrid, **dict(zip(named_params, params)))
                y = np.array(y)
                sns.lineplot(x=xgrid, y=y, ax=ax, linewidth=.5)
                if not di and not j:
                    ax.set_title(f"j = {j}")
                ax.axvline(params[0], linestyle="--", linewidth=.5)

                c1 = posterior[mep.site.c1][draw, pidx, j, 0]
                c2 = posterior[mep.site.c2][draw, pidx, j, 0]
                alpha, beta = model.gamma_likelihood(y, c1, c2)
                num_samples = 2000
                key = random.key(draw + 399)
                key, sub = random.split(key)
                gamma_samples = np.array(
                    dist.Gamma(concentration=alpha, rate=beta)
                    .sample(sub, (num_samples,))
                )
                hdi = hpdi(gamma_samples, prob=0.95, axis=0)
                lower = hdi[0]
                upper = hdi[1]
                ax.fill_between(
                    xgrid,
                    lower,
                    upper,
                    alpha=0.25
                )
                ax.set_ylim(top=.5)
                ax.set_title((draw, j))

        di += 1
        while di < num_rows:
            for j in range(2):
                ax = axes[di, j]
                ax.set_axis_off()
            di += 1

        return fig

    figs = []
    num_draws = 20
    for i in range(0, num_draws, num_rows):
        draws = list(range(i, i + num_rows))
        fig = body_make(draws)
        figs.append(fig)
    
    out = os.path.join(model_dir, "c2_high.pdf")
    mep.make_pdf(figs, out)
    return


if __name__ == "__main__":
    main()
    # plot_data()
