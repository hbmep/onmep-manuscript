import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import logging

import matplotlib as mpl
os.environ.setdefault("MPLBACKEND", "Agg")
mpl.use("Agg", force=True)

import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd

import hbmep as mep
import numpy as np
from jax import random
from numpyro import distributions as dist

from paper.model.simulate import Simulator

from constants import LEARN_POSTERIOR_DIR, SIMULATIONS_DATA_DIR

logger = logging.getLogger(__name__)


def sim_accuracy_data(simulations_model_name):
    model_dir = os.path.join(LEARN_POSTERIOR_DIR, simulations_model_name)
    model_state, posterior, df, encoder, mcmc = mep.load(model_dir)
    model = Simulator()
    model.load_state_dict(model_state)

    assert model._model.__name__ == simulations_model_name
    assert not model.use_mixture
    assert not model.simulate
    model.simulate = True

    model.build_dir = os.path.join(SIMULATIONS_DATA_DIR, simulations_model_name, "accuracy_data")
    mep.enable_logging(model.build_dir)

    logger.info("Learn posterior shapes:")
    for u, v in posterior.items():
        logger.info(f"{u}: {v.shape}")

    # Create template dataframe for simulation
    total_reps = 8
    total_pulses = 101
    total_subjects = 1
    df = pd.DataFrame(
        np.arange(0, total_subjects, 1),
        columns=model.features
    )
    df[model.intensity] = 0
    df = model.make_prediction_dataset(
        df=df,
        min_intensity=0,
        max_intensity=100,
        num_points=total_pulses
    )
    df = (
        pd.concat([df] * total_reps, ignore_index=True)
        .reset_index(drop=True)
        .copy()
    )
    arr = []
    for i in range(0, total_reps):
        arr += [i] * total_subjects * total_pulses
    df["rep"] = arr
    logger.info(f"Simulation dataframe: {df.shape}")

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

    # Make thresholds higher for non-target muscles
    posterior[mep.site.a.loc][..., 1] += 15
    posterior[mep.site.a.loc][..., 2] += 30
    print(posterior[mep.site.a.loc].mean(axis=0).reshape(-1,))

    # Simulate
    logger.info(f"Simulating new subjects ...")
    ppd = model.predict(df=df, posterior=posterior)
    logger.info(f"simulation_ppd: {sorted(list(ppd.keys()))}")
    for u, v in ppd.items():
        logger.info(f"{u}: {v.shape}")

    temp_mu = ppd.pop(mep.site.mu)
    temp_obs = ppd.pop(mep.site.obs)
    ppd.update(posterior)
    print(ppd[mep.site.a].mean(axis=0).reshape(-1,))

    # Redraw valid baseline noise floor
    g = ppd[mep.site.g]
    keep = (g[:, 0, 0] > 0.01) & (g[:, 0, 0] < 0.025)
    g_valid = g[:, 0, 0][keep]
    use_gamma = True if simulations_model_name == "rl_gamma" else False
    key = random.key(23923 + int(use_gamma))
    key, sub = random.split(key)
    g_redraw = np.array(random.choice(
        sub,
        g_valid,
        shape=(g.shape[0] * g.shape[1] * g.shape[2],),
        replace=True,
    ))
    g_redraw = g_redraw.reshape(g.shape)
    ppd[mep.site.g] = g_redraw

    # Response observed
    named_params = [mep.site.a, mep.site.b, mep.site.g, mep.site.h, mep.site.v]
    ylo = mep.functional.rectified_logistic(
        x=5, **{u: v for u, v in ppd.items() if u in named_params}
    )
    yhi = mep.functional.rectified_logistic(
        x=90, **{u: v for u, v in ppd.items() if u in named_params}
    )
    y = yhi - ylo
    y = y[:, 0]

    # Filter draws
    a = ppd[mep.site.a][:, 0]
    g = ppd[mep.site.g][:, 0]
    mep_threshold = 0.05
    t = np.array(mep.inverse.rectified_logistic(
        y=mep_threshold, **{u: ppd[u] for u in named_params}
    ))
    t = t[:, 0]

    idx = (
        # Baseline
        np.all(g > 0.01, axis=-1)
        & np.all(g < 0.025, axis=-1)
        # Threshold
        & np.all((a > 10) & (a < 70), axis=(-1))
        # RMT
        & np.all((t > 10) & (t < 70), axis=(-1))
        # Response
        & np.all(y > .15, axis=(-1))
    )

    ppd = {u: v[idx, ...] for u, v in ppd.items()}
    logger.info(f"Num draws left: {ppd[mep.site.a].shape[0]}")

    # Shuffle draws
    logger.info(f"Shuffling draws ...")
    idx = np.arange(0, ppd[mep.site.a].shape[0], 1)
    key, sub = random.split(key)
    idx = np.array(random.permutation(sub, idx))
    ppd = {u: v[idx, ...] for u, v in ppd.items()}
    ppd = {u: v[:5000, ...] for u, v in ppd.items()}

    model_state = model.state_dict()
    mep.save(
        model_state=model_state,
        posterior=ppd,
        df=df,
        encoder=encoder,
        mcmc=None,
        output_dir=model.build_dir
    )
    return


def sim_power_data(fp=False, check_power=True):
    simulations_model_name = "rl_gamma"
    model_dir = os.path.join(LEARN_POSTERIOR_DIR, simulations_model_name)
    model_state, posterior, df, encoder, mcmc = mep.load(model_dir)
    model = Simulator()
    model.load_state_dict(model_state)

    assert model._model.__name__ == simulations_model_name
    assert not model.use_mixture
    assert not model.simulate
    model.simulate = True

    model.features = ["participant", "intervention"]

    if fp:
        build_dir = os.path.join(SIMULATIONS_DATA_DIR, simulations_model_name, "power_data_fp")
        a_delta_loc, a_delta_scale = 0., 2.5
    else:
        build_dir = os.path.join(SIMULATIONS_DATA_DIR, simulations_model_name, "power_data_tp")
        a_delta_loc, a_delta_scale = -3., 2.5

    model.build_dir = build_dir
    mep.enable_logging(model.build_dir)

    logger.info("Learn posterior shapes:")
    for u, v in posterior.items():
        logger.info(f"{u}: {v.shape}")

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
        num_points=2
    )
    logger.info(f"Simulation dataframe: {df.shape}")

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
 
    ppd.pop(mep.site.mu)
    ppd.pop(mep.site.obs)
    ppd.update(posterior)

    a = ppd[mep.site.a]
    key = random.key(82382)
    key, sub = random.split(key)
    a_delta = np.asarray((
        dist.Normal(loc=a_delta_loc, scale=a_delta_scale)
        .sample(sub, a.shape)
    ))

    a[..., 1, :] = a[..., 0, :] + a_delta[..., 1, :]
    np.testing.assert_almost_equal(
        a[..., 1, :] - a[..., 0, :],
        a_delta[..., 1, :]
    )
    ppd[mep.site.a] = a

    # Redraw valid baseline noise floor
    g = ppd[mep.site.g]
    keep = (g[:, 0, 0, 0] > 0.01) & (g[:, 0, 0, 0] < 0.025)
    g_valid = g[:, 0, 0, 0][keep]
    key, sub = random.split(key)
    g_redraw = np.array(random.choice(
        sub,
        g_valid,
        shape=(g.shape[0] * g.shape[1] * g.shape[2] * g.shape[3],),
        replace=True,
    ))
    g_redraw = g_redraw.reshape(g.shape)
    g_redraw[..., 1, :] = g_redraw[..., 0, :]
    ppd[mep.site.g] = g_redraw

    # Response observed
    named_params = [mep.site.a, mep.site.b, mep.site.g, mep.site.h, mep.site.v]
    ylo = mep.functional.rectified_logistic(
        x=5, **{u: v for u, v in ppd.items() if u in named_params}
    )
    yhi = mep.functional.rectified_logistic(
        x=90, **{u: v for u, v in ppd.items() if u in named_params}
    )
    y = yhi - ylo
    y = y[..., 0]

    # Filter draws
    a = ppd[mep.site.a][..., 0]
    g = ppd[mep.site.g][..., 0]
    mep_threshold = 0.05
    t = np.array(mep.inverse.rectified_logistic(
        y=mep_threshold, **{u: ppd[u] for u in named_params}
    ))
    t = t[..., 0]

    idx = (
        # Baseline
        np.all(g > 0.01, axis=(-1, -2))
        & np.all(g < 0.025, axis=(-1, -2))
        # Threshold
        & np.all((a > 10) & (a < 70), axis=(-1, -2))
        # RMT
        & np.all((t > 10) & (t < 70), axis=(-1, -2))
        # Response
        & np.all(y > .15, axis=(-1, -2))
    )

    ppd = {u: v[idx, ...] for u, v in ppd.items()}
    logger.info(f"Num draws left: {ppd[mep.site.a].shape[0]}")

    # Shuffle draws
    logger.info(f"Shuffling draws ...")
    idx = np.arange(0, ppd[mep.site.a].shape[0], 1)
    key, sub = random.split(key)
    idx = np.array(random.permutation(sub, idx))
    ppd = {u: v[idx, ...] for u, v in ppd.items()}
    ppd = {u: v[:5000, ...] for u, v in ppd.items()}

    model_state = model.state_dict()
    mep.save(
        model_state=model_state,
        posterior=ppd,
        df=df,
        encoder=encoder,
        mcmc=None,
        output_dir=model.build_dir
    )

    if check_power:
        SIGNIFICANCE_LEVEL = 0.05
        true = ppd[mep.site.a]
        reject = []
        num_subjects_space = list(range(2, true.shape[1], 1))
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
        fig, axes = plt.subplots(
            nr, nc,
            figsize=(5.0 * nc, 3.4 * nr),
            constrained_layout=True,
            squeeze=False,
            sharex=True,
            sharey=True,
        )
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
            if fp:
                ax.set_ylim(top=0.11)
                ax.axhline(SIGNIFICANCE_LEVEL, color="r", linestyle="--")
            else:
                ax.set_ylim(top=1.1)
                ax.axhline(.8, color="r", linestyle="--")
        out = os.path.join(model.build_dir, "power.pdf")
        mep.make_pdf([fig], output_path=out)
    return


def main():
    # sim_accuracy_data(simulations_model_name="rl_gamma")
    # sim_accuracy_data(simulations_model_name="rl_lognormal")

    sim_power_data(fp=True, check_power=True)
    sim_power_data(fp=False, check_power=True)

    return


if __name__ == "__main__":
    main()
