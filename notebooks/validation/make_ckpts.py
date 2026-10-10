import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""     # comment out to use GPU
import traceback
import pickle

import hbmep as mep
import onmep as one
import numpy as np
from jax import random
from numpyro.diagnostics import hpdi

import pandas as pd

from paper.agent import EIG
from paper.default_kw import (
    MCMC_kw,
    NUTS_kw,
    BaseAgent_kw,
    EIG_kw,
)

from constants import DATA_DIR, CKPT_DIR, NUM_POINTS, SUBJECTS

SESSIONS = SUBJECTS

N0 = 8
N1 = 3
N = 40

BS = 6


def run(session, step):
    variables = dict(
        intensity="intensity_used",
        features=["feature_used"],
        response=["cAPB"],
    )
    mcmc_kw = MCMC_kw(num_warmup=4000, num_samples=12000, thinning=4).model_dump()
    nuts_kw = NUTS_kw(target_accept_prob=.95, max_tree_depth=(12, 12)).model_dump()
    config = dict(variables=variables, mcmc_kw=mcmc_kw, nuts_kw=nuts_kw)

    src = os.path.join(DATA_DIR, session, "data.csv")
    df = pd.read_csv(src)
    df = df.iloc[:step]
    assert df.shape[0] == step

    model_dir = os.path.join(CKPT_DIR, session)
    ckpt_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
        
    ppd_dir = os.path.join(model_dir, "ppd")
    os.makedirs(ppd_dir, exist_ok=True)

    key = random.key(0)
    key = random.fold_in(key, step)
    batch_size = 11
    base_kw = BaseAgent_kw().model_dump()
    common_kw = dict(
        key=key,
        config=config,
        **base_kw,
    )
    model_cls = EIG
    agent_kw = EIG_kw(batch_size=batch_size).model_dump()

    df = df.iloc[:step]

    model = model_cls(**common_kw, **agent_kw)
    model._model = model.rl_gamma
    model.use_mixture = True

    model.data = one.ObsArrays.from_df(df, **model.variables)
    model._update_posterior()

    ckpt = one.CheckpointManager(ckpt_dir)
    agent_state = model.state_dict()
    ckpt.save(step=step, agent_state=agent_state, env_state=None)

    x = None
    mu = None
    obs_lo = None
    obs_hi = None

    posterior = model.posterior.as_dict()

    if step == 48:
        out = os.path.join(model_dir, f"curves_{step:03d}.pdf")
        model._plot_curves(output_path=out)

        pred_df = model.make_prediction_dataset(df, num_points=NUM_POINTS, min_intensity=0, max_intensity=100)
        ppd = model.predict(
            pred_df,
            posterior=posterior,
            num_features=[1],
            num_response=1,
        )

        x = pred_df["intensity_used"].to_numpy()
        mu = ppd[mep.site.mu].mean(axis=0)[..., 0]
        obs_hdi = hpdi(
            ppd[mep.site.obs],
            axis=0,
            prob=0.95,
        )
        obs_lo = obs_hdi[0][..., 0]
        obs_hi = obs_hdi[1][..., 0]

        x = np.asarray(x)
        mu = np.asarray(mu)
        obs_lo = np.asarray(obs_lo)
        obs_hi = np.asarray(obs_hi)

    threshold = np.asarray(posterior[mep.site.a].reshape(-1,))
    saturation = np.asarray(posterior[mep.site.g] + posterior[mep.site.h]).reshape(-1,)

    result = {
        "x": x,
        "mu": mu,
        "obs_lo": obs_lo,
        "obs_hi": obs_hi,
        "threshold_samples": threshold,
        "saturation_samples": saturation,
    }

    out = os.path.join(ppd_dir, f"step_{step:03d}.pkl")
    with open(out, "wb") as f:
        pickle.dump(result, f)

    print(f"{session}, {step} ok.")
    return


def safe_run(session, step):
    try:
        run(session, step)
    except Exception:
        out = os.path.join(CKPT_DIR, f"failed_{session}_{step:03d}.txt")
        with open(out, "w") as f:
            traceback.print_exc(file=f)


def main():
    steps = [i for i in range(N1 + N0, N + N0 + 1)]
    tasks = [(session, step) for session in SESSIONS for step in steps]
    safe_run = run
    mep.run_batched(safe_run, tasks=tasks, batch_size=BS)
    return


if __name__ == "__main__":
    print(CKPT_DIR)
    main()
