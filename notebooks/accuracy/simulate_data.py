import os
import logging

import pandas as pd
import numpy as np
from jax import random
from hbmep import functional as F
from hbmep.util import site
from rtmep.model.simulate import Simulator

from rtmep.notebooks.util import enable_logger, load_model, save_model
from rtmep.notebooks.constants import TMS_TOML
from learn_posterior import BUILD_DIR as LEARN_POSTERIOR_DIR

logger = logging.getLogger(__name__)
BUILD_DIR = os.path.join(LEARN_POSTERIOR_DIR, "simulate_data")
os.makedirs(BUILD_DIR, exist_ok=True)

TOTAL_SUBJECTS = 1
TOTAL_PULSES = 101
TOTAL_REPS = 8


def main():
    model = Simulator(toml_path=TMS_TOML)
    model.simulate = True
    model.use_mixture = False
    model._model = model.hb_rl
    model.build_dir = BUILD_DIR
    enable_logger(model.build_dir)

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
        site.a.loc, site.a.scale,
        site.b.loc, site.b.scale,
        site.h.loc, site.h.scale,
        site.g.scale,
        site.v.scale,
        site.c1.scale,
        site.c2.scale,
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

    named_params = [site.a, site.b, site.g, site.h, site.v]
    ylo = F.rectified_logistic(
        x=5, **{u: v for u, v in ppd.items() if u in named_params}
    )
    yhi = F.rectified_logistic(
        x=95, **{u: v for u, v in ppd.items() if u in named_params}
    )
    y = yhi - ylo
    y = y[:, 0]

    # Filter draws
    a = ppd[site.a]
    a = a[:, 0]
    idx = (
        np.all(y > .15, axis=(-1))
        & np.all((a > 10) & (a < 70), axis=(-1))
    )
    ppd = {u: v[idx, ...] for u, v in ppd.items()}
    logger.info(f"Num draws left: {ppd[site.a].shape[0]}")
 
    # Shuffle draws
    logger.info(f"Shuffling draws ...")
    idx = np.arange(0, ppd[site.a].shape[0], 1)
    _, subkey = random.split(model.key)
    idx = np.array(random.permutation(subkey, idx))
    ppd = {u: v[idx, ...] for u, v in ppd.items()}
    ppd = {u: v[:5000, ...] for u, v in ppd.items()}

    save_model(model, df, ppd)
    return


if __name__ == "__main__":
    main()
