import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
import sys
import psutil
import logging

import hbmep as mep
from jax import random
from joblib import Parallel, delayed
from rtmep import CheckpointManager
from rtmep.agent import EIG
from rtmep.environment import SimulationEnv

from rtmep.notebooks.util import save_model, enable_logger
from rtmep.notebooks.constants import TMS_TOML, REPORTS

logger = logging.getLogger(__name__)
BUILD_DIR = os.path.join(REPORTS, "tests", "time")
os.makedirs(BUILD_DIR, exist_ok=True)

MODEL_DIR = os.path.join(REPORTS, "accuracy", "learn_posterior", "simulate_data")
MAX_OBSERVATIONS = 72
NUM_JOBS = 1

RESPONSE = ['PKPK_APB', 'PKPK_ECR', 'PKPK_FCR']
# RESPONSE = ['PKPK_APB']


def run_one_draw(
    build_dir,
    model_dir,
    toml_path,
    response,
    max_obs,
    run_id,
    draw
):
    draw_id = f"draw_{draw:04d}"
    build_dir = os.path.join(
        build_dir,
        f"response_{'__'.join(response)}",
        run_id,
        draw_id,
    )
    enable_logger(build_dir)
    logger.info(f"response: {response}")
    logger.info(f"max_obs: {max_obs}")
    logger.info(f"run_id: {run_id}")
    logger.info(f"draw: {draw}")

    # Build agent
    match run_id:
        case "eig-mcmc":
            model = EIG(
                toml_path=toml_path,
                infer="mcmc",
                min_observations=3,
                grid_lo=0., grid_hi=100., grid_n=201,
                method="mc",
                num_outer=None,
                max_step=50.,
                batch_size=67
            )
            model._model = model.rl
            model.propose = model.eig_gamma_proposal
            model.update = model.update_on_each
            model.mcmc_params = dict(
                thinning=2,
                num_chains=16,
                num_warmup=200,
                num_samples=200,
                progress_bar=False,
            )
            save_every = 1

        case "eig-svi":
            model = EIG(
                toml_path=toml_path,
                infer="svi",
                min_observations=3,
                grid_lo=0., grid_hi=100., grid_n=201,
                method="mc",
                num_outer=None,
                max_step=50.,
                batch_size=67
            )
            model._model = model.rl
            model.propose = model.eig_gamma_proposal
            model.update = model.update_on_each
            model.mcmc_params = dict(
                thinning=1,
                num_chains=16,
                num_warmup=100,
                num_samples=100,
            )
            save_every = 1
        
        case "random":
            model = EIG(
                toml_path=toml_path,
                min_observations=3,
                grid_lo=0., grid_hi=100., grid_n=201,
            )
            model._model = model.rl
            model.propose = model.random_proposal
            model.update = model.update_on_each
            model.mcmc_params = dict(
                thinning=1,
                num_chains=4,
                num_warmup=400,
                num_samples=400,
                progress_bar=False
            )
            save_every = 1

        case _:
            raise ValueError(f"Invalid run_id {run_id}")

    model.build_dir = build_dir
    if response != model.response:
        model.response = [u for u in response]
    model._key = random.key(draw)

    # Build environment
    env = SimulationEnv(model_dir=model_dir, draw=draw)
    if response != env.model.response:
        env.response = [u for u in response]

    # Build checkpoint manager
    ckpt_dir = os.path.join(model.build_dir, "checkpoints")
    ckpt = CheckpointManager(ckpt_dir)

    # Run one episode
    env.interact(
        agent=model,
        max_observations=max_obs,
        ckpt=ckpt,
        save_every=save_every,
        resume=False,
        timings_out=os.path.join(model.build_dir, "timings.ndjson"),
        timings_mode="ndjson",
        timings_flush_every=1,
        timings_truncate_on_start=True,
        timing_tags=dict(
            agent="EIG", draw=env.draw, max_obs=max_obs, grid_n=model.grid_n,
            _model=model._model.__name__, propose=model.propose.__name__,
            update=model.update.__name__, batch_size=model.batch_size,
            num_outer=model.num_outer, **model.mcmc_params, infer=model.infer
        )
    )

    # Save to disk
    save_model(model, None, None)
    logger.info(f"{draw} ok.")
    return


def main(lo, hi, run_id, n_jobs):
    print(f"Running draws from {lo} to {hi - 1}")
    draws = list(range(lo, hi))
    tasks = [(run_id, d) for d in draws]

    with Parallel(n_jobs=n_jobs) as parallel:
        parallel(
            delayed(run_one_draw)(
                BUILD_DIR,
                MODEL_DIR,
                TMS_TOML,
                RESPONSE,
                MAX_OBSERVATIONS,
                *args
            ) for args in tasks
        )
    return


if __name__ == "__main__":

    args = sys.argv[1:]
    run_id, lo, hi = args
    lo = int(lo)
    hi = int(hi)

    # run_id = "eig-mcmc"
    # # run_id = "eig-svi"
    # lo, hi = 0, 200
    # # lo = 1
    # # hi = lo + 1

    main(lo, hi, run_id, NUM_JOBS)
