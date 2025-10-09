import os
import sys
import psutil
import logging
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
# os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# import numpyro
# numpyro.set_host_device_count(1)
from jax import random
from joblib import Parallel, delayed

from rtmep import CheckpointManager
from rtmep.agent import EIG
from rtmep.environment import SimulationEnv

from rtmep.notebooks.util import save_model, enable_logger
from rtmep.notebooks.constants import TMS_TOML
from learn_posterior import BUILD_DIR as LEARN_POSTERIOR_DIR
from simulate_data import BUILD_DIR as MODEL_DIR

logger = logging.getLogger(__name__)
ACCURACY_DIR = os.path.dirname(LEARN_POSTERIOR_DIR)
BUILD_DIR = os.path.join(ACCURACY_DIR, "out")
os.makedirs(BUILD_DIR, exist_ok=True)

MAX_OBSERVATIONS = 64
RESPONSE = ['PKPK_APB', 'PKPK_ECR', 'PKPK_FCR']
RESPONSE = ['PKPK_APB']

# NUM_CORES = psutil.cpu_count(logical=False) or 1
# NUM_JOBS = max(1, NUM_CORES - 2)
NUM_JOBS = 1


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
        case "eig":
            model = EIG(
                toml_path=toml_path,
                min_observations=3,
                grid_lo=0., grid_hi=100., grid_n=201,
                method="mc",
                num_outer=None,
                max_step=50.,
                batch_size=41
            )
            model._model = model.rl
            model.propose = model.eig_gamma_proposal
            model.update = model.update_on_each
            model.mcmc_params = dict(
                thinning=1,
                num_chains=4,
                num_warmup=400,
                num_samples=400,
                progress_bar=False
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
        resume=False
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
    # lo, hi = 0, 1
    main(lo, hi, run_id, NUM_JOBS)
