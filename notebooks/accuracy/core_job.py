import os
# os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
# os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import logging
from contextlib import suppress

import matplotlib as mpl
os.environ.setdefault("MPLBACKEND", "Agg")
mpl.use("Agg", force=True)

import hbmep as mep
from jax import random
from joblib import Parallel, delayed
from rtmep import CheckpointManager
from rtmep.environment import SimulationEnv
from rtmep.agent import EIG, EVR

from rtmep.notebooks.util import save_model, get_subname
from rtmep.notebooks.constants import TMS_TOML
from learn_posterior import BUILD_DIR as LEARN_POSTERIOR_DIR
from simulate_data import BUILD_DIR as MODEL_DIR

logger = logging.getLogger(__name__)
ACCURACY_DIR = os.path.dirname(LEARN_POSTERIOR_DIR)

BUILD_DIR = os.path.join(ACCURACY_DIR, "out")
# BUILD_DIR = os.path.join(ACCURACY_DIR, "out-time")
os.makedirs(BUILD_DIR, exist_ok=True)

MCMC_PARAMS = dict(
    thinning=1, num_chains=4, num_warmup=400,
    num_samples=400, progress_bar=False,
)

K = 1
N0 = 8
N1 = 3
N = 64
GRID_N = 101

NUM_OUTER = None
BATCH_SIZE = 51
MAX_STEP = None
NUM_JOBS = 1

RESPONSE = [
    'PKPK_APB',
    'PKPK_ECR',
    'PKPK_FCR'
]

SAVE_EVERY = 1

RUN_ID, LO, HI = (
    # "random-mcmc", 0, 1000
    # "eig-mcmc", 0, 1000
    # "eig-sequential-3", 0, 1000
    # "eig-sequential-5", 0, 1000
    "evr-mcmc", 0, 1000
)


def run_one_draw(
    build_dir,
    model_dir,
    toml_path,
    response,
    max_obs,
    run_id,
    draw
):
    common_kw = dict(
        key=random.key(1729 + draw), toml_path=toml_path,
        K=K, N0=N0, N1=N1, grid_lo=0., grid_hi=100., grid_n=GRID_N,
        initial_capacity=(N0 + N) // 2,
    )
    # Build agent
    match run_id:
        case "random-mcmc":
            model = EIG(
                **common_kw,
                infer="mcmc",
                policy="random",
            )
            model._model = model.rl
            model.mcmc_params = dict(**MCMC_PARAMS)
            agent_tags = dict(agent="EIG", policy="random")

        case "eig-mcmc":
            model = EIG(
                **common_kw,
                infer="mcmc",
                num_outer=NUM_OUTER,
                batch_size=BATCH_SIZE,
                max_step=MAX_STEP,
                policy="eig",
            )
            model._model = model.rl
            model.mcmc_params = dict(**MCMC_PARAMS)
            agent_tags = dict(agent="EIG", policy="eig")

        case s if s.startswith("eig-sequential"):
            ess_threshold = run_id.split("-")[-1]
            ess_threshold = int(ess_threshold) / (10 ** len(ess_threshold))
            model = EIG(
                **common_kw,
                infer="sequential",
                ess_threshold=ess_threshold,
                num_outer=NUM_OUTER,
                batch_size=BATCH_SIZE,
                max_step=MAX_STEP,
                policy="eig",
            )
            model._model = model.rl
            model.mcmc_params = dict(**MCMC_PARAMS)
            agent_tags = dict(agent="EIG", policy="eig")

        case "evr-mcmc":
            model = EVR(
                **common_kw,
                infer="mcmc",
                num_outer=NUM_OUTER,
                batch_size=BATCH_SIZE,
                max_step=MAX_STEP,
            )
            model._model = model.rl
            model.mcmc_params = dict(**MCMC_PARAMS)
            agent_tags = dict(agent="EVR", policy="threshold")
 
        case _:
            raise ValueError(f"Invalid run_id {run_id}")

    draw_id = f"draw_{draw:04d}"
    build_dir = os.path.join(
        build_dir,
        get_subname(model),
        "__".join(response),
        run_id,
        draw_id,
    )
    mep.enable_logging(build_dir)
    logger.info(f"response: {response}")
    logger.info(f"draw:{draw}, run_id: {run_id}, max_obs: {max_obs}")

    model.build_dir = build_dir
    if response != model.response:
        model.response = [u for u in response]

    # Build environment
    env = SimulationEnv(model_dir=model_dir, draw=draw)
    if response != env.model.response:
        env.response = [u for u in response]

    # Build checkpoint manager
    ckpt_dir = os.path.join(model.build_dir, "checkpoints")
    ckpt = CheckpointManager(ckpt_dir)

    # Timings
    timings_out = os.path.join(model.build_dir, "timings.ndjson")
    timings_flush_every = 1
    common_tags=dict(
        infer=model.infer, draw=env.draw, max_obs=max_obs,
        grid_n=model.grid_n, _model=model._model.__name__,
        batch_size=model.batch_size, num_outer=model.num_outer,
        mcmc_params=model.mcmc_params,
    )

    # Run
    env.interact(
        agent=model,
        max_observations=max_obs,
        ckpt=ckpt,
        save_every=SAVE_EVERY,
        resume=False,
        timings_out=timings_out,
        timings_flush_every=timings_flush_every,
        timing_tags=dict(**agent_tags, **common_tags)
    )

    # Save to disk
    if draw < 20:
        model.plot_curves()
    save_model(model, None, None)
    logger.info(f"{draw} ok.")
    return


def main(
    run_id,
    draws_range: tuple[int],
    n_jobs=-1,
):
    draws = list(range(*draws_range))
    print(f"Running {run_id} for draws {draws[0]} to {draws[-1]}")
    tasks = [(run_id, d) for d in draws]

    try:
        with Parallel(n_jobs=n_jobs) as parallel:
            parallel(
                delayed(run_one_draw)(
                    BUILD_DIR,
                    MODEL_DIR,
                    TMS_TOML,
                    RESPONSE,
                    N0 + N,
                    *args
                ) for args in tasks
        )

    except KeyboardInterrupt:
        with suppress(Exception):
            parallel._backend.terminate()
        with suppress(Exception):
            parallel._backend.abort_everything(ensure_ready=True)
        raise

    return


if __name__ == "__main__":
    n_jobs = NUM_JOBS

    args = sys.argv[1:]
    run_id = args[0]
    dlo, dhi = map(int, args[1:])
    draws_range = (dlo, dhi)
    main(run_id, draws_range, n_jobs=n_jobs)

    # run_id = RUN_ID
    # draws_range = (LO, HI)
    # main(run_id, draws_range, n_jobs=n_jobs)
