import os
# os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
# os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import gc
import sys
import psutil
import logging
from contextlib import suppress

import matplotlib as mpl
os.environ.setdefault("MPLBACKEND", "Agg")
mpl.use("Agg", force=True)

import hbmep as mep
import jax
from jax import random
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from rtmep import CheckpointManager
from rtmep.environment import SimulationEnv
from rtmep.agent import EIG

from rtmep.notebooks.util import save_model, get_subname
from rtmep.notebooks.constants import TMS_TOML
from constants import LEARN_POSTERIOR_DIR, TP_DIR, FP_DIR, POWER_DIR

logger = logging.getLogger(__name__)

MODEL_DIR = FP_DIR
# MODEL_DIR = TP_DIR
basename = os.path.basename(MODEL_DIR).split("_")[-1]
BUILD_DIR = os.path.join(POWER_DIR, basename)
os.makedirs(BUILD_DIR, exist_ok=True)

MCMC_PARAMS = dict(
    thinning=1, num_chains=4, num_warmup=400,
    num_samples=400, progress_bar=False,
)

K = 2
N0 = 8
N1 = 3
N = 24
GRID_N = 101

NUM_OUTER = None
BATCH_SIZE = 21
MAX_STEP = None
NUM_JOBS = 1

RESPONSE = [
    'PKPK_APB',
    # 'PKPK_ECR',
    # 'PKPK_FCR'
]

SAVE_EVERY = 8
TOTAL_SUBJECTS = 16

RUN_ID, LO, HI = "eig-mcmc", 0, 1000
# RUN_ID, LO, HI = "random-mcmc", 0, 1000


def run_one_draw(
    build_dir,
    model_dir,
    toml_path,
    response,
    max_obs,
    run_id,
    draw,
    participant,
):
    model = None
    env = None
    ckpt = None
    try:
        common_kw = dict(
            key=random.key(1729 + 10_000 * draw + participant), toml_path=toml_path,
            K=K, N0=N0, N1=N1, grid_lo=0., grid_hi=100., grid_n=GRID_N,
            initial_capacity=(K * (N0 + N)) // 4,
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
                save_every = SAVE_EVERY
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
                save_every = SAVE_EVERY
                agent_tags = dict(agent="EIG", policy="eig")

            # case "eig-focused-mcmc":
            #     model = EIGFocused(
            #         **common_kw,
            #         infer="mcmc",
            #         num_outer=NUM_OUTER,
            #         max_step=MAX_STEP,
            #         batch_size=BATCH_SIZE,
            #     )
            #     model._model = model.rl
            #     model.propose = model.eig_focused_proposal
            #     model.update = model.update_on_each
            #     model.mcmc_params = dict(
            #         **MCMC_PARAMS
            #     )
            #     save_every = SAVE_EVERY
    
            case _:
                raise ValueError(f"Invalid run_id {run_id}")

        draw_id = f"d{draw:04d}"
        participant_id = f"p{participant:02d}"
        build_dir = os.path.join(
            build_dir,
            get_subname(model),
            "__".join(response),
            run_id,
            draw_id,
            participant_id,
        )
        mep.enable_logging(build_dir)
        logger.info(f"response: {response}")
        logger.info(
            f"draw:{draw:04d}, participant:{participant:02d}, "
            f"run_id: {run_id}, max_obs: {max_obs}"
        )

        model.build_dir = build_dir
        model.features = ["intervention"]
        if response != model.response:
            model.response = [u for u in response]

        # Build environment
        env = SimulationEnv(
            model_dir=model_dir,
            draw=draw,
            participant=participant
        )
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
            save_every=save_every,
            resume=False,
            timings_out=timings_out,
            timings_flush_every=timings_flush_every,
            timing_tags=dict(**agent_tags, **common_tags)
        )

        # Save to disk
        # model.plot_curves()
        save_model(model, None, None)
        logger.info(f"draw:{draw:04d}, participant{participant:02d} ok.")
        return

    finally:
        with suppress(Exception):
            del ckpt
        with suppress(Exception):
            del env
        with suppress(Exception):
            del model
        with suppress(Exception):
            plt.close("all")
        with suppress(Exception):
            jax.clear_caches()
        with suppress(Exception):
            gc.collect()


def main(
    run_id,
    draws_range: tuple[int],
    p_range: tuple[int] | None = None,
    n_jobs=-1,
):
    draws = list(range(*draws_range))
    if p_range is None:
        p_range = (0, TOTAL_SUBJECTS)
    participants = list(range(*p_range))
    tasks = [(run_id, d, p) for d in draws for p in participants]
    print(
        f"Running {run_id} for draws {draws[0]} to {draws[-1]}, "
        f"participants {participants[0]} to {participants[-1]}"
    )
    # for job in tasks:
    #     print(job)

    try:
        with Parallel(n_jobs=n_jobs) as parallel:
            parallel(
                delayed(run_one_draw)(
                    BUILD_DIR,
                    MODEL_DIR,
                    TMS_TOML,
                    RESPONSE,
                    (N0 + N) * K,
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
    args = sys.argv[1:]
    run_id = args[0]
    dlo, dhi, plo, phi = map(int, args[1:])
    plo, phi = max(0, plo), min(TOTAL_SUBJECTS, phi)
    draws_range = (dlo, dhi)
    p_range = (plo, phi)
    n_jobs = NUM_JOBS
    main(run_id, draws_range, p_range=p_range, n_jobs=n_jobs)

    # run_id = RUN_ID
    # draws_range = (LO, HI)
    # n_jobs = NUM_JOBS
    # main(run_id, draws_range, n_jobs=n_jobs)
