import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""     # comment out to use GPU
os.environ.setdefault("MPLBACKEND", "Agg")
import sys
import logging
import traceback

import hbmep as mep
import onmep as one
from onmep.agent.multiprocessing import _shutdown_executors
import numpy as np
from jax import random

from paper.environment import SimulationEnv
from paper.agent import BaseAgent, EIG, PEST
from paper.default_kw import (
    MCMC_kw,
    NUTS_kw,
    BaseAgent_kw,
    Uniform_kw,
    EIG_kw,
    PEST_kw,
)

from constants import BOOTSTRAP_DATA_DIR, BOOTSTRAP_OUT_DIR

os.makedirs(BOOTSTRAP_OUT_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
FORMAT = "%(name)s - %(levelname)s - %(message)s"

N0 = 8
N1 = 3
N = 64
SAVE_EVERY = 1

RESPONSE = ["PKPK_APB"]


def run_one(
    draw,
    subject,
    run_id,
):
    variables = dict(
        intensity="TMSInt",
        features=["participant"],
        response=RESPONSE,
    )
    config = dict(
        variables=variables,
        mcmc_kw=MCMC_kw().model_dump(),
        nuts_kw=NUTS_kw().model_dump(),
    )

    key = random.key(6002)
    key = random.fold_in(key, draw)
    key = random.fold_in(key, subject)

    key, sub = random.split(key)
    common_kw = dict(
        key=sub,
        config=config,
        **BaseAgent_kw(N0=N0, N1=N1).model_dump(),
    )

    match run_id:
        case "uniform-mcmc":
            agent_cls = BaseAgent
            agent_kw = Uniform_kw().model_dump()

        case s if s in {"eig-mcmc"}:
            agent_cls = EIG
            agent_kw = EIG_kw().model_dump()

        case s if s in {"pest-mle1", "pest-mle2"}:
            agent_cls = PEST
            agent_kw = PEST_kw(
                method=s.split("-")[1]
            ).model_dump()

        case _:
            raise ValueError(f"Invalid run_id {run_id}")

    model = agent_cls(
        **common_kw,
        **agent_kw,
    )
    model.use_mixture = False
    model._model = getattr(model, "rl_gamma")
    model.build_dir = os.path.join(
        BOOTSTRAP_OUT_DIR,
        run_id,
        f"d{draw:04d}",
        f"s{subject:02d}",
    )

    mep.enable_logging(model.build_dir, format=FORMAT)
    target = N0 + N
    logger.info(f"draw:{draw}, subject:{subject}, run_id: {run_id}, target: {target}")

    # Build environment
    key, sub = random.split(key)
    env = SimulationEnv(
        key=sub,
        model_dir=BOOTSTRAP_DATA_DIR,
        draw=draw,
        response=RESPONSE,
    )
    assert env.model.simulate == True
    env.post_one = {u: v[:, subject] for u, v in env.post_one.items()}

    # Build checkpoint manager
    ckpt_dir = os.path.join(model.build_dir, "checkpoints")
    ckpt = one.CheckpointManager(ckpt_dir)

    # Timings
    timings_out = os.path.join(model.build_dir, "timings.ndjson")
    tags = {mep.site.a: env.post_one[mep.site.a].reshape(-1,).tolist()}

    # Run
    env.interact(
        agent=model,
        target=target,
        ckpt=ckpt,
        save_every=SAVE_EVERY,
        timings_out=timings_out,
        timings_tags=tags,
    )

    # Save to disk
    mep.save(
        model_state=model.state_dict(include_posterior=False),
        output_dir=model.build_dir
    )
    if draw < 150:
        model._plot_curves()
    
    return


def safe_run_one(*args, **kw):
    try:
        run_one(*args, **kw)
        return None
    except Exception:
        return traceback.format_exc()
    finally:
        _shutdown_executors()


def main(draws_range, run_id):
    tasks = [
        (draw, subject, run_id, )
        for draw in range(*draws_range)
        for subject in range(8)
    ]
    for task in tasks:
        err = safe_run_one(*task)
        if err is None:
            print(f"{', '.join(map(str, task))} ok.")
        else:
            print(f"{', '.join(map(str, task))} FAILED")
            print(err)
            output_path = os.path.join(
                BOOTSTRAP_OUT_DIR,
                f"failed_{task[2]}_{task[0]}_{task[1]}.txt",
            )
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"Task: {task}\n\n")
                f.write(traceback.format_exc())
    return


if __name__ == "__main__":
    RUN_ID = "eig-mcmc"
    LO, HI = 0, 200

    args = sys.argv[1:]
    if len(args) == 3:
        run_id = args[0]
        dlo, dhi = map(int, args[1:3])
        draws_range = (dlo, dhi)
    else:
        run_id = RUN_ID
        draws_range = (LO, HI)

    main(draws_range, run_id)
