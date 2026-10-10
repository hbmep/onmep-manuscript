import os
os.environ.setdefault("MPLBACKEND", "Agg")
import sys
import logging
import traceback

import hbmep as mep
import onmep as one
from onmep.agent.multiprocessing import _shutdown_executors
from jax import random

from paper.agent import (
    EIG,
    EVR,
    FocusedEIG,
)
from paper.environment import SimulationEnv

from default_kw import (
    BaseAgent_kw,
    MCMC_kw,
    EIG_kw,
    EVR_kw,
    FocusedEIG_kw,
)
from constants import (
    SIMULATIONS_DATA_DIR,
    TIMINGS_OUT_DIR,
    SIM_RESPONSE,
)

logger = logging.getLogger(__name__)
FORMAT = "%(name)s - %(levelname)s - %(message)s"

BUILD_DIR = os.path.join(TIMINGS_OUT_DIR, "times_Sep10")
os.makedirs(BUILD_DIR, exist_ok=True)

N0 = 8
N1 = 3
N = 64


def run_one(
    draw,
    run_id,
    response_len,
):
    target = N0 + N

    variables = dict(
        intensity="TMSInt",
        features=["participant"],
        response=SIM_RESPONSE[:response_len],
    )
    config = dict(
        variables=variables,
        mcmc_kw=MCMC_kw().model_dump()
    )

    key = random.key(9241)
    key = random.fold_in(key, draw)

    key, sub = random.split(key)
    common_kw = dict(
        key=sub,
        config=config,
        **BaseAgent_kw(N0=N0, N1=N1, initial_capacity=target).model_dump(),
    )

    # Build agent
    match run_id:
        case s if s in {"eig-mcmc"}:
            agent_cls = EIG
            agent_kw = EIG_kw().model_dump()

        case s if s.startswith("focused_eig_") and s.endswith("-mcmc"):
            agent_cls = FocusedEIG
            suffix = s[len("focused_eig_"):-len("-mcmc")]
            mapping = {
                "a": mep.site.a,
                "b": mep.site.b,
                "g": mep.site.g,
                "h": mep.site.h,
                "v": mep.site.v,
            }
            try:
                focus_params = tuple(mapping[c] for c in suffix)
            except KeyError as e:
                raise ValueError(f"Unknown focused EIG parameter: {e.args[0]}")
            batch_size = FocusedEIG_kw().batch_size
            if response_len > 1:
                batch_size = 41
            agent_kw = FocusedEIG_kw(
                focus_parameters=focus_params,
                batch_size=batch_size,
            ).model_dump()

        case s if s in {"evr-mcmc", "evr-mcmc-sum"}:
            agent_cls = EVR
            agent_kw = EVR_kw().model_dump()

        case _:
            raise ValueError(f"Invalid run_id {run_id}")

    model = agent_cls(
        **common_kw,
        **agent_kw,
    )

    model.use_mixture = False
    model._model = getattr(model, "rl_gamma")
    model.build_dir = os.path.join(
        BUILD_DIR,
        f"r{response_len:02d}",
        run_id,
        f"d{draw:04d}",
    )

    mep.enable_logging(model.build_dir, format=FORMAT)
    logger.info(f"draw:{draw}, run_id: {run_id}, target: {target}")

    # Build environment
    simulations_data_dir = os.path.join(
        SIMULATIONS_DATA_DIR,
        "rl_gamma/accuracy_data",
    )
    key, sub = random.split(key)
    env = SimulationEnv(
        key=sub,
        model_dir=simulations_data_dir,
        draw=draw,
        response=SIM_RESPONSE[:response_len],
    )
    assert env.model.simulate == True

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
        timings_out=timings_out,
        timings_tags=tags,
    )

    # Save to disk
    mep.save(
        model_state=model.state_dict(include_posterior=False),
        output_dir=model.build_dir
    )
    
    return


def safe_run_one(*args, **kw):
    try:
        run_one(*args, **kw)
        return None
    except Exception:
        return traceback.format_exc()
    finally:
        _shutdown_executors()


def main(
    draws_range,
    run_id,
    response_len,
):
    tasks = [(d, run_id, response_len) for d in range(*draws_range)]
    for task in tasks:
        err = safe_run_one(*task)
        if err is None:
            print(f"{', '.join(map(str, task))} ok.")
        else:
            print(f"{', '.join(map(str, task))} FAILED")
            print(err)
            output_path = os.path.join(
                BUILD_DIR,
                f"failed_{task[2]}_{task[0]}_{task[1]}.txt",
            )
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"Task: {task}\n\n")
                f.write(err)
    return


if __name__ == "__main__":
    RUN_ID = "eig-mcmc"
    LO, HI = 0, 1
    RESPONSE_LEN = 1

    args = sys.argv[1:]
    if not args:
        run_id = RUN_ID
        draws_range = (LO, HI)
        response_len = RESPONSE_LEN
    else:
        run_id = args[0]
        draws_range = tuple(map(int, args[1:3]))
        response_len = int(args[3])

    main(
        draws_range,
        run_id,
        response_len=response_len,
    )
