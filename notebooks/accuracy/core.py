import os
import pickle
import logging
import traceback

import hbmep as mep
from jax import random
from onmep import CheckpointManager
from onmep.protocols.round_robin_response import make_round_robin_response
from onmep.agent.multiprocessing import _shutdown_executors

from paper.agent import BaseAgent, EIG, EVR, TTD, PEST, FocusedEIG
from paper.environment import SimulationEnv
from paper.util import get_subname

from default_kw import (
    MCMC_kw,
    Uniform_kw,
    EIG_kw,
    EVR_kw,
    PEST_kw,
    FocusedEIG_kw,
)

logger = logging.getLogger(__name__)
FORMAT = "%(name)s - %(levelname)s - %(message)s"


def safe_run_one(draw, run_id, target, **kw):
    try:
        run_one(draw, run_id, target, **kw)
        return draw, run_id, target, None
    except Exception:
        return draw, run_id, target, traceback.format_exc()
    finally:
        _shutdown_executors()


def run_one(
    draw,
    run_id,
    target,
    *,
    simulations_dir,
    build_dir,
    model_name,
    response,
    save_every,
    baseagent_kw,
):
    variables = dict(
        intensity="TMSInt",
        features=["participant"],
        response=response,
    )
    config = dict(
        variables=variables,
        mcmc_kw=MCMC_kw().model_dump()
    )

    key = random.key(1729 + draw)
    key, sub = random.split(key)
    common_kw = dict(
        key=sub,
        config=config,
        **baseagent_kw,
    )

    # Build agent
    match run_id:
        case "uniform-mcmc":
            agent_cls = BaseAgent
            agent_kw = Uniform_kw().model_dump()

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
            if len(response) > 1:
                batch_size = 41
            agent_kw = FocusedEIG_kw(
                focus_parameters=focus_params,
                batch_size=batch_size,
            ).model_dump()

        case s if s in {"evr-mcmc"}:
            agent_cls = EVR
            agent_kw = EVR_kw().model_dump()

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
    model._model = getattr(model, model_name)

    model.build_dir = os.path.join(
        build_dir,
        get_subname(model),
        "__".join(response),
        model._model.__name__,
        run_id,
        f"d{draw:04d}",
    )

    mep.enable_logging(model.build_dir, format=FORMAT)
    logger.info(f"response: {response}")
    logger.info(f"draw:{draw}, run_id: {run_id}, target: {target}")

    # Build environment
    key, sub = random.split(key)
    env = SimulationEnv(
        key=sub,
        model_dir=simulations_dir,
        draw=draw,
        response=response,
    )
    assert env.model.simulate == True

    # Build checkpoint manager
    ckpt_dir = os.path.join(model.build_dir, "checkpoints")
    ckpt = CheckpointManager(ckpt_dir)

    # Timings
    timings_out = os.path.join(model.build_dir, "timings.ndjson")
    tags = {mep.site.a: env.post_one[mep.site.a].reshape(-1,).tolist()}

    # Run
    env.interact(
        agent=model,
        target=target,
        ckpt=ckpt,
        save_every=save_every,
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


def main(
    draws_range: tuple[int, int],
    run_id,
    *,
    target,
    simulations_data_dir,
    simulations_model_name,
    model_name,
    simulations_out_dir,
    response,
    save_every,
    baseagent_kw,
):
    simulations_dir = os.path.join(
        simulations_data_dir,
        simulations_model_name,
        "accuracy_data",
    )
    build_dir = os.path.join(
        simulations_out_dir,
        simulations_model_name,
        "accuracy_data",
    )

    tasks = [(draw, run_id, target) for draw in range(*draws_range)]
    for task in tasks:
        _, _, _, err = safe_run_one(
            *task,
            simulations_dir=simulations_dir,
            build_dir=build_dir,
            model_name=model_name,
            response=response,
            save_every=save_every,
            baseagent_kw=baseagent_kw,
        )
        if err is None:
            print(f"draw {task[0]}, run_id={task[1]}, target={task[2]} ok.")
        else:
            print(f"draw {task[0]}, run_id={task[1]}, target={task[2]} FAILED")
            print(err)
            output_path = os.path.join(build_dir, f"failed_{task[0]}_{task[1]}_{task[2]}.pkl")
            with open(output_path, "wb") as f:
                pickle.dump({"task": task, "error": err}, f)
    return
