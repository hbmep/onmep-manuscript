import os
import pickle
import logging
import traceback

import hbmep as mep
from jax import random
from onmep import CheckpointManager
from onmep.agent.multiprocessing import _shutdown_executors

from paper.agent import BaseAgent, EIG, PEST
from paper.environment import SimulationEnv
from paper.util import get_subname

from default_kw import (
    MCMC_kw,
    Uniform_kw,
    EIG_kw,
    PEST_kw,
)

logger = logging.getLogger(__name__)
FORMAT = "%(name)s - %(levelname)s - %(message)s"


def safe_run_one(draw, run_id, target, subject, feature0, **kw):
    try:
        run_one(draw, run_id, target, subject, feature0, **kw)
        return draw, run_id, target, subject, feature0, None
    except Exception:
        return draw, run_id, target, subject, feature0, traceback.format_exc()
    finally:
        _shutdown_executors()


def run_one(
    draw,
    run_id,
    target,
    subject,
    feature0,
    *,
    simulations_dir,
    build_dir,
    response,
    save_every,
    baseagent_kw,
):
    variables = dict(
        intensity="TMSInt",
        features=["intervention"],
        response=response,
    )
    config = dict(
        variables=variables,
        mcmc_kw=MCMC_kw().model_dump()
    )

    key = random.key(2081)
    key = random.fold_in(key, draw)
    key = random.fold_in(key, subject)
    key, sub = random.split(key)
    common_kw = dict(
        key=sub,
        config=config,
        **baseagent_kw,
    )

    model_name = "rl_gamma"

    # Build agent
    match run_id:
        case "uniform-mcmc":
            agent_cls = BaseAgent
            agent_kw = Uniform_kw(update_every=16).model_dump()

        case s if s in {"eig-mcmc"}:
            agent_cls = EIG
            agent_kw = EIG_kw().model_dump()

        case s if s in {"pest-mle1", "pest-mle2"}:
            agent_cls = PEST
            agent_kw = PEST_kw(
                method=s.split("-")[1],
                update_every=16,
            ).model_dump()

        case _:
            raise ValueError(f"Invalid run_id {run_id}")

    model = agent_cls(**common_kw, **agent_kw)
    model.use_mixture = False
    model._model = getattr(model, model_name)

    if feature0 is not None:
        assert common_kw["K"] == 1
        f0_dir = f"k1_f{feature0}"
    else:
        assert common_kw["K"] == 2
        f0_dir = "k2"

    model.build_dir = os.path.join(
        build_dir,
        get_subname(model),
        "__".join(response),
        model._model.__name__,
        run_id,
        f"d{draw:04d}",
        f"s{subject:02d}",
        f0_dir,
    )

    mep.enable_logging(model.build_dir, format=FORMAT)
    logger.info(f"response: {response}")
    logger.info(
        f"draw:{draw}, run_id: {run_id}, target: {target}, "
        f"subject: {subject}, feature0: {feature0}, "
    )

    # Build environment
    key, sub = random.split(key)
    env = SimulationEnv(
        key=sub,
        model_dir=simulations_dir,
        draw=draw,
        response=response,
    )
    assert env.model.simulate == True

    # Set the subject and feature0
    priors = [
        mep.site.a, mep.site.b, mep.site.g, mep.site.h, mep.site.v,
        mep.site.c1, mep.site.c2,
    ]

    for u in priors:
        env.post_one[u] = env.post_one[u][:, subject:subject + 1]
        if feature0 is not None:
            env.post_one[u] = env.post_one[u][:, :, feature0:feature0 + 1]

    for feat in env.model.features:
        env.df[feat] = 0

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
        update_on_action_feature=True if model.K == 2 else False,
    )

    # Save to disk
    mep.save(
        model_state=model.state_dict(include_posterior=False),
        output_dir=model.build_dir
    )
    if draw < 10:
        model._plot_curves()

    return


def main(
    draws_range: tuple[int, int],
    subjects_range: tuple[int, int],
    feature0_range: tuple[int, int] | None,
    run_id,
    *,
    target,
    simulations_data_dir,
    simulations_out_dir,
    response,
    save_every,
    baseagent_kw,
    use_fp,
):
    simulations_dir = os.path.join(simulations_data_dir, "rl_gamma")
    build_dir = os.path.join(simulations_out_dir, "rl_gamma")

    if use_fp:
        simulations_dir = os.path.join(simulations_dir, "power_data_fp")
        build_dir = os.path.join(build_dir, "power_data_fp")
    else:
        simulations_dir = os.path.join(simulations_dir, "power_data_tp")
        build_dir = os.path.join(build_dir, "power_data_tp")

    if feature0_range is not None:
        feature0_range = range(*feature0_range)
    else:
        feature0_range = [None]

    tasks = [
        (draw, run_id, target, subject, feature0)
        for draw in range(*draws_range)
        for subject in range(*subjects_range)
        for feature0 in feature0_range
    ]

    for task in tasks:
        *_, err = safe_run_one(
            *task,
            simulations_dir=simulations_dir,
            build_dir=build_dir,
            response=response,
            save_every=save_every,
            baseagent_kw=baseagent_kw,
        )
        if err is None:
            print(f"draw {task[0]}, run_id={task[1]}, target={task[2]}, subject={task[3]}, feature0={task[4]}, ok.")
        else:
            print(f"draw {task[0]}, run_id={task[1]}, target={task[2]}, subject={task[3]}, feature0={task[4]}, FAILED.")
            print(err)
            output_path = os.path.join(build_dir, f"failed_{task[0]}_{task[1]}_{task[2]}_{task[3]}_{task[4]}.pkl")
            with open(output_path, "wb") as f:
                pickle.dump({"task": task, "error": err}, f)

    return
