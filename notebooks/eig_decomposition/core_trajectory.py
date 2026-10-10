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
from jax import random, numpy as jnp

from paper.agent import FocusedEIG
from paper.default_kw import BaseAgent_kw, FocusedEIG_kw

from constants import (
    SIMULATIONS_OUT_DIR,
    BUILD_DIR,
    TRAJECTORY_OUT_DIR,
)

logger = logging.getLogger(__name__)
FORMAT = "%(name)s - %(levelname)s - %(message)s"

# SIMULATIONS_OUT_DIR = "/home/vishu/reports/c2b2-Jun-2026/eig_out_simulations"

REFERENCE_DIR = os.path.join(
    SIMULATIONS_OUT_DIR,
    "rl_gamma/accuracy_data/400w_400s_4c_1t_10d_80a_fm/PKPK_APB/rl_gamma/eig-mcmc"
)

N0 = 8
N1 = 3
N = 16
RESPONSE = ["PKPK_APB"]


def load_posterior(draw, num_samples):
    model_dir = os.path.join(REFERENCE_DIR, f"d{draw:04d}")
    ckpt_dir = os.path.join(model_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError
    ckpt = one.CheckpointManager(ckpt_dir)
    step = N0 + num_samples
    pay = ckpt.load_step(step)
    model = one.BaseAgent().load_state_dict(pay["agent_state"])
    vars = model.variables
    data = model.data
    post = model.posterior
    named_params = list(post.named_params)
    model._posterior_u = None
    post_u = model.posterior_u
    return data, named_params, post.as_dict(), post_u


def make_config():
    variables = dict(
        intensity="TMSInt",
        features=["participant"],
        response=RESPONSE,
    )
    config = dict(variables=variables)
    return config


def make_agent(run_id, data, named_params, post, post_u):
    focus_parameters = run_id.split(",")
    focus_parameters = [getattr(mep.site, u) for u in focus_parameters]
    model = FocusedEIG(
        config=make_config(),
        **BaseAgent_kw(N0=N0, N1=N1).model_dump(),
        **FocusedEIG_kw(focus_parameters=focus_parameters).model_dump()
    )
    for u in focus_parameters:
        z = f"z.{u}"
        assert z not in post
        post[z] = post_u[u]
        named_params.append(z)
    
    model._model = model.rl_gamma
    model.use_mixture = False
    model.data = data
    model.posterior = one.Posterior.from_dict(post, named_params=named_params)
    model._posterior_u = None
    return model


def run_one(run_id, draw, num_samples):
    mep.enable_logging(format=FORMAT)
    key = random.key(9937)
    key = random.fold_in(key, draw)
    key = random.fold_in(key, num_samples)
    out = load_posterior(draw, num_samples)
    model = make_agent(run_id, *out)
    key, sub = random.split(key)
    model._key = sub
    logger.info(f"({draw}, {num_samples}) Proposing ...")
    model.propose()
    logger.info(f"({draw}, {num_samples}) Finished.")

    agent_state = model.state_dict(include_posterior=False)
    model_dir = os.path.join(
        TRAJECTORY_OUT_DIR,
        "".join(run_id.split(",")),
        f"d{draw:04d}",
    )
    ckpt_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = one.CheckpointManager(ckpt_dir)
    step = N0 + num_samples
    ckpt.save(step=step, agent_state=agent_state, env_state=None)
    logger.info(f"({draw}, {num_samples}) Saved to {model_dir}")
    return


def safe_run_one(*args, **kw):
    try:
        run_one(*args, **kw)
        return None
    except Exception:
        return traceback.format_exc()
    finally:
        _shutdown_executors()


def main(run_id, draws_range, samples_range):
    tasks = [(run_id, d, n) for d in range(*draws_range) for n in range(*samples_range)]
    for task in tasks:
        err = safe_run_one(*task)
        if err is None:
            print(f"{', '.join(map(str, task))} ok.")
        else:
            print(f"{', '.join(map(str, task))} FAILED")
            print(err)
            output_path = os.path.join(
                TRAJECTORY_OUT_DIR,
                f"failed_{task[0]}_{task[1]}_{task[2]}.txt",
            )
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"Task: {task}\n\n")
                f.write(traceback.format_exc())
    return


if __name__ == "__main__":
    RUN_ID = "a"
    LO, HI = 0, 1

    args = sys.argv[1:]
    if len(args) == 5:
        run_id = args[0]
        dlo, dhi = map(int, args[1:3])
        draws_range = (dlo, dhi)
        nlo, nhi = map(int, args[3:5])
        samples_range = (nlo, nhi)
    else:
        run_id = RUN_ID
        draws_range = (LO, HI)
        samples_range = (3, 16 + 1)

    main(run_id, draws_range, samples_range)
