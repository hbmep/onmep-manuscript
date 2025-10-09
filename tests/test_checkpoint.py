import os
import logging

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
from jax import random
from rtmep import CheckpointManager
from rtmep.agent import EIG
from rtmep.environment import SimulationEnv

from rtmep.notebooks.constants import TMS_TOML, REPORTS
from rtmep.notebooks.util import enable_logger

logger = logging.getLogger(__name__)
BUILD_DIR = os.path.join(REPORTS, "tests")
os.makedirs(BUILD_DIR, exist_ok=True)

MODEL_DIR = os.path.join(REPORTS, "accuracy", "learn_posterior", "simulate_data")
MAX_OBS = 5
STOP_THEN = 2
LO, HI = 0, 50


def build_agent(build_dir: str):
    model = EIG(
        toml_path=TMS_TOML,
        min_observations=3,
        grid_lo=0., grid_hi=100., grid_n=201,
        method="mc",
        num_outer=None,
        max_step=50.,
        batch_size=41
    )
    model.build_dir = build_dir
    model._model = model.rl
    model.propose = model.eig_gamma_proposal
    model.update = model.update_on_each
    model.mcmc_params = dict(
        thinning=1, num_chains=4, num_warmup=200, num_samples=200,
        progress_bar=False
    )
    return model


def build_env(model_dir: str, *, draw: int):
    return SimulationEnv(model_dir=model_dir, draw=draw)


def posterior_equal(p1: dict, p2: dict):
    k1 = set(p1.keys())
    k2 = set(p2.keys())
    if k1 != k2:
        missing_1 = sorted(k2 - k1)
        missing_2 = sorted(k1 - k2)
        raise AssertionError(
            f"Posterior key mismatch."
            f"\nOnly in p2: {missing_1}\nOnly in p1: {missing_2}"
        )
    for k in sorted(k1):
        a = np.asarray(p1[k])
        b = np.asarray(p2[k])
        if a.ndim != 2:
            logger.info(f"Skipped {k}.")
            continue
        if a.shape != b.shape:
            raise AssertionError(f"[{k}] shape mismatch: {a.shape} vs {b.shape}")
        np.testing.assert_array_equal(a, b)
        logger.info(f"{k} ok.")
    return True


def run_continuous(model_dir: str, out_root: str, draw: int, *, max_obs: int):
    """
    One uninterrupted run to max_obs. Returns posterior dict.
    """
    base = os.path.join(out_root, f"draw_{draw:04d}", "direct")
    os.makedirs(base, exist_ok=True)

    model = build_agent(base)
    model._key = random.key(draw)
    env = build_env(model_dir, draw=draw)

    env.interact(agent=model, max_observations=max_obs, ckpt=None)
    return model.posterior, np.array(model.data.X.ravel(), copy=True)


def run_resume(model_dir: str, out_root: str, draw: int, *, stop_then: int, max_obs: int):
    """
    Run to `stop_then`, checkpoint, then re-instantiate and resume to `max_obs`.
    Returns posterior dict.
    """
    base = os.path.join(out_root, f"draw_{draw:04d}", "resume")
    ckpt_dir = os.path.join(base, "checkpoints")
    os.makedirs(base, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = CheckpointManager(ckpt_dir)

    # Phase 1: stop early at `stop_then`
    model1 = build_agent(base)
    model1._key = random.key(draw)
    env1   = build_env(model_dir, draw=draw)

    env1.interact(
        agent=model1, max_observations=stop_then, ckpt=ckpt,
        save_every=1, resume=False
    )

    # Phase 2: simulate a "crash + restart": create fresh instances and resume
    model2 = build_agent(base)
    env2   = build_env(model_dir, draw=draw)

    # Resume from latest
    ckpt = CheckpointManager(ckpt_dir)
    env2.interact(agent=model2, max_observations=max_obs, ckpt=ckpt, save_every=1, resume=True)

    return model2.posterior, np.array(model2.data.X.ravel(), copy=True)


def main():
    draws = range(LO, HI)
    model_dir = MODEL_DIR
    max_obs = MAX_OBS
    stop_then = STOP_THEN
    out_root = BUILD_DIR
    enable_logger(out_root)

    all_ok = True
    for d in draws:
        logger.info(f"\n=== Draw {d} ===")
        logger.info(f"Running direct...")
        post_direct, xs_direct = run_continuous(
            model_dir=model_dir, out_root=out_root, draw=d, max_obs=max_obs
        )
        logger.info(f"Finished direct, draw {d}.")
        logger.info(f"Running resume, draw {d}...")
        post_resume, xs_resume = run_resume(
            model_dir=model_dir, out_root=out_root, draw=d,
            stop_then=stop_then, max_obs=max_obs
        )
        logger.info(f"Finished resume, draw {d}.")

        np.testing.assert_array_equal(xs_direct, xs_resume)
        assert posterior_equal(post_direct, post_resume)
        print(f"[OK] Draw {d}: posteriors are exactly identical.")

    if not all_ok:
        raise SystemExit(1)
    print("\nAll draws matched exactly.")


if __name__ == "__main__":
    main()
