import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""     # comment out to use GPU
os.environ.setdefault("MPLBACKEND", "Agg")

import hbmep as mep
import onmep as one
import numpy as np
from jax import random

from paper.model.simulate import Simulator

from constants import (
    CKPT_DIR,
    BOOTSTRAP_DATA_DIR,
    SUBJECTS,
    REPORTS,
)

N0 = 8
N1 = 3
N = 40


def main():
    key = random.key(2366)

    arrs = []
    named_params = None
    for subject in SUBJECTS:
        model_dir = os.path.join(CKPT_DIR, subject)
        ckpt_dir = os.path.join(model_dir, "checkpoints")
        if not os.path.exists(ckpt_dir):
            raise FileNotFoundError
        ckpt = one.CheckpointManager(ckpt_dir)
        step = N0 + N
        pay = ckpt.load_step(step)
        post = one.Posterior.from_state(pay["agent_state"]["posterior_state"])
        if named_params is None:
            named_params = post.named_params
        else:
            assert named_params == post.named_params
        arr = post.array
        key, sub = random.split(key)
        idx = np.asarray(random.choice(sub, np.arange(arr.shape[1]), replace=False, shape=(4000,)))
        arrs.append(arr[:, idx])

    post = np.concatenate(arrs, axis=-2)

    model_dir = os.path.join(REPORTS, "learn_posterior", "rl_gamma")
    model_state, posterior, df, encoder, mcmc = mep.load(model_dir)
    model = Simulator()
    model.load_state_dict(model_state)
    assert model._model.__name__ == "rl_gamma"
    assert not model.use_mixture
    assert not model.simulate
    model.simulate = True
    model.build_dir = BOOTSTRAP_DATA_DIR
    os.makedirs(model.build_dir, exist_ok=True)

    ppd = one.Posterior(array=post, named_params=named_params)
    ppd = ppd.as_dict()

    g = ppd[mep.site.g]
    g = g[(g > 0.01) & (g < 0.025)]
    key, sub = random.split(key)
    g = random.choice(sub, g, shape=ppd[mep.site.a].shape, replace=True)
    g = np.asarray(g)
    ppd[mep.site.g] = g

    df = df.iloc[[0]].copy().reset_index(drop=True)
    df[model.intensity] = 0.
    df[model.features[0]] = 0
    model_state = model.state_dict()
    mep.save(
        model_state=model_state,
        posterior=ppd,
        df=df,
        encoder=None,
        mcmc=None,
        output_dir=model.build_dir
    )
    return


if __name__ == "__main__":
    main()
