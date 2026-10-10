import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""     # comment out to use GPU
os.environ.setdefault("MPLBACKEND", "Agg")
import pickle

import hbmep as mep
import onmep as one
import numpy as np

from constants import (
    BOOTSTRAP_DATA_DIR,
    BOOTSTRAP_OUT_DIR,
    ACCURACY_OUT_DIR,
)

BUILD_DIR = ACCURACY_OUT_DIR
os.makedirs(BUILD_DIR, exist_ok=True)

RUN_IDS = [
    "eig-mcmc",
    "pest-mle1",
    "uniform-mcmc",
]
NS_SPACE = list(range(3, 64 + 1))
NUM_SUBJECTS = 8
N0 = 8

NUM_DRAWS = 200


def run_one(draw, param):
    try:
        _, post_one, *_ = mep.load(BOOTSTRAP_DATA_DIR, memmap_draw=draw)
        match param:
            case mep.site.a:
                true = post_one[param].reshape(-1,) # (8,)
            case "mu_int":
                post_one[mep.site.g] *= 0
                xgrid = np.linspace(0, 100, 501)[:, None]
                with mep.use_gpu():
                    true = np.asarray(mep.functional.rectified_logistic(
                        xgrid,
                        **{
                            u: post_one[u].reshape(-1,)[None]
                            for u in [
                                mep.site.a,
                                mep.site.b,
                                mep.site.g,
                                mep.site.h,
                                mep.site.v,
                            ]
                        }
                    ))      # (500, 8)
                h = post_one[mep.site.h].reshape(-1,)
            case _:
                raise ValueError
    except Exception as e:
        print(e)
        return None

    errs = []
    try:
        for run_id in RUN_IDS:
            for subject in range(NUM_SUBJECTS):
                model_dir = os.path.join(
                    BOOTSTRAP_OUT_DIR,
                    run_id,
                    f"d{draw:04d}",
                    f"s{subject:02d}"
                )
                ckpt_dir = os.path.join(model_dir, "checkpoints")
                if not os.path.exists(ckpt_dir):
                    raise FileNotFoundError
                ckpt = one.CheckpointManager(ckpt_dir)
                for ns in NS_SPACE:
                    step = N0 + ns
                    pay = ckpt.load_step(step)
                    post = one.Posterior.from_state(pay["agent_state"]["posterior_state"]).as_dict()
                    match param:
                        case mep.site.a:
                            pred = post[param].reshape(-1,)
                        case "mu_int":
                            post[mep.site.g] *= 0
                            with mep.use_gpu():
                                pred = np.asarray(mep.functional.rectified_logistic(
                                    xgrid,
                                    **{
                                        u: post[u].reshape(-1,)[None]
                                        for u in [
                                            mep.site.a,
                                            mep.site.b,
                                            mep.site.g,
                                            mep.site.h,
                                            mep.site.v,
                                        ]
                                    }
                                ))      # (500, 1600)
                        case _:
                            raise ValueError

                    err = mep.metrics.lp_risk(
                        true[..., subject][..., None],
                        pred,
                        axis=-1,
                        ignore_nan=True,
                        p=1
                    )

                    if param == "mu_int":
                        dx = xgrid[1] - xgrid[0]
                        dx = dx.item()
                        err = np.trapz(err, dx=dx, axis=0)
                        err = err / h[subject]

                    errs.append(err)

    except Exception as e:
        print(e)
        return None

    else:
        errs = np.asarray(errs)
        errs = errs.reshape(len(RUN_IDS), NUM_SUBJECTS, len(NS_SPACE), *errs.shape[1:])
        return errs, draw


def main(param):
    num_draws = NUM_DRAWS
    tasks = [(d, param) for d in range(num_draws)]
    batch_size = 24
    out = mep.run_batched(run_one, tasks=tasks, batch_size=batch_size)
    err, draws = zip(*out)
    err = np.asarray(err)
    draws = np.asarray(draws)
    assert err.shape[0] == draws.shape[0] == num_draws, [
        u for u in range(num_draws) if u not in draws
    ]
    assert sorted(draws) == list(draws)
    d = dict(
        run_ids=RUN_IDS,
        ns_space=NS_SPACE,
        param=param,
        err=err,
        draws=draws,
    )
    out_path = os.path.join(BUILD_DIR, f"{param}_{num_draws:04d}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(d, f)
    return out_path, err.shape


if __name__ == "__main__":
    out = []
    out.append(main(mep.site.a))
    out.append(main("mu_int"))
    for row in out:
        print(f"Saved to {row[0]}, shape {row[1]}")
