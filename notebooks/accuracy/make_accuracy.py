import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLBACKEND", "Agg")
import pickle

import hbmep as mep
import onmep as one
from onmep.agent.multiprocessing import _shutdown_executors
import numpy as np

from constants import (
    SIMULATIONS_DATA_DIR,
    SIMULATIONS_OUT_DIR,
    MAKE_RESULTS_DIR
)

SIMULATIONS_MODEL_NAME = "rl_gamma"

BUILD_DIR = MAKE_RESULTS_DIR
os.makedirs(BUILD_DIR, exist_ok=True)

PRESENT_RESPONSE = ["PKPK_APB", "PKPK_ECR", "PKPK_FCR"]

RUN_IDS = [
    "uniform-mcmc",
    "pest-mle1",
    "eig-mcmc",
    "focused_eig_a-mcmc",
    "evr-mcmc",
]

NS_SPACE = list(range(3, 64 + 1))

N0 = 8

NUM_DRAWS = 2000


def run_one(draw, param, response_idx):
    simulations_data_dir = os.path.join(
        SIMULATIONS_DATA_DIR,
        SIMULATIONS_MODEL_NAME,
        "accuracy_data"
    )
    named_params = [
        mep.site.a,
        mep.site.b,
        mep.site.g,
        mep.site.h,
        mep.site.v
    ]
    try:
        _, post_one, *_ = mep.load(simulations_data_dir, memmap_draw=draw)
        match param:
            case mep.site.a:
                true = post_one[param].reshape(-1,)
                true = true[response_idx][None]     # (1, 3)
            case "mu_int":
                post_one[mep.site.g] *= 0
                xgrid = np.linspace(0, 100, 501)[:, None, None]
                with mep.use_gpu():
                    true = np.asarray(mep.functional.rectified_logistic(
                        xgrid,                                                      # (501, 1, 1)
                        **{
                            u: post_one[u].reshape(-1,)[response_idx][None, None]   # (1, 1, 3)
                            for u in named_params
                        }
                    ))      # (501, 1, 3)
                h = post_one[mep.site.h].reshape(-1,)[response_idx]     # (3,)
            case _:
                raise ValueError
    except Exception as e:
        print(e)
        return None

    errs = []
    try:
        for run_id in RUN_IDS:
            model_dir = os.path.join(
                SIMULATIONS_OUT_DIR,
                SIMULATIONS_MODEL_NAME,
                "accuracy_data/400w_400s_4c_1t_10d_80a_fm",
                f"{'__'.join(np.asarray(PRESENT_RESPONSE)[response_idx].tolist())}",
                "rl_gamma",
                run_id,
                f"d{draw:04d}",
            )
            ckpt_dir = os.path.join(model_dir, "checkpoints")
            if not os.path.exists(ckpt_dir):
                raise FileNotFoundError
            ckpt = one.CheckpointManager(ckpt_dir)
            for ns in NS_SPACE:
                step = N0 + ns
                pay = ckpt.load_step(step)
                post_state = pay["agent_state"].get("posterior_state", None)
                if post_state is None:
                    if len(response_idx) > 1:
                        raise ValueError
                    post = pay["agent_state"].get("posterior")
                    assert post is not None
                    post = one.Posterior.from_dict(post, named_params=named_params).as_dict()
                else:
                    post = one.Posterior.from_state(pay["agent_state"]["posterior_state"]).as_dict()
                match param:
                    case mep.site.a:
                        pred = post[param][:, 0]    # (1600, 3)
                    case "mu_int":
                        post[mep.site.g] *= 0
                        with mep.use_gpu():
                            pred = np.asarray(mep.functional.rectified_logistic(
                                xgrid,  # (501, 1, 3)
                                **{
                                    u: post[u][:, 0][None]  # (1, 1600, 3)
                                    for u in named_params
                                }
                            ))      # (501, 1600, 3)
                    case _:
                        raise ValueError

                err = mep.metrics.lp_risk(
                    true,
                    pred,
                    axis=-2,
                    ignore_nan=True,
                    p=1
                )   # (501, 3)

                if param == "mu_int":
                    dx = xgrid[1] - xgrid[0]
                    dx = dx.item()
                    err = np.trapz(err, dx=dx, axis=0)
                    err = err / h

                errs.append(err)

    except Exception as e:
        print(e)
        return None

    else:
        errs = np.asarray(errs)
        errs = errs.reshape(len(RUN_IDS), len(NS_SPACE), *errs.shape[1:])
        return errs, draw


def main(param, response):
    response_idx = [PRESENT_RESPONSE.index(u) for u in response]
    num_draws = NUM_DRAWS
    tasks = [(d, param, response_idx) for d in range(num_draws)]
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
        response=response,
        simulation_model_name=SIMULATIONS_MODEL_NAME,
    )
    out_path = os.path.join(BUILD_DIR, f"{len(response):02d}_{SIMULATIONS_MODEL_NAME}_{param}_{num_draws:04d}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(d, f)
    return out_path, err.shape


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if not len(args):
        out = []
        response = ["PKPK_APB"]
        out.append(main(mep.site.a, response))
        out.append(main("mu_int", response))
        response = ["PKPK_APB", "PKPK_ECR", "PKPK_FCR"]
        out.append(main(mep.site.a, response))
        out.append(main("mu_int", response))
        for row in out:
            print(f"Saved to {row[0]}, shape {row[1]}")
    else:
        response_len = int(args[0])
        match response_len:
            case 1:
                response = ["PKPK_APB"]
            case 3:
                response = ["PKPK_APB", "PKPK_ECR", "PKPK_FCR"]
            case _:
                raise ValueError
        param = args[1]
        row = main(param, response)
        print(f"Saved to {row[0]}, shape {row[1]}")
