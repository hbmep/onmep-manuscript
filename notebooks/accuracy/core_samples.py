import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""     # comment out to use GPU
os.environ.setdefault("MPLBACKEND", "Agg")
import sys
import traceback
import logging
import pickle

import hbmep as mep
import onmep as one
from onmep.agent.multiprocessing import _shutdown_executors
import numpy as np
from jax import random

import pandas as pd

from paper.model.model_samples import HBestimation

from constants import (
    SIMULATIONS_DATA_DIR,
    MAKE_RESULTS_DIR,
    SAMPLES_DIR,
)

logger = logging.getLogger(__name__)

BUILD_DIR = SAMPLES_DIR
os.makedirs(BUILD_DIR, exist_ok=True)

TOTAL_SUBJECTS = 32
MODEL_NAME = "broken_power_decay"

KEEP_RUN_IDS = [
    'eig-mcmc',
    'focused_eig_a-mcmc',
    'pest-mle1',
    'uniform-mcmc',
]

NUM_SUBJECTS_SPACE = [16]

USE_RELATIVE = True


def make_out_dir(
    *,
    base_dir,
    response,
    simulations_model_name,
    model_name,
):
    build_dir = os.path.join(
        base_dir,
        simulations_model_name,
        "accuracy_data",
        "__".join(response),
        model_name,
    )
    return build_dir


def make_df_wide(
    err,
    space,
    model_names,
    source_draws,
):
    P, M, S = err.shape
    assert P == len(source_draws)
    assert M == len(model_names)
    assert S == len(space)

    arr = np.moveaxis(err, 1, -1)   # (P, S, M)
    arr = arr.reshape(P * S, M)

    df = pd.DataFrame(arr, columns=model_names)

    df.insert(
        0,
        "num_samples",
        np.tile(space, P),
    )
    df.insert(
        0,
        "source_draw",
        np.repeat(source_draws, S)
    )
    df.insert(
        0,
        "participant",
        np.repeat(np.arange(P), S),
    )

    x = df["num_samples"].to_numpy(dtype=float)
    if (~np.isfinite(x) | (x <= 0)).any():
        raise ValueError("Bad num_samples before log")

    y = df[model_names].to_numpy(dtype=float)
    bad_y = ~np.isfinite(y) | (y <= 0)

    if bad_y.any():
        bad_rows, _ = np.where(bad_y)
        print(df.iloc[np.unique(bad_rows)])
        raise ValueError("Bad error before log")

    df["log_num_samples"] = np.log2(x)

    for col in model_names:
        df[f"log_error__{col}"] = np.log2(
            df[col].to_numpy(dtype=float)
        )

    df["participant"] = [
        f"{int(p):02d}__{int(sd)}"
        for p, sd in zip(df["participant"], df["source_draw"])
    ]

    return df


def run_one(draw, num_subjects, param, response_len):
    src = os.path.join(MAKE_RESULTS_DIR, f"{response_len:02d}_rl_gamma_{param}_2000.pkl")
    with open(src, "rb") as f:
        out = pickle.load(f)
    assert len(out["draws"]) == 2000
    assert out["draws"].tolist() == sorted(out["draws"])
    assert out["ns_space"] == list(range(3, 64 + 1))

    named_params = [mep.site.a, mep.site.b, mep.site.g, mep.site.h, mep.site.v]

    if USE_RELATIVE:
        src = os.path.join(SIMULATIONS_DATA_DIR, "rl_gamma", "accuracy_data")
        _, post, *_ = mep.load(src)
        draws = out["draws"]
        post = {u: v[draws][..., :response_len] for u, v in post.items() if u in named_params}
        post = {u: v[:, 0][:, None, None] for u, v in post.items() if u in named_params}

        if param == mep.site.a:
            t = out["err"]
            t = t / post[mep.site.a]
            t = t * 100
            out["err"] = t

        elif param == "mu_int":
            t = out["err"]
            t = t / (100 - post[mep.site.a])
            t = t * 100
            out["err"] = t

        else:
            raise ValueError

    err = out["err"]
    run_ids = out["run_ids"]
    ns_space = out["ns_space"]

    idx = [run_ids.index(u) for u in KEEP_RUN_IDS]
    err = err[:, idx]
    run_ids = KEEP_RUN_IDS

    err = np.mean(err, axis=-1, keepdims=False)

    key = random.key(1729)
    key = random.fold_in(key, draw)
    key, sub = random.split(key)
    idx = np.asarray(random.choice(
        sub,
        np.arange(err.shape[0]),
        shape=(TOTAL_SUBJECTS,),
        replace=False,
    ))
    idx = idx[:num_subjects]

    err = err[idx]          # (P, num_ids, ns_space)
    source_draws = out["draws"][idx]

    df = make_df_wide(
        err=err,
        space=ns_space,
        model_names=run_ids,
        source_draws=source_draws,
    )

    variables = dict(
        intensity="log_num_samples",
        features=["participant"],
        response=[f"log_error__{u}" for u in run_ids],
    )
    mcmc_kw = dict(
        num_chains=4,
        num_warmup=4000,
        num_samples=4000,
        thinning=4,
    )
    nuts_kw = dict(
        max_tree_depth=(12, 12),
        target_accept_prob=.95
    )
    config = dict(variables=variables, mcmc_kw=mcmc_kw, nuts_kw=nuts_kw,)

    model = HBestimation(config=config)
    model._model = getattr(model, MODEL_NAME)

    if USE_RELATIVE:
        if param == mep.site.a:
            cutoff = 2
        elif param == "mu_int":
            cutoff = 10
        else:
            raise ValueError
    else:
        if param == mep.site.a:
            cutoff = 1
        elif param == "mu_int":
            cutoff = 8
        else:
            raise ValueError

    model.log_error_threshold = np.log2(cutoff)
    model.build_dir = os.path.join(
        BUILD_DIR, 
        f"r{response_len:02d}",
        param,
        f"d{draw:04}",
        f"n{num_subjects:02d}"
    )
    mep.enable_logging(model.build_dir)

    df, encoder = model.process(df, mask_non_positive=False)
    mcmc, post = model.run(df=df)

    if draw < 50:
        pred_df = model.make_prediction_dataset(df)
        ppd = model.predict(pred_df, posterior=post)
        model.plot_curves(
            df=df,
            prediction_df=pred_df,
            predictive=ppd,
            posterior=post,
            predictive_hdi_prob=.95,
            encoder=encoder,
        )

    model_state = model.state_dict()
    mep.save(
        model_state=model_state,
        posterior=post,
        df=df,
        output_dir=model.build_dir,
        encoder=encoder,
    )

    if draw < 50:
        if mep.site.outlier_prob in post.keys():
            post.pop(mep.site.outlier_prob)
        summary_df = model.summary(post)
        logger.info(f"Summary:\n{summary_df.to_string()}")
        dest = os.path.join(model.build_dir, "summary.csv")
        summary_df.to_csv(dest)
        logger.info(f"Saved summary to {dest}")
        logger.info(f"Finished running {model._model.__name__}")
        try:
            divergences = mcmc.get_extra_fields()["diverging"].sum().item()
            logger.info(f"No. of divergences {divergences}")
            num_steps = mcmc.get_extra_fields()["num_steps"]
            tree_depth = np.floor(np.log2(num_steps)).astype(int)
            logger.info(f"Tree depth statistics:")
            logger.info(f"Min: {tree_depth.min()}")
            logger.info(f"Max: {tree_depth.max()}")
            logger.info(f"Mean: {tree_depth.mean()}")
        except: pass

    logger.info(f"Saved results to {model.build_dir}")
    return


def safe_run_one(*args, **kw):
    try:
        run_one(*args, **kw)
        return None
    except Exception:
        return traceback.format_exc()
    finally:
        _shutdown_executors()


def main(param, draws_range, response_len):
    tasks = [
        (d, ns, param, response_len)
        for d in range(*draws_range)
        for ns in NUM_SUBJECTS_SPACE
    ]

    for task in tasks:
        err = safe_run_one(*task)
        if err is None:
            print(f"{', '.join(map(str, task))} ok.")
        else:
            print(f"{', '.join(map(str, task))} FAILED")
            print(err)
            output_path = os.path.join(
                BUILD_DIR,
                f"failed_{task[3]}{task[2]}_{task[0]}_{task[1]}.txt",
            )
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"Task: {task}\n\n")
                f.write(err)

    return


if __name__ == "__main__":
    PARAM = "mu_int"
    LO, HI = 0, 1
    RESPONSE_LEN = 1

    args = sys.argv[1:]
    if not len(args):
        param = PARAM
        draws_range = (LO, HI)
        response_len = RESPONSE_LEN
    else:
        param = args[0]
        draws_range = tuple(map(int, args[1:3]))
        response_len = int(args[3])

    main(param, draws_range, response_len)
