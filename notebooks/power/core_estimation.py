import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""     # comment out to use GPU
os.environ.setdefault("MPLBACKEND", "Agg")
import sys
import logging
import traceback

import hbmep as mep
import onmep as one
import pandas as pd
from jax import random

N0 = 8
N = 16

from model import HBestimation
from default_kw import NUTS_kw, MCMC_kw
from constants import SIMULATIONS_OUT_DIR, REPORTS

logger = logging.getLogger(__name__)

BUILD_DIR = os.path.join(REPORTS, "out_estimation")
os.makedirs(BUILD_DIR, exist_ok=True)


def safe_run_one(draw, run_id, step, num_subjects, **kw):
    try:
        run_one(draw, run_id, step, num_subjects, **kw)
        return draw, run_id, step, num_subjects, None
    except Exception:
        return draw, run_id, step, num_subjects, traceback.format_exc()


def run_one(
    draw,
    run_id,
    step,
    num_subjects,
    *,
    source_dir,
    build_dir,
):
    read_run_id = run_id
    use_k2 = False

    if run_id.endswith("-k2"):
        read_run_id = run_id[:-3]
        use_k2 = True

    model_dir = os.path.join(
        source_dir,
        "400w_400s_4c_1t_10d_80a_fm",
        "PKPK_APB",
        "rl_gamma",
        read_run_id,
        f"d{draw:04d}",
    )

    data = []
    for subject in range(num_subjects):
        subject_data = []
        if use_k2:
            ckpt_dir = os.path.join(
                model_dir,
                f"s{subject:02d}",
                "k2",
                "checkpoints"
            )
            ckpt = one.CheckpointManager(ckpt_dir)
            if not os.path.exists(ckpt_dir):
                raise FileNotFoundError
            payload = ckpt.load_step(2 * step)
            model = one.BaseAgent().load_state_dict(payload["agent_state"])
            assert model.data.n == 2 * step
            df = model.data.to_df(**model.variables)
            df["participant"] = subject
            subject_data.append(df)
        else:
            for f0 in range(2):
                ckpt_dir = os.path.join(
                    model_dir,
                    f"s{subject:02d}",
                    f"k1_f{f0}",
                    "checkpoints"
                )
                ckpt = one.CheckpointManager(ckpt_dir)
                if not os.path.exists(ckpt_dir):
                    raise FileNotFoundError
                payload = ckpt.load_step(step)
                model = one.BaseAgent().load_state_dict(payload["agent_state"])
                assert model.data.n == step
                df = model.data.to_df(**model.variables)
                df["participant"] = subject
                df["intervention"] = f0
                subject_data.append(df)
        data += subject_data

    df = pd.concat(data, ignore_index=True).reset_index(drop=True).copy()
    assert df.shape[0] == num_subjects * step * 2

    mcmc_kw = MCMC_kw(num_warmup=2000, num_samples=1000, progress_bar=True).model_dump()
    nuts_kw = NUTS_kw(target_accept_prob=.95, max_tree_depth=(12, 12)).model_dump()
    variables = dict(
        intensity="TMSInt",
        features=["participant", "intervention"],
        response=["PKPK_APB"]
    )
    config = dict(mcmc_kw=mcmc_kw, nuts_kw=nuts_kw, variables=variables)
    model = HBestimation(config=config)

    key = random.key(0)
    key = random.fold_in(key, draw)
    key = random.fold_in(key, num_subjects)
    model.key = key

    model.build_dir = os.path.join(
        build_dir,
        run_id,
        f"d{draw:04d}",
        f"n{num_subjects:02d}"
    )
    mep.enable_logging(model.build_dir)

    _, post = model.run(df=df)
    mep.save(
        model_state=model.state_dict(),
        posterior=post,
        output_dir=model.build_dir
    )

    if draw < 10:
        pred_df = model.make_prediction_dataset(df)
        ppd = model.predict(pred_df, posterior=post)
        model.plot_curves(
            df=df,
            prediction_df=pred_df,
            predictive=ppd,
            posterior=ppd,
        )

    return


def main(
    draws_range: tuple[int, int],
    num_subjects_space: list,
    run_id,
    *,
    step,
    simulations_out_dir,
    use_fp,
):
    source_dir = os.path.join(simulations_out_dir, "rl_gamma")
    build_dir = os.path.join(BUILD_DIR, f"step_{step:02d}")

    if use_fp:
        source_dir = os.path.join(source_dir, "power_data_fp")
        build_dir = os.path.join(build_dir, "power_data_fp")
    else:
        source_dir = os.path.join(source_dir, "power_data_tp")
        build_dir = os.path.join(build_dir, "power_data_tp")

    os.makedirs(build_dir, exist_ok=True)

    tasks = [
        (draw, run_id, step, num_subjects)
        for draw in range(*draws_range)
        for num_subjects in num_subjects_space
    ]

    for task in tasks:
        *_, err = safe_run_one(
            *task,
            source_dir=source_dir,
            build_dir=build_dir,
        )
        if err is None:
            print(f"draw {task[0]}, run_id={task[1]}, step={task[2]}, num_subjects={task[3]} ok.")
        else:
            print(f"draw {task[0]}, run_id={task[1]}, step={task[2]}, num_subjects={task[3]} FAILED.")
            print(err)
            output_path = os.path.join(build_dir, f"failed_{task[0]}_{task[1]}_{task[2]}_{task[3]}.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("Simulation failed\n")
                f.write("=================\n\n")
                f.write(f"draw: {task[0]}\n")
                f.write(f"run_id: {task[1]}\n")
                f.write(f"step: {task[2]}\n")
                f.write(f"num_subjects: {task[3]}\n")
                f.write(f"use_fp: {use_fp}\n")
                f.write(f"source_dir: {source_dir}\n")
                f.write(f"build_dir: {build_dir}\n")
                f.write("\nTraceback\n")
                f.write("=========\n\n")
                f.write(err)

    return


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 6:
        run_id = args[0]
        dlo = int(args[1])
        dhi = int(args[2])
        num_subjects_space = [
            int(x) for x in args[3].split(",")
        ]
        step = int(args[4])
        use_fp = bool(int(args[5]))
    else:
        dlo, dhi = 0, 10
        num_subjects_space = [6, 8, 12]
        run_id = "eig-mcmc-k2"
        step = N0 + N
        use_fp = False

    main(
        draws_range=(dlo, dhi),
        num_subjects_space=num_subjects_space,
        run_id=run_id,
        step=step,
        simulations_out_dir=SIMULATIONS_OUT_DIR,
        use_fp=use_fp,
    )
