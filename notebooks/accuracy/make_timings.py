import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import pickle

import hbmep as mep
import numpy as np

from constants import (
    TIMINGS_OUT_DIR,
    MAKE_RESULTS_DIR,
)

BUILD_DIR = MAKE_RESULTS_DIR
os.makedirs(BUILD_DIR, exist_ok=True)
# TIMINGS_OUT_DIR = "/home/vishu/reports/hbmep-adaptive/out_simulations/main_timings/"

TIMING_METRICS = [
    "t_update_ns",
    "t_propose_ns",
    "t_iter_ns",
]

N0 = 8
NS_SPACE = list(range(3, 64 + 1))

NUM_DRAWS = 500
BS = 28

RUN_IDS = [
    "eig-mcmc",
    "focused_eig_a-mcmc",
    "focused_eig_abh-mcmc",
    "evr-mcmc",
]


def run_one(draw, response_len):
    times = []
    try:
        for run_id in RUN_IDS:
            model_dir = os.path.join(
                TIMINGS_OUT_DIR,
                f"r{response_len:02d}",
                run_id,
                f"d{draw:04d}",
            )
            ckpt_dir = os.path.join(model_dir, "checkpoints")
            if not os.path.exists(ckpt_dir):
                raise FileNotFoundError
            src = os.path.join(model_dir, "timings.ndjson")
            with open(src, "r") as f:
                timings = {
                    row["step"]: row
                    for row in map(json.loads, f)
                }
            for ns in NS_SPACE:
                step = N0 + ns
                for metric in TIMING_METRICS:
                    times.append(timings[step][metric])
    except Exception as e:
        print(e)
        return None
    else:
        times = np.asarray(times)
        times = times.reshape(
            len(RUN_IDS),
            len(NS_SPACE),
            len(TIMING_METRICS),
            *times.shape[1:],
        )
        return times, draw


def main(response_len):
    tasks = [(d, response_len) for d in range(NUM_DRAWS)]
    out = mep.run_batched(run_one, tasks=tasks, batch_size=BS)
    times, draws = zip(*out)
    times = np.asarray(times)
    draws = np.asarray(draws)
    times[..., NS_SPACE.index(3), :] = times[..., NS_SPACE.index(4), :]
    out = dict(
        times=times,
        draws=draws,
        response_len=response_len,
        run_ids=RUN_IDS,
        metrics=TIMING_METRICS,
        ns_space=NS_SPACE,
    )
    output_path = os.path.join(BUILD_DIR, f"times_r{response_len:02d}.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(out, f)
    return output_path, times


if __name__ == "__main__":
    out = []
    out.append(main(1))
    out.append(main(3))
    for row in out:
        print(f"Saved to {row[0]}, shape {row[1].shape}")
