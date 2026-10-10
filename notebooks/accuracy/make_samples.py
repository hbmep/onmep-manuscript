import os
import pickle

import hbmep as mep
import numpy as np

from core_samples import KEEP_RUN_IDS
from constants import SAMPLES_DIR

BUILD_DIR = os.path.join(SAMPLES_DIR, "make_samples")
os.makedirs(BUILD_DIR, exist_ok=True)
# SAMPLES_DIR = "/home/vishu/reports/c2b2-Jun-2026/out_samples"

NUM_SUBJECTS = 16
NUM_DRAWS = 500
BS = 32


def read_response(response_len):
    def process(draw):
        try:
            a = []
            for param in [mep.site.a, "mu_int"]:
                model_dir = os.path.join(
                    SAMPLES_DIR,
                    f"r{response_len:02d}",
                    param,
                    f"d{draw:04d}",
                    f"n{NUM_SUBJECTS:02d}"
                )
                if not os.path.exists(model_dir):
                    raise FileNotFoundError
                _, post, *_ = mep.load(model_dir)
                a_fixed_loc = post["a_fixed_loc"]
                a_delta_loc = post["a_delta_loc"]
                curr = np.concatenate([a_fixed_loc[:, None], a_fixed_loc[:, None] + a_delta_loc], axis=-1)
                a.append(curr[None])
        except Exception as e:
            return None
        a = np.concatenate(a, axis=0)
        return a, draw

    tasks = [(draw,) for draw in range(NUM_DRAWS)]
    out = mep.run_batched(process, tasks, batch_size=BS)

    a, draws = zip(*out)
    a = np.array(a)
    draws = np.asarray(draws)
    return a, draws


def save_response(response_len):
    a, draws = read_response(response_len)
    out = os.path.join(BUILD_DIR, f"{response_len:02d}_samples.pkl")
    d = dict(
        a=a,
        draws=draws,
        response_len=response_len,
        run_ids=KEEP_RUN_IDS,
    )
    with open(out, "wb") as f:
        pickle.dump(d, f)
    print(f"Saved to {out}")
    return out, a, draws


out = []
out.append(save_response(1))
out.append(save_response(3))
for row in out:
    print(row[0], row[1].shape)
