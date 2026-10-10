import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""     # comment out to use GPU
os.environ.setdefault("MPLBACKEND", "Agg")
import pickle

import hbmep as mep
import numpy as np
from numpyro.diagnostics import hpdi

from make_accuracy import RUN_IDS
from constants import (
    SAMPLES_OUT_DIR,
    MAKE_SAMPLES_DIR,
)

BUILD_DIR = MAKE_SAMPLES_DIR
os.makedirs(BUILD_DIR, exist_ok=True)

PARAMS = [mep.site.a, "mu_int"]
NUM_SUBJECTS_SPACE = [2, 4, 8, 12, 16]

NUM_BOOTSTRAPS = 500
BATCH_SIZE = 32


def process(bootstrap):
    a = []
    arr = []
    try:
        for num_subjects in NUM_SUBJECTS_SPACE:
            for param in PARAMS:
                model_dir = os.path.join(
                    SAMPLES_OUT_DIR,
                    param,
                    f"b{bootstrap:04d}",
                    f"n{num_subjects:02d}",
                )
                _, post, *_ = mep.load(model_dir)
                a_fixed_loc = post["a_fixed_loc"]
                a_delta_loc = post["a_delta_loc"]
                curr = np.concatenate([a_fixed_loc[:, None], a_fixed_loc[:, None] + a_delta_loc], axis=-1)
                a.append(curr)
                arr.append(a_delta_loc)

    except Exception as e:
        return None

    else:
        a = np.asarray(a)
        a = a.reshape(len(NUM_SUBJECTS_SPACE), len(PARAMS), *a.shape[1:])
        arr = np.asarray(arr)
        arr = arr.reshape(len(NUM_SUBJECTS_SPACE), len(PARAMS), *arr.shape[1:])
        return a, arr, bootstrap


def main():
    tasks = [(b,) for b in range(NUM_BOOTSTRAPS)]
    out = mep.run_batched(process, tasks=tasks, batch_size=BATCH_SIZE)
    a, delta, boots = zip(*out)
    a = np.asarray(a)
    delta = np.asarray(delta)
    boots = np.asarray(boots)
    print(a.shape)
    print(delta.shape)
    print(boots.shape)

    # Two-sided test at alpha = 0.05
    hdi_two_sided = hpdi(
        delta,
        prob=.95,
        axis=-2,
    )
    reject_two_sided = (
        (hdi_two_sided[..., 0, :] > 0)
        |
        (hdi_two_sided[..., 1, :] < 0)
    )
    print(hdi_two_sided.shape, reject_two_sided.shape)

    # One-sided test at alpha = 0.05.
    # Positive delta means the comparator requires more samples than EIG.
    prob_positive = np.mean(delta > 0, axis=-2)
    reject_one_sided = prob_positive >= .95
    print(prob_positive.shape, reject_one_sided.shape)

    out = os.path.join(BUILD_DIR, f"samples.pkl")
    d = dict(
        run_ids=RUN_IDS,
        comparison_run_ids=RUN_IDS[1:],
        a=a,
        delta=delta,
        hdi_two_sided=hdi_two_sided,
        reject_two_sided=reject_two_sided,
        prob_positive=prob_positive,
        reject_one_sided=reject_one_sided,
        boots=boots,
        num_subjects_space=NUM_SUBJECTS_SPACE,
    )
    with open(out, "wb") as f:
        pickle.dump(d, f)
    print(f"Saved to {out}")
    return out, a, delta, boots


if __name__ == "__main__":
    out = []
    out.append(main())
    for row in out:
        print(row[0], row[1].shape, row[2].shape)
