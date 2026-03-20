import os
import logging

import pandas as pd
import hbmep as mep
from rtmep.model.simulate import Simulator

from rtmep.notebooks.util import run
from rtmep.notebooks.constants import REPORTS, TMS_DATA, TMS_TOML

logger = logging.getLogger(__name__)
BUILD_DIR = os.path.join(REPORTS, "accuracy", "learn_posterior")
os.makedirs(BUILD_DIR, exist_ok=True)


def main(data_path):
    df = pd.read_csv(data_path)
    idx = df.participant.apply(lambda x: x.startswith("SCA"))
    df = df[idx].reset_index(drop=True).copy()
    run(df, model, run_kw={"extra_fields": ["num_steps"]})
    return


if __name__ == "__main__":
    model = Simulator(toml_path=TMS_TOML)
    model.simulate = False
    model.use_mixture = False
    model._model = model.hb_rl
    model.features = ["participant"]
    model.build_dir = BUILD_DIR
    model.mcmc_params = {
        "thinning": 4,
        "num_chains": 20,
        "num_warmup": 4000,
        "num_samples": 4000,
    }
    model.nuts_params = {
        "max_tree_depth": (15, 15),
        "target_accept_prob": .95,
    }

    response_id = int(input("Enter response id:"))
    if response_id != -1:
        model.response = model.response[response_id:response_id + 1]
        model.build_dir = os.path.join(model.build_dir, model.response[0])

    mep.enable_logging(model.build_dir)
    main(TMS_DATA)
