import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import logging

import pandas as pd
import hbmep as mep

from paper.model import Simulator
from paper.util import predict

from constants import (
    TMS_DATA,
    TMS_TOML,
    LEARN_POSTERIOR_DIR as BUILD_DIR,
)

logger = logging.getLogger(__name__)
os.makedirs(BUILD_DIR, exist_ok=True)

MODEL_NAME = "rl_gamma"
# MODEL_NAME = "rl_lognormal"


@mep.timing
def main(data_path, **kw):
    df = pd.read_csv(data_path)
    idx = df.participant.apply(lambda x: x.startswith("SCA"))
    df = df[idx].reset_index(drop=True).copy()

    df, encoder = model.process(df=df)
    logger.info(f"df.shape {df.shape}")
    mcmc, posterior = model.run(df=df, **kw)

    model_state = model.state_dict()
    output_dir = model.build_dir
    mep.save(
        model_state=model_state,
        posterior=posterior,
        df=df,
        encoder=encoder,
        mcmc=mcmc,
        output_dir=output_dir
    )

    predict(
        df=df,
        encoder=encoder,
        posterior=posterior,
        model=model,
        mcmc=mcmc,
        **kw
    )
    return


if __name__  == "__main__":
    model = Simulator(toml_path=TMS_TOML)
    model.simulate = False
    model.use_mixture = False
    model.features = ["participant"]
    model.build_dir = BUILD_DIR

    model._model = getattr(model, MODEL_NAME)
    model.build_dir = os.path.join(model.build_dir, model._model.__name__)

    model.mcmc_kw = {
        "thinning": 10,
        "num_chains": 4,
        "num_warmup": 40_000,
        "num_samples": 40_000,
    }
    model.nuts_kw = {
        "max_tree_depth": (15, 15),
        "target_accept_prob": .95,
    }

    mep.enable_logging(model.build_dir)
    main(TMS_DATA)
