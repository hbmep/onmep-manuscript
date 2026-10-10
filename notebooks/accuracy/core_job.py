import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""     # comment out to use GPU
os.environ.setdefault("MPLBACKEND", "Agg")
import sys
import logging

import hbmep as mep

from core import main
from constants import (
    SIMULATIONS_DATA_DIR,
    SIMULATIONS_OUT_DIR,
)
from default_kw import (
    BaseAgent_kw,
)

logger = logging.getLogger(__name__)

N0 = 8
N = 64
SAVE_EVERY = 1

RESPONSE = [
    'PKPK_APB',
    # 'PKPK_ECR',
    # 'PKPK_FCR'
]

SIMULATIONS_MODEL_NAME, MODEL_NAME = (
    "rl_gamma",
    "rl_gamma"
)

RUN_ID = (
    # "uniform-mcmc"
    # "pest-mle1"
    # "pest-mle2"
    "eig-mcmc"
    # "focused_eig_a-mcmc"
    # "focused_eig_abh-mcmc"
    # "evr-mcmc"
)

LO, HI = (0, 2000)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 3:
        run_id = args[0]
        dlo, dhi = map(int, args[1:])
        draws_range = (dlo, dhi)
    else:
        run_id = RUN_ID
        draws_range = (LO, HI)

    target = N0 + N

    main(
        draws_range,
        run_id,
        target=target,
        simulations_data_dir=SIMULATIONS_DATA_DIR,
        simulations_model_name=SIMULATIONS_MODEL_NAME,
        model_name=MODEL_NAME,
        simulations_out_dir=SIMULATIONS_OUT_DIR,
        response=RESPONSE,
        save_every=SAVE_EVERY,
        baseagent_kw=BaseAgent_kw().model_dump(),
    )
