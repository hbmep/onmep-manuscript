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

RESPONSE = ['PKPK_APB']
SAVE_EVERY = 24

K = 1
USE_FP = False

RUN_ID = (
    # "uniform-mcmc"
    "pest-mle1"
    "eig-mcmc"
)

LO, HI = (0, 2000)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 7:
        run_id = args[0]
        K = int(args[1])
        dlo, dhi, slo, shi = map(int, args[2:6])
        use_fp = bool(int(args[6]))
        draws_range = (dlo, dhi)
        subjects_range = (slo, shi)
    else:
        run_id = RUN_ID
        draws_range = (LO, HI)
        subjects_range = (0, 16)
        use_fp = USE_FP

    target = (16 + 8) * K

    if K == 2:
        feature0_range = None
    elif K == 1:
        feature0_range = (0, 2)
    else:
        raise ValueError

    main(
        draws_range,
        subjects_range,
        feature0_range,
        run_id,
        target=target,
        simulations_data_dir=SIMULATIONS_DATA_DIR,
        simulations_out_dir=SIMULATIONS_OUT_DIR,
        response=RESPONSE,
        save_every=SAVE_EVERY,
        baseagent_kw=BaseAgent_kw(K=K).model_dump(),
        use_fp=use_fp,
    )
