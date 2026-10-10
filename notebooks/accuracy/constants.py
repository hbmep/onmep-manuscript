import os

from paper.constants import REPORTS, TMS_DATA, TMS_TOML

LEARN_POSTERIOR_DIR = os.path.join(REPORTS, "learn_posterior")

SIMULATIONS_DATA_DIR = os.path.join(REPORTS, "simulations_data")

SIMULATIONS_OUT_DIR = os.path.join(REPORTS, "out_simulations")

TIMINGS_OUT_DIR = os.path.join(REPORTS, "out_time")

SIM_RESPONSE = [
    'PKPK_APB',
    'PKPK_ECR',
    'PKPK_FCR'
]

MAKE_RESULTS_DIR = os.path.join(REPORTS, "make_results")

SAMPLES_DIR = os.path.join(REPORTS, "out_samples")
