import os

from rtmep.notebooks.constants import REPORTS

LEARN_POSTERIOR_DIR = os.path.join(REPORTS, "accuracy", "learn_posterior")
TP_DIR = os.path.join(LEARN_POSTERIOR_DIR, "simulate_power_data_tp")
FP_DIR = os.path.join(LEARN_POSTERIOR_DIR, "simulate_power_data_fp")

POWER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(LEARN_POSTERIOR_DIR)),
    "power"
)
