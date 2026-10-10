import os

from paper.constants import REPORTS, TMS_DATA, TMS_TOML

SIMULATIONS_DATA_DIR = os.path.join(REPORTS, "simulations_data")

SIMULATIONS_OUT_DIR = os.path.join(REPORTS, "out_simulations")

BUILD_DIR = os.path.join(REPORTS, "eig_decomposition")

TRAJECTORY_OUT_DIR = os.path.join(BUILD_DIR, "out_trajectory")
