import os

from paper.constants import REPORTS

BUILD_DIR = os.path.join(REPORTS, "validation")
DATA_DIR = os.path.join(BUILD_DIR, "make_data")
CKPT_DIR = os.path.join(BUILD_DIR, "make_ckpts")

BOOTSTRAP_DATA_DIR = os.path.join(BUILD_DIR, "bootstrap_data")
BOOTSTRAP_OUT_DIR = os.path.join(BUILD_DIR, "out_bootstrap")
ACCURACY_OUT_DIR = os.path.join(BUILD_DIR, "out_accuracy")
SAMPLES_OUT_DIR = os.path.join(BUILD_DIR, "out_samples")
MAKE_SAMPLES_DIR = os.path.join(BUILD_DIR, "make_samples")

SUBJECTS = [
    "S01P04_s01r01_2026-05-14T194845",
    "S01P02_s01r01_2026-05-09T172532",
    "S01P00_s01r01_2026-05-08T121359",
    "S01P09_s02r01_2026-06-19T150720",
    "S01P06_s01r01_2026-05-18T175628",
    "S01P03_s01r01_2026-05-11T151707",
    "S01P08_s01r01_2026-05-26T175530",
    "S01P05_s01r01_2026-05-15T093940",
]

NUM_POINTS = 10000
