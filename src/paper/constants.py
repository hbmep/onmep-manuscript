import os

HOME = os.getenv("HOME")
# Point this to hbmep-adaptive directory
REPO = os.path.join(HOME, "repos", "hbmep-adaptive")
# Point this to directory where datasets are stored
DATA_DIR = os.path.join(HOME, "data", "hbmep-paper-dataset")
# Point this to output directory
REPORTS = os.path.join(HOME, "reports", "hbmep-adaptive")

TMS_DATA = os.path.join(DATA_DIR, "human_tms.csv")
TMS_TOML = os.path.join(REPO, "configs", "TMS.toml")

FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
