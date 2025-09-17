import os
import logging

from joblib import Parallel, delayed
from hbmep.util import setup_logging, site

from simulate.model import HB
from paper.util import load_model, predict
from constants import BUILD_DIR

logger = logging.getLogger(__name__)


def run(model_dir, **kw):
    setup_logging(os.path.join(model_dir, "predict.log"))
    logger.info(f"Predicting for {model_dir}...")
    try:
        df, encoder, posterior, model, mcmc = load_model(model_dir, **kw)
    except FileNotFoundError:
        logger.info(f"Model not found in {model_dir}")
        return
    except Exception as e:
        raise e
    else:
        model.build_dir = model_dir
        predict(df, encoder, posterior, model, mcmc)
    return


def main():
    model_dirs = []
    run_ids = ["rat", "tms", "intraoperative"]
    model_names = [
        "rectified_logistic", "logistic5", "logistic4",
        "rectified_linear", "constln_rlog",
    ]
    for run_id in run_ids:
        for model_name in model_names:
            model_dirs.append(os.path.join(
                BUILD_DIR, run_id, "4000w_4000s_4c_4t_15d_95a_fm", model_name
            ))
            if model_name in {"rectified_logistic", "constln_rlog"}:
                # Mixture extension
                model_dirs.append(os.path.join(
                    BUILD_DIR, "rat", "4000w_4000s_4c_4t_15d_95a_tm", model_name
                ))
    
    exceptions = []
    for model_dir in model_dirs:
        print(model_dir)
        try:
            run(model_dir)
        except Exception as e:
            exceptions.append((model_dir, e))
    if exceptions:
        for model_dir, e in exceptions:
            print(f"Encountered {e} with {model_dir}")
    else:
        print("Ok.")
    return


if __name__ == "__main__":
    main()
