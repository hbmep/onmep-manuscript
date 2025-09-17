import os
import pickle
import logging

import numpy as np
from hbmep.util import site

from rtmep.notebooks.constants import FORMAT

logger = logging.getLogger(__name__)


def enable_logger(output, *, level=logging.INFO, format=FORMAT):
    root, ext = os.path.splitext(output)
    if not ext: output_file = os.path.join(output, "logs.log")
    else: output_file = output

    # Create the directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        format=format,
        level=level,
        handlers=[
            logging.FileHandler(output_file, mode="w"),
            logging.StreamHandler()
        ],
        force=True
    )
    logger.info(f"Logging to {output_file}")


def run(data, model, encoder=None, run_kw={}, predict_kw={}):
    # Run
    if encoder is None:
        df, encoder = model.load(df=data)
    else:
        df = data.copy()
    logger.info(f"df.shape {df.shape}")
    mcmc, posterior = model.run(df=df, **run_kw)
    save_model(model, df, encoder, posterior, mcmc)
    predict(
        df, encoder, posterior, model, mcmc, **predict_kw
    )
    return df, encoder, posterior, model, mcmc


def save_model(model, df, encoder, posterior, mcmc=None, output_dir=None):
    if output_dir is None:
        output_dir = model.build_dir

    output_path = os.path.join(output_dir, "inf.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((df, encoder, posterior,), f)
    logger.info(f"Saved to {output_path}")

    output_path = os.path.join(output_dir, "model.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((model,), f)
    logger.info(f"Saved to {output_path}")

    output_path = os.path.join(output_dir, "model_dict.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((model.__dict__,), f)
    logger.info(f"Saved to {output_path}")

    if mcmc is not None:
        output_path = os.path.join(output_dir, "mcmc.pkl")
        with open(output_path, "wb") as f:
            pickle.dump((mcmc,), f)
        logger.info(f"Saved to {output_path}")
    return


def load_model(
    model_dir,
    inference_file="inf.pkl",
    model_file="model.pkl",
    mcmc_file="mcmc.pkl",
):
    src = os.path.join(model_dir, inference_file)
    with open(src, "rb") as f:
        df, encoder, posterior, = pickle.load(f)

    model = None
    try:
        src = os.path.join(model_dir, model_file)
        with open(src, "rb") as f:
            model, = pickle.load(f)
    except ModuleNotFoundError as e:
        logger.info(e)

    mcmc = None
    try:
        src = os.path.join(model_dir, mcmc_file)
        with open(src, "rb") as f:
            mcmc, = pickle.load(f)
    except FileNotFoundError:
        logger.info(
            f"{mcmc_file} not found. Attempting to read from model_dict.pkl"
        )
    except ValueError as e:
        logger.info("Encountered ValueError, trace is below")
        logger.info(e)
    else:
        logger.info(f"Found {model_file}")

    if mcmc is None:
        try:
            src = os.path.join(model_dir, "model_dict.pkl")
            with open(src, "rb") as f:
                mcmc, _ = pickle.load(f)
        except FileNotFoundError:
            logger.info("model_dict.pkl not found.")
        except ValueError as e:
            logger.info(
                "Encountered ValueError, trace is below."
                + " Possible issue with unpacking."
            )
            logger.info(e)
        else:
            logger.info("Found model_dict.pkl")

    return df, encoder, posterior, model, mcmc


def predict(df, encoder, posterior, model, mcmc, **kw):
    # Predictions
    prediction_df = model.make_prediction_dataset(df=df)
    if site.outlier_prob in posterior.keys():
        posterior[site.outlier_prob] = 0 * posterior[site.outlier_prob]
    predictive = model.predict(prediction_df, posterior=posterior, **kw)
    model.plot_curves(
        df=df,
        prediction_df=prediction_df,
        predictive=predictive,
        posterior=posterior,
        encoder=encoder,
    )

    if site.outlier_prob in posterior.keys():
        posterior.pop(site.outlier_prob)
    summary_df = model.summary(posterior)
    logger.info(f"Summary:\n{summary_df.to_string()}")
    dest = os.path.join(model.build_dir, "summary.csv")
    summary_df.to_csv(dest)
    logger.info(f"Saved summary to {dest}")
    logger.info(f"Finished running {model._model.__name__}")
    try:
        divergences = mcmc.get_extra_fields()["diverging"].sum().item()
        logger.info(f"No. of divergences {divergences}")
        num_steps = mcmc.get_extra_fields()["num_steps"]
        tree_depth = np.floor(np.log2(num_steps)).astype(int)
        logger.info(f"Tree depth statistics:")
        logger.info(f"Min: {tree_depth.min()}")
        logger.info(f"Max: {tree_depth.max()}")
        logger.info(f"Mean: {tree_depth.mean()}")
    except: pass
    logger.info(f"Saved results to {model.build_dir}")
    return
