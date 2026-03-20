import os
import gc
import json
import pickle
import logging

import numpy as np
import pandas as pd
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from numpyro.infer import MCMC
from hbmep import BaseModel, site

from rtmep.notebooks.constants import FORMAT

logger = logging.getLogger(__name__)

POSTERIOR_DIR = "posterior"
POSTERIOR_KEYS_PKL = "keys.pkl"
POSTERIOR_META_JSON = "meta.json"


def run(data, model, encoder=None, run_kw={}, predict_kw={}):
    # Run
    if encoder is None:
        df, encoder = model.load(df=data)
    else:
        df = data.copy()
    logger.info(f"df.shape {df.shape}")
    mcmc, posterior = model.run(df=df, **run_kw)
    save_model(model, df, posterior, encoder, mcmc)
    predict(
        df, encoder, posterior, model, mcmc, **predict_kw
    )
    return df, encoder, posterior, model, mcmc


def save_model(model, df, posterior, encoder=None, mcmc=None, output_dir=None):
    """
    Save:
      - df & encoder -> data.pkl
      - model        -> model.pkl
      - mcmc (opt)   -> mcmc.pkl
      - posterior    -> posterior/arr_*.npy (per key), keys.pkl, meta.json
    """
    if output_dir is None:
        output_dir = getattr(model, "build_dir", None) or "."
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "data.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((df, encoder), f, protocol=pickle.HIGHEST_PROTOCOL)
    # logger.info(f"Saved df, encoder to {output_path}")

    output_path = os.path.join(output_dir, "model.pkl")
    with open(output_path, "wb") as f:
        pickle.dump((model,), f)
    # logger.info(f"Saved model to {output_path}")

    if mcmc is not None:
        output_path = os.path.join(output_dir, "mcmc.pkl")
        with open(output_path, "wb") as f:
            pickle.dump((mcmc,), f, protocol=pickle.HIGHEST_PROTOCOL)
        # logger.info(f"Saved mcmc to {output_path}")

    if posterior is None:
        return

    posterior_dir = os.path.join(output_dir, POSTERIOR_DIR)
    os.makedirs(posterior_dir, exist_ok=True)

    keys = list(posterior.keys())
    output_path = os.path.join(posterior_dir, POSTERIOR_KEYS_PKL)
    with open(output_path, "wb") as f:
        pickle.dump(keys, f, protocol=pickle.HIGHEST_PROTOCOL)

    items_meta = []
    num_draws = None
    for i, k in enumerate(keys):
        arr = np.asarray(posterior[k])
        if num_draws is None:
            num_draws = arr.shape[0]
        else:
            if arr.shape[0] != num_draws:
                raise ValueError(
                    f"All posterior arrays must share dim0 (#draws); "
                    f"got {num_draws} and {arr.shape[0]} for key {k}"
                )
        fname = f"arr_{i}.npy"
        np.save(os.path.join(posterior_dir, fname), arr)
        items_meta.append({
            "index": i,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "file": fname,
        })

    meta = {
        "num_draws": int(num_draws) if num_draws is not None else 0,
        "num_items": len(items_meta),
        "items": items_meta,
    }
    output_path = os.path.join(posterior_dir, POSTERIOR_META_JSON)
    with open(output_path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    # logger.info(f"Saved posterior arrays to {posterior_dir}")
    logger.info(f"Saved to {output_dir}")
    gc.collect()
    return


def load_model(
    model_dir: str,
    data_file: str = "data.pkl",
    model_file: str = "model.pkl",
    mcmc_file: str = "mcmc.pkl",
    *,
    memmap_draw=None,
) -> list[BaseModel, pd.DataFrame, dict[str, np.ndarray], dict, MCMC]:
    """
    Load the output saved with `save_model`.

    If {model_dir}/posterior/ exists:
      - If memmap_draw is None: load full posterior into RAM (dict of arrays)
      - If memmap_draw is int:  memory-map each array and return ONLY that slice
                                    as shape [1, ...] (no giant RAM allocation).
                                    In this case, mcmc is not loaded

    Returns: (df, encoder, posterior, model, mcmc)
    """
    df, encoder = None, None
    src = os.path.join(model_dir, data_file)
    try:
        with open(src, "rb") as f:
            df, encoder = pickle.load(f)
            if memmap_draw is not None:
                df = df.iloc[[0]].copy()
    except FileNotFoundError:
        logger.info(f"{data_file} not found in {model_dir}")

    model = None
    try:
        with open(os.path.join(model_dir, model_file), "rb") as f:
            model, = pickle.load(f)
    except Exception as e:
        logger.info(f"Model load warning: {e}")

    mcmc = None
    if memmap_draw is None:
        try:
            with open(os.path.join(model_dir, mcmc_file), "rb") as f:
                mcmc, = pickle.load(f)
        except Exception as e:
            logger.info(f"MCMC load info: {e}")

    posterior_dir = os.path.join(model_dir, POSTERIOR_DIR)
    posterior = {}

    if os.path.isdir(posterior_dir):
        src = os.path.join(posterior_dir, POSTERIOR_KEYS_PKL)
        with open(src, "rb") as f:
            keys_list = pickle.load(f)

        src = os.path.join(posterior_dir, POSTERIOR_META_JSON)
        with open(src, "r") as f:
            meta = json.load(f)

        S = int(meta.get("num_draws", 0))
        if memmap_draw is not None:
            if not (0 <= memmap_draw < S):
                raise IndexError(f"memmap_draw {memmap_draw} out of range [0, {S-1}]")

        items = sorted(meta["items"], key=lambda d: d["index"])
        if len(items) != len(keys_list):
            raise ValueError("Posterior keys and items length mismatch.")

        for i, key in enumerate(keys_list):
            fname = items[i]["file"]
            src = os.path.join(posterior_dir, fname)
            if memmap_draw is None:
                posterior[key] = np.load(src, mmap_mode=None)
            else:
                if key not in [site.mu, site.obs]:
                    mm = np.load(src, mmap_mode="r")
                    posterior[key] = mm[memmap_draw:memmap_draw + 1]        # shape (1, ...)

    return model, df, posterior, encoder, mcmc


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


def _adjust_brightness(rgb_in, val):
    hsv = rgb_to_hsv(rgb_in)
    hsv[2] = val  # Adjust the brightness (value component in HSV)
    rgb_out = hsv_to_rgb(hsv)
    return rgb_out


def _get_cmap_muscles_alt():
    vec_muscle = np.array(
        [
            "Trapezius", "Deltoid", "Biceps", "Triceps", "ECR", "FCR", "APB",
            "ADM", "TA", "EDB", "AH", "FDI", "auc_target"
        ]
    )
    cmap_mus_dark = np.array([
        _adjust_brightness(np.array([0.6350, 0.0780, 0.1840]), 0.5),  # trapz
        np.array([1, 133, 113]) / 255,  # delt
        np.array([166, 97, 26]) / 255,  # biceps
        np.array([44, 123, 182]) / 255,  # triceps
        np.array([52, 0, 102]) / 255,  # ecr
        _adjust_brightness(np.array([0.5, 0.5, 0.5]), 0.3),  # fcr
        np.array([208, 28, 139]) / 255,  # apb
        np.array([77, 172, 38]) / 255,  # adm
        np.array([215, 25, 28]) / 255,  # ta
        np.array([123, 50, 148]) / 255,  # edb
        _adjust_brightness(np.array([153, 79, 0]) / 256, 0.4),  # ah
        np.array([231, 226, 61]) / 255,  # fdi
        np.array([255, 100, 0]) / 255,  # auc_target
    ])
    cmap_mus_light = np.array([
        _adjust_brightness(np.array([0.6350, 0.0780, 0.1840]), 0.8),  # trapz
        np.array([128, 205, 193]) / 255,  # delt
        np.array([223, 194, 125]) / 255,  # biceps
        np.array([171, 217, 233]) / 255,  # triceps
        _adjust_brightness(np.array([200, 40, 0]) / 255, 0.6),  # ecr
        _adjust_brightness(np.array([0.5, 0.5, 0.5]), 0.6),  # fcr
        np.array([241, 182, 218]) / 255,  # apb
        np.array([184, 225, 134]) / 255,  # adm
        np.array([253, 174, 97]) / 255,  # ta
        np.array([194, 165, 207]) / 255,  # edb
        _adjust_brightness(np.array([153, 79, 0]) / 256, 0.6),  # ah
        _adjust_brightness(np.array([23, 54, 124]) / 256, 0.6),  # fdi
        np.array([255, 100, 0]) / 255,  # auc_target
    ])
    # Create a DataFrame to hold muscle names and corresponding colors
    T_color = pd.DataFrame({
        'muscle': vec_muscle,
        'cmap_mus_light': [tuple(c) for c in cmap_mus_light],
        'cmap_mus_dark': [tuple(c) for c in cmap_mus_dark],
    })
    # Convert RGB to hex
    T_color['cmap_mus_light_hex'] = T_color['cmap_mus_light'].apply(
        lambda x: '#%02x%02x%02x' % tuple([int(255 * v) for v in x]))
    T_color['cmap_mus_dark_hex'] = T_color['cmap_mus_dark'].apply(
        lambda x: '#%02x%02x%02x' % tuple([int(255 * v) for v in x]))
    return cmap_mus_dark, cmap_mus_light, vec_muscle, T_color


def get_response_colors(response: list[str]):
    (
        cmap_mus_dark, cmap_mus_light, vec_muscle, T_color
    ) = _get_cmap_muscles_alt()
    cmap_dict = dict(zip(vec_muscle, cmap_mus_dark))
    colors = []
    for response in response: colors.append(cmap_dict[response[1:]])
    return colors


def get_subname(model: BaseModel):
    return (
        f'{model.mcmc_params["num_warmup"]}w'
        f'_{model.mcmc_params["num_samples"]}s'
        f'_{model.mcmc_params["num_chains"]}c'
        f'_{model.mcmc_params["thinning"]}t'
        f'_{model.nuts_params["max_tree_depth"][0]}d'
        f'_{model.nuts_params["target_accept_prob"] * 100:.0f}a'
        f'_{"t" if model.use_mixture else "f"}m'
    )
