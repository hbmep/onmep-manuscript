import os
import logging

import hbmep as mep
import numpy as np
import pandas as pd
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

logger = logging.getLogger(__name__)


def predict(df, encoder, posterior, model: mep.BaseModel, mcmc, **kw):
    # Predictions
    prediction_df = model.make_prediction_dataset(df=df)
    if mep.site.outlier_prob in posterior.keys():
        posterior[mep.site.outlier_prob] = 0 * posterior[mep.site.outlier_prob]
    predictive = model.predict(prediction_df, posterior=posterior, **kw)
    model.plot_curves(
        df=df,
        prediction_df=prediction_df,
        predictive=predictive,
        posterior=posterior,
        encoder=encoder,
    )

    if mep.site.outlier_prob in posterior.keys():
        posterior.pop(mep.site.outlier_prob)
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


def get_subname(model: mep.BaseModel):
    return (
        f'{model.mcmc_kw["num_warmup"]}w'
        f'_{model.mcmc_kw["num_samples"]}s'
        f'_{model.mcmc_kw["num_chains"]}c'
        f'_{model.mcmc_kw["thinning"]}t'
        f'_{model.nuts_kw["max_tree_depth"][0]}d'
        f'_{model.nuts_kw["target_accept_prob"] * 100:.0f}a'
        f'_{"t" if model.use_mixture else "f"}m'
    )
