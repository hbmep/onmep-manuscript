import os

import numpy as np
import pandas as pd
import arviz as az
import xarray as xr
from hbmep.util import site

from rtmep.model.simulate import Simulator
from rtmep.notebooks.util import enable_logger, save_model
from rtmep.notebooks.constants import REPORTS, TMS_DATA, TMS_TOML
from learn_posterior import BUILD_DIR


def main(model_dir):
    base_model = Simulator(toml_path=TMS_TOML)
    response = base_model.response

    df = None
    mcmc = None
    mcmc_list = []

    outlier_prob = []
    states_outlier_prob = []
    states_flat_outlier_prob = []

    p = []
    states_p = []
    states_flat_p = []

    for response_name in response:
        subdir = os.path.join(model_dir, response_name)
        ccdf, encoder, ccposterior, ccmodel, ccmc, *_ = load_model(subdir)
        mcmc_list.append(ccmc)

        _, _, _, _, ccmcmc, *_ = load_model(subdir)
        if site.outlier_prob in ccposterior.keys():
            outlier_prob.append(ccposterior.pop(site.outlier_prob)[..., None])
            states_outlier_prob.append(
                ccmcmc._states[ccmcmc._sample_field]
                .pop(site.outlier_prob)[..., None]
            )
            states_flat_outlier_prob.append(
                ccmcmc._states_flat[ccmcmc._sample_field]
                .pop(site.outlier_prob)[..., None]
            )
            if "p" in ccposterior.keys():
                p.append(ccposterior.pop("p"))
                states_p.append(
                    ccmcmc._states[ccmcmc._sample_field].pop("p")
                )
                states_flat_p.append(
                    ccmcmc._states_flat[ccmcmc._sample_field].pop("p")
                )

        if df is None:
            df = ccdf.copy()
            model = ccmodel
            shape = (
                ccposterior[site.a].shape[0],
                *(df[ccmodel.features].to_numpy().max(axis=0) + 1),
                ccmodel.num_response
            )
            assert (
                sorted(list(set([v.ndim for _, v in ccposterior.items()])))
                == [1, len(shape)]
            )

            assert (
                (
                    ccmcmc._states[ccmcmc._sample_field].keys()
                    == ccposterior.keys()
                )
                and (
                    ccmcmc._states_flat[ccmcmc._sample_field].keys()
                    == ccposterior.keys()
                )
            )
            mcmc = ccmcmc

            posterior = {}
            for u, v in ccposterior.items():
                v_states = ccmcmc._states[ccmcmc._sample_field][u].copy()
                v_states_flat = (
                    ccmcmc._states_flat[ccmcmc._sample_field][u].copy()
                )

                if v.ndim != len(shape):
                    v = v[..., None]
                    v_states = v_states[..., None]
                    v_states_flat = v_states_flat[..., None]

                np.testing.assert_almost_equal(v, v_states_flat)
                np.testing.assert_almost_equal(
                    v.reshape(mcmc.num_chains, -1, *v.shape[1:]),
                    v_states
                )
                posterior[u] = v.copy()
                mcmc._states[mcmc._sample_field][u] = v_states
                mcmc._states_flat[mcmc._sample_field][u] = v_states_flat

        else:
            pd.testing.assert_frame_equal(df, ccdf)
            for u in posterior.keys():
                v = ccposterior[u].copy()
                v_states = ccmcmc._states[ccmcmc._sample_field][u].copy()
                v_states_flat = (
                    ccmcmc._states_flat[ccmcmc._sample_field][u].copy()
                )
                if v.ndim != len(shape):
                    v = v[..., None]
                    v_states = v_states[..., None]
                    v_states_flat = v_states_flat[..., None]
                posterior[u] = np.concatenate([posterior[u], v], axis=-1)
                mcmc._states[mcmc._sample_field][u] = np.concatenate(
                    [mcmc._states[mcmc._sample_field][u], v_states],
                    axis=-1
                )
                mcmc._states_flat[mcmc._sample_field][u] = np.concatenate(
                    [mcmc._states_flat[mcmc._sample_field][u], v_states_flat],
                    axis=-1
                )
            for u, v in model.__dict__.items():
                if u in {"build_dir", "_response", "_model"}: continue
                if isinstance(v, list):
                    assert sorted(v) == sorted(ccmodel.__dict__[u])
                else:
                    assert v == ccmodel.__dict__[u]

    model.build_dir = model_dir
    os.makedirs(model.build_dir, exist_ok=True)
    model.response = response

    if len(outlier_prob):
        outlier_prob = np.concatenate(outlier_prob, axis=-1).mean(axis=-1)
        states_outlier_prob = (
            np.concatenate(states_outlier_prob, axis=-1)
            .mean(axis=-1)
        )
        states_flat_outlier_prob = np.concatenate(
            states_flat_outlier_prob, axis=-1
        ).mean(axis=-1)
        posterior[site.outlier_prob] = outlier_prob.copy()
        mcmc._states[mcmc._sample_field][site.outlier_prob] = (
            states_outlier_prob.copy()
        )
        mcmc._states_flat[mcmc._sample_field][site.outlier_prob] = (
            states_flat_outlier_prob.copy()
        )

        if len(p):
            p = np.concatenate(p, axis=-2)
            states_p = np.concatenate(states_p, axis=-2)
            states_flat_p = np.concatenate(states_flat_p, axis=-2)
            posterior["p"] = p.copy()
            mcmc._states[mcmc._sample_field]["p"] = states_p.copy()
            mcmc._states_flat[mcmc._sample_field]["p"] = states_flat_p.copy()

    for u, v in mcmc._states[mcmc._sample_field].items():
        print(u, v.shape)

    for u, v in mcmc_list[0]._states[mcmc_list[0]._sample_field].items():
        print(u, v.shape)

    mcmc._args = (*model.get_regressors(df), *model.get_response(df))
    idata = az.from_numpyro(mcmc)
    loo = az.loo(idata)
    print(loo)

    # Correct loo calculation using inference data instead
    inf_data = [az.from_numpyro(mc) for mc in mcmc_list]
    log_likelihoods = [idata.log_likelihood["obs"] for idata in inf_data]
    combined_ll = xr.concat(log_likelihoods, dim="obs")
    idata_combined = inf_data[0].copy()
    idata_combined.log_likelihood = xr.Dataset({"obs": combined_ll})
    idata_combined.log_likelihood = xr.Dataset({"log_likelihood": combined_ll})
    loo_result = az.loo(idata_combined)
    print(loo_result)

    save_model(model, df, encoder, posterior, mcmc)
    print(f"Saved to {model.build_dir}")
    return


if __name__ == "__main__":
    model_dirs = []
    model_names = [
        "rectified_logistic", "logistic5", "logistic4",
        "rectified_linear", "constln_rlog",
    ]
    for model_name in model_names:
        model_dirs.append(os.path.join(
            BUILD_DIR, "rat", "4000w_4000s_4c_4t_15d_95a_fm", model_name
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
            main(model_dir)
        except Exception as e:
            exceptions.append((model_dir, e))
    if exceptions:
        for model_dir, e in exceptions:
            print(f"Encountered {e} with {model_dir}")
    else:
        print("Ok.")
