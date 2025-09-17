import os
import psutil
from joblib import Parallel, delayed
import numpy as np
from jax import random

from hbmep import site
from rtmep.model.environment import SimulationEnv

from rtmep.notebooks.agent import RTAgent
from rtmep.notebooks.util import save_model, enable_logger
from rtmep.notebooks.constants import TMS_TOML
from learn_posterior import BUILD_DIR as LEARN_POSTERIOR_DIR
from simulate_data import BUILD_DIR as MODEL_DIR

BUILD_DIR = os.path.join(LEARN_POSTERIOR_DIR, "out")
os.makedirs(BUILD_DIR, exist_ok=True)

NUM_CORES = psutil.cpu_count(logical=False)
NUM_CHAINS = 4
NUM_JOBS = max(1, NUM_CORES // NUM_CHAINS)
# NUM_JOBS = 1

RUN_ID = "random"


def run_one_draw(run_id, draw_idx: int, *, out_dir: str):
    assert run_id in {
        "random",
    }

    model = RTAgent(toml_path=TMS_TOML, max_observations=24)
    model._model = model.rl
    model.features = []
    model.mcmc_params = dict(thinning=1, num_chains=4, num_warmup=2000, num_samples=1000)
    model.nuts_params = dict(max_tree_depth=(12, 12), target_accept_prob=0.95)

    match run_id:
        case "random":
            model.propose = model.random_proposal
            model.update = model.update_on_max_obs
        case _:
            raise ValueError(f"Invalid run_id {run_id}")
    
    draw_id = f"draw_{draw_idx:04d}"
    model.build_dir = os.path.join(out_dir, run_id, draw_id)
    enable_logger(model.build_dir)

    env = SimulationEnv(model_dir=MODEL_DIR)
    env.draw = draw_idx

    # Run one episode
    env.interact(agent=model)
    mcmc, posterior = model.mcmc, model.posterior

    # Save to disk
    data = model.data
    df = data.to_dataframe(intensity=model.intensity, response=model.response)
    save_model(model=model, df=df, encoder=None, posterior=posterior, mcmc=None)
    # prediction_df = model.make_prediction_dataset(df)
    # predictive = model.predict(df=prediction_df, posterior=posterior)
    # model.plot_curves(
    #     df=df,
    #     prediction_df=prediction_df,
    #     predictive=predictive,
    #     posterior=posterior,
    #     output_path=os.path.join(out_dir, draw_id + ".pdf")
    # )

    a_true = env._post_one[site.a].reshape(-1,)[None, :]
    a_pred = posterior[site.a]
    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(model.build_dir, "a.npz"),
        a_true=a_true, a_pred=a_pred
    )
    return


def run(run_id, draws, out_dir, n_jobs=NUM_JOBS, verbose=10):
    Parallel(n_jobs=n_jobs, backend="loky", verbose=verbose)(
        delayed(run_one_draw)(run_id, d, out_dir=out_dir) for d in draws
    )
    return


def main():
    OUT = BUILD_DIR
    draws = list(range(0, 1000))
    draws = list(range(588, 1000))
    run(RUN_ID, draws, OUT, n_jobs=NUM_JOBS)
    print("Done")


if __name__ == "__main__":
    main()
