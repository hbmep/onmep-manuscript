import os
# os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
# os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import xmltodict
import logging
import time

import matplotlib as mpl
os.environ.setdefault("MPLBACKEND", "Agg")
mpl.use("Agg", force=True)

import hbmep as mep
import numpy as np
from jax import random
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop

from rtmep.agent import BaseAgent
from rtmep.agent.eig import EIG
from rtmep import Observation
from rtmep.telemetry import Telemetry, StepTimes, _now_ns
from rtmep import CheckpointManager

from rtmep.notebooks.constants import REPORTS

BUILD_DIR = os.path.join(REPORTS, "lsl")
# BUILD_DIR = os.path.join(BUILD_DIR, time.strftime("%Y%m%d-%H%M%S"))
logger = logging.getLogger(__name__)


def build_model(info: dict, agent: BaseAgent | None, ckpt: CheckpointManager):
    data = info["desc"]
    channels = data["channels"]["channel"]
    if isinstance(channels, dict):
        channels = [channels]

    custom = data.get("custom_configuration")
    do_reset = custom.get("request_reset", "true") == "true"
    do_reset = do_reset or (agent is None)

    if do_reset:
        # if agent:
        #     out = os.path.join(agent.build_dir, f"{agent.data.n:02d}.pdf")
        #     agent.plot_curves(output_path=out)

        intensity = "intensity_used"
        features = ["participant"]
        response = [ch["label"] for ch in channels if ch["type"] == "MEP"]
        variables = dict(
            intensity=intensity,
            features=features,
            response=response
        )

        mcmc_params = dict(
            thinning=1, num_chains=4, num_warmup=400,
            num_samples=400, progress_bar=False,
        )
        mcmc_params.update(custom.get("mcmc_params", {}))

        nuts_params = dict(target_accept_prob=.8, max_tree_depth=(10, 10))
        nuts_params.update(custom.get("nuts_params", {}))

        config = dict(
            variables=variables,
            mcmc=mcmc_params,
            nuts=nuts_params
        )

        agent_params = dict(
            infer="mcmc", initial_capacity=36, K=1, N0=8, N1=3,
            grid_lo=0., grid_hi=100., grid_max=100, grid_n=100,
            num_outer=None, batch_size=51, max_step=None,
            policy="eig", update_every=1
        )
        agent_params.update(custom.get("agent_params", {}))

        seed = custom.get("seed", int(time.time()))
        key = random.key(seed)

        agent = EIG(
            key=key,
            config=config,
            **agent_params
        )
        agent._model = agent.rl

        source_id = info.get("source_id")
        build_dir = os.path.join(BUILD_DIR, source_id)
        build_dir = custom.get("build_dir", build_dir)
        os.makedirs(build_dir, exist_ok=True)
        agent.build_dir = build_dir
        ckpt_dir = os.path.join(agent.build_dir, "checkpoints")
        ckpt = CheckpointManager(ckpt_dir)

        logger.info(f"Agent built from source_id {source_id}")
        logger.info(f"Build dir: {build_dir}")
        logger.info(f"Checkpoints: {ckpt_dir}")

    return agent, ckpt


def run():
    info = StreamInfo('node_adaptivehbmep', 'Vectors', 1, 0, 'float32', 'eig_node_001')
    outlet = StreamOutlet(info)

    current_inlet = None
    current_source_id = None
    current_info = None
    tel = None

    agent = None
    ckpt = None
    mep.enable_logging(BUILD_DIR)

    try:
        while True:
            streams = resolve_byprop('name', 'node_stimulator', timeout=1.0)

            if streams:
                new_stream = streams[0]
                new_source_id = new_stream.source_id()

                if new_source_id != current_source_id:
                    if current_inlet:
                        current_inlet.close_stream()
                    if tel:
                        tel.close()
                    
                    current_inlet = StreamInlet(new_stream)
                    current_source_id = new_source_id
                    current_info = xmltodict.parse(current_inlet.info().as_xml())["info"]

                    while True:
                        stale_sample, _ = current_inlet.pull_sample(timeout=0.0)
                        if stale_sample is None:
                            break

                    try:
                        agent, ckpt = build_model(current_info, agent, ckpt)
                        timings_path = os.path.join(agent.build_dir, "timings.ndjson")
                        tel = Telemetry(timings_path, mode="ndjson", flush_every=1)
                    except Exception as e:
                        logger.error(f"Error on new source id {new_source_id}: {e}")

            if current_inlet and agent:
                t_iter0 = _now_ns()
                sample, ts = current_inlet.pull_sample(timeout=0.0)

                if sample:
                    st = StepTimes(step=len(agent.data) + 1, tags=current_info)
                    # logger.info(f"Received {np.round(sample, 2).tolist()}")

                    # Agent update
                    if any(val > 0 for val in sample):
                        # Parse new observation from hardware
                        intensity_proposed, intensity_used = sample[:2]
                        response = sample[2:-1]
                        new_max = sample[:-1]
                        obs = Observation(
                            intensity=intensity_used,
                            response=response,
                            features=np.array([0])
                        )
                        agent.grid_max = new_max

                        t0 = _now_ns()
                        agent.update(obs)
                        st.t_update_ns = _now_ns() - t0

                    # Agent propose
                    t0 = _now_ns()
                    action = agent.propose()
                    st.t_propose_ns = _now_ns() - t0

                    if ckpt:
                        t0 = _now_ns()
                        ckpt.save(
                            step=len(agent.data),
                            agent_state=agent.state_dict(),
                            env_state={},
                            extra_meta={"source_id": current_source_id}
                        )
                        st.t_ckpt_ns = _now_ns() - t0

                    # Network send
                    t0 = _now_ns()
                    outlet.push_sample([action[0]])
                    st.t_step_ns = _now_ns() - t0
    
                    st.t_iter_ns = _now_ns() - t_iter0
                    logger.info(
                        f"num_samples: {len(agent.data):02d}/??, "
                        f"propose: {st.t_propose_ns / 1e9:.3f}s, "
                        f"update: {st.t_update_ns / 1e9:.3f}s, "
                        f"iter: {st.t_iter_ns / 1e9:.3f}s"
                    )

                    if tel:
                        tel.add(st)

            time.sleep(0.01)
    
    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Cleaning up ...")

    finally:
        if tel:
            tel.close()

        if current_inlet:
            current_inlet.close_stream()

        del outlet 
        logger.info("Agent shutdown complete.")


def main():
    run()
    return


if __name__ == '__main__':
    main()
