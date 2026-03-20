import os
# os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
# os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

if "LSLAPICFG" not in os.environ:
    print("LSLAPICFG not found in OS environment.")
    lsl_cfg = os.path.abspath(os.path.join(os.path.dirname(__file__), "lsl_api.cfg"))
    if not os.path.exists(lsl_cfg):
        raise FileNotFoundError(f"CRITICAL: LSL configuration file NOT FOUND at: {lsl_cfg}")
os.environ["LSLAPICFG"] = lsl_cfg

import sys
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
logger = logging.getLogger(__name__)


def cast_recursive(d):
    if isinstance(d, dict):
        return {k: cast_recursive(v) for k, v in d.items()}
    if isinstance(d, list):
        return [cast_recursive(v) for v in d]
    if isinstance(d, str):
        try:
            if "." in d: return float(d)
            return int(d)
        except ValueError:
            if d.lower() == "true": return True
            if d.lower() == "false": return False
            return d
    return d


def build_model(info: dict, agent: BaseAgent | None, ckpt: CheckpointManager):
    data = info["desc"]
    channels = data["channels"]["channel"]
    if isinstance(channels, dict):
        channels = [channels]

    custom = cast_recursive(data.get("custom_configuration", {}))
    if custom is None: custom = {}
    
    # Defaults to True if not specified
    request_reset_str = str(custom.get("request_reset", "true"))
    do_reset = request_reset_str.lower() == "true"
    
    source_id = info.get("source_id", "default_source")
    # Priority: recording_id from custom config > LSL source_id
    source_id = custom.get("recording_id", source_id)
    
    build_dir = os.path.join(BUILD_DIR, source_id)
    build_dir = custom.get("build_dir", build_dir)
    os.makedirs(build_dir, exist_ok=True)
    
    mep.enable_logging(build_dir)
    
    ckpt_dir = os.path.join(build_dir, "checkpoints")
    ckpt = CheckpointManager(ckpt_dir)

    # If we already have an agent and no reset is requested, just keep it
    if agent is not None and not do_reset:
        logger.info("Keeping existing in-memory agent (no reset requested).")
        return agent, ckpt

    # Fresh initialization
    logger.info("Initializing new agent instance...")
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
    # Map n_zeros/n_random from MATLAB to N0/N1 in agent_params
    if "n_zeros" in custom:
        agent_params["N0"] = int(custom["n_zeros"])
    if "n_random" in custom:
        agent_params["N1"] = int(custom["n_random"])
        
    agent_params.update(custom.get("agent_params", {}))

    seed = int(custom.get("seed", int(time.time())))
    key = random.key(seed)

    agent = EIG(
        key=key,
        config=config,
        **agent_params
    )
    agent._model = agent.rl
    agent.build_dir = build_dir

    if not do_reset:
        latest = ckpt.latest()
        if latest is not None:
            logger.info(
                f"Agent object created. "
                f"Resuming from checkpoint step {latest.get('step')} "
                f"is NOT FULLY IMPLEMENTED yet."
            )

    logger.info(f"Agent ready. Source: {source_id}, Reset: {do_reset}")
    return agent, ckpt


def run():
    info = StreamInfo('node_adaptivehbmep', 'Vectors', 1, 0, 'float32', 'eig_node_001')
    outlet = StreamOutlet(info)

    current_inlet = None
    current_uid = None
    current_info = None
    tel = None

    agent = None
    ckpt = None

    posterior_outlet = None
    POSTERIOR_KEYS = [
        mep.site.a,
        mep.site.b,
        mep.site.c1,
        mep.site.c2,
        mep.site.g,
        mep.site.h,
        mep.site.v,
    ]

    TARGET_SOURCE_ID = 'node_stim_uid_fixed_for_testing'

    try:
        while True:
            # Resolve all stimulator streams
            streams = resolve_byprop('name', 'node_stimulator', timeout=1.0)

            if streams:
                # Filter for our specific target stream
                target_streams = [s for s in streams if s.source_id() == TARGET_SOURCE_ID]
                if not target_streams:
                    new_stream = streams[-1]
                else:
                    new_stream = target_streams[-1]
                
                new_uid = new_stream.uid()

                if new_uid != current_uid:
                    logger.info(
                        f"New stream detected (UID: {new_uid}, Source: {new_stream.source_id()})"
                    )
                    if current_inlet:
                        del current_inlet
                        current_inlet = None
                    if tel:
                        tel.close()
                        tel = None
                    
                    if posterior_outlet:
                        logger.info("Closing old posterior outlet...")
                        del posterior_outlet
                        posterior_outlet = None
                    
                    try:
                        current_inlet = StreamInlet(new_stream)
                        current_uid = new_uid
                        current_info = xmltodict.parse(current_inlet.info().as_xml())["info"]

                        while True:
                            stale_sample, _ = current_inlet.pull_sample(timeout=0.0)
                            if stale_sample is None:
                                break

                        agent, ckpt = build_model(current_info, agent, ckpt)
                        timings_path = os.path.join(agent.build_dir, "timings.ndjson")
                        tel = Telemetry(timings_path, mode="ndjson", flush_every=1)

                        # Initialize Posterior Outlet
                        channels = current_info["desc"]["channels"]["channel"]
                        if isinstance(channels, dict): channels = [channels]
                        post_resp = [ch["label"] for ch in channels if ch["type"] == "MEP"]
                        num_post_resp = len(post_resp)
                        num_mcmc_samples = int(
                            agent.mcmc_params["num_chains"] * agent.mcmc_params["num_samples"]
                        )
                        num_post_channels = len(POSTERIOR_KEYS) * num_mcmc_samples * num_post_resp
                        
                        logger.info(
                            f"Creating Posterior Outlet with {num_post_channels} channels "
                            f"({len(POSTERIOR_KEYS)} params * {num_mcmc_samples} samples * "
                            f"{num_post_resp} responses)"
                        )
                        post_info = StreamInfo(
                            'node_adaptivehbmep_posterior',
                            'Posterior',
                            num_post_channels,
                            0,
                            'float32',
                            f'post_{current_uid}'
                        )
                        pdesc = post_info.desc()
                        pdesc.append_child_value("num_samples", str(num_mcmc_samples))
                        pdesc.append_child_value("num_responses", str(num_post_resp))
                        pdesc.append_child_value(
                            "parameter_keys", ","
                            .join(POSTERIOR_KEYS)
                            .replace('₁', '1')
                            .replace('₂', '2')
                        )
                        posterior_outlet = StreamOutlet(post_info)
                    except Exception as e:
                        logger.error(f"Error initializing model for UID {new_uid}: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        current_uid = None  # Force retry next loop

            if current_inlet and agent:
                t_iter0 = _now_ns()
                try:
                    sample, ts = current_inlet.pull_sample(timeout=0.0)
                except Exception as e:
                    logger.warning(f"Error pulling from inlet: {e}")
                    sample = None

                if sample:
                    st = StepTimes(step=len(agent.data) + 1, tags=current_info)
                    
                    # Agent update
                    if any(val > 0 for val in sample):
                        # Parse new observation from hardware
                        intensity_proposed, intensity_used = sample[:2]
                        response = sample[2:-1]
                        new_max = sample[-1] 
                        obs = Observation(
                            intensity=float(intensity_used),
                            response=np.array(response),
                            features=np.array([0])
                        )
                        agent.grid_max = float(new_max)

                        t0 = _now_ns()
                        agent.update(obs)
                        st.t_update_ns = _now_ns() - t0

                        if agent.posterior is not None:
                            logger.info(
                                f"Posterior updated. "
                                f"Transmitting {len(POSTERIOR_KEYS)} parameters..."
                            )
                            # Flatten and send global parameters (excluding mep.site.mu)
                            # Shape of each param: (num_samples, 1, num_responses)
                            # Use Fortran order 'F' to match MATLAB's column-major reshape
                            flat_post = []
                            for k in POSTERIOR_KEYS:
                                v = np.array(agent.posterior[k])
                                flat_post.extend(v.flatten(order='F').astype(np.float32).tolist())
                            
                            if posterior_outlet:
                                logger.info(
                                    f"Pushing posterior sample "
                                    f"({len(flat_post)} elements) to LSL..."
                                )
                                posterior_outlet.push_sample(flat_post)

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
                            extra_meta={"source_id": current_info.get("source_id", "unknown")}
                        )
                        st.t_ckpt_ns = _now_ns() - t0

                    # Network send
                    t0 = _now_ns()
                    outlet.push_sample([float(action[0])])
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
            del current_inlet

        if posterior_outlet:
            del posterior_outlet

        del outlet 
        logger.info("Agent shutdown complete.")


def main():
    logging.basicConfig(level=logging.INFO)
    run()
    return


if __name__ == '__main__':
    main()
