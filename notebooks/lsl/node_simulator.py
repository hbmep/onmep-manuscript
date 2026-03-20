import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
lsl_cfg = os.path.abspath(os.path.join(os.path.dirname(__file__), "lsl_api.cfg"))
if not os.path.exists(lsl_cfg):
    raise FileNotFoundError(f"CRITICAL: LSL configuration file NOT FOUND at: {lsl_cfg}")
os.environ["LSLAPICFG"] = lsl_cfg
import time
import uuid
from dataclasses import dataclass

import numpy as np
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop
from rtmep.environment import SimulationEnv

from rtmep.notebooks.constants import REPORTS

RESET_ON_CONNECT = True
MODEL_DIR = "/home/vishu/reports/hbmep-adaptive/accuracy/learn_posterior/simulate_data"
DRAW_TO_SIMULATE = 1
PARTICIPANT_ID = 0
RESPONSE = [
    'PKPK_APB',
    'PKPK_ECR',
    'PKPK_FCR'
]

BUILD_DIR = os.path.join(REPORTS, "lsl-simulations", "__".join(RESPONSE))
os.makedirs(BUILD_DIR, exist_ok=True)


@dataclass
class ChannelConfig:
    label: str
    unit: str
    type: str


def main(draw, num_observations=None):
    env = SimulationEnv(model_dir=MODEL_DIR, draw=draw, participant=PARTICIPANT_ID)
    env.response = RESPONSE

    source_id = f"sim_node_{uuid.uuid4().hex[:8]}"
    vec_entries = [
        ChannelConfig(label="intensity_proposed", unit="MSOPCT", type="Intensity"),
        ChannelConfig(label="intensity_used", unit="MSOPCT", type="Intensity"),
    ]
    vec_entries += [
        ChannelConfig(label=response, unit="uVs", type="MEP")
        for response in RESPONSE
    ]
    vec_entries += [
        ChannelConfig(label="MAX", unit="MSOPCT", type="Config"),
    ]

    info = StreamInfo('node_stimulator', 'Vectors', len(vec_entries), 0, 'float32', source_id)
    channels = info.desc().append_child("channels")
    for entry in vec_entries:
        ch = channels.append_child("channel")
        ch.append_child_value("label", entry.label)
        ch.append_child_value("type", entry.type)

    config = info.desc().append_child("custom_configuration")
    config.append_child_value("request_reset", str(RESET_ON_CONNECT).lower())
    sim_build_dir = os.path.join(BUILD_DIR, f"draw_{draw:04d}")
    config.append_child_value("build_dir", sim_build_dir)

    outlet = StreamOutlet(info)
    print(f"Hardware: Outlet created (ID: {source_id})")

    print("Hardware: Looking for Agent...")
    streams = resolve_byprop('name', 'node_adaptivehbmep')
    inlet = StreamInlet(streams[0])

    dummy_vector = [-1.0] * len(vec_entries)
    while True:
        outlet.push_sample(dummy_vector)
        sample, ts = inlet.pull_sample(timeout=1.0)
        if sample: 
            print("Hardware: Handshake successful!")
            break

    num_samples = 0
    first_run = True

    try:
        while True:
            if not first_run:
                # sample, ts = inlet.pull_sample(timeout=FOREVER)
                # Removing FOREVER to respect ctrl + C
                sample = None
                while sample is None:
                    sample, ts = inlet.pull_sample(timeout=0.1)
            first_run = False

            if sample:
                # Agent sends [intensity]
                intensity = sample[0]
                action = (intensity, np.array([0])) 
                
                out_of = "??" if num_observations is None else f"{num_observations}"
                print(f"[{draw} | {num_samples+1:02d}/{out_of}] Intensity: {intensity:.2f}")
                observation = env.step(action)
                
                response = observation.response
                response_vector = [
                    intensity,
                    intensity,
                    *response,
                    100.0
                ]

                outlet.push_sample(response_vector)
                num_samples += 1
                if num_observations is not None and num_samples >= num_observations:
                    print(f"Draw {draw} done.")
                    sys.exit(0)
                
    except KeyboardInterrupt:
        print("Hardware stopping...")

    finally:
        if inlet:
            inlet.close_stream()

        del outlet


if __name__ == '__main__':
    args = list(map(int, sys.argv[1:]))
    try:
        draw = args[0]
    except:
        draw = DRAW_TO_SIMULATE

    try:
        num_observations = args[1]
    except:
        num_observations = None

    main(draw, num_observations=num_observations)
