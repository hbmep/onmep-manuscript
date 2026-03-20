import os
# os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
# os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time
import random
import uuid
import numpy as np
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop, FOREVER
from dataclasses import dataclass

RESET_ON_CONNECT = True


@dataclass
class ChannelConfig:
    label: str
    unit: str
    type: str


def stimulate():
    print("  >> HARDWARE STIMULATION TRIGGERED <<")


def main():
    # 1. SETUP OUTLET
    source_id = f"node_stim_uid_{uuid.uuid4()}"

    vec_entries = [
        ChannelConfig(label="intensity_proposed", unit="MSOPCT", type="Intensity"),
        ChannelConfig(label="intensity_used", unit="MSOPCT", type="Intensity"),
        ChannelConfig(label="ECR", unit="uVs", type="MEP"),
        ChannelConfig(label="FCR", unit="uVs", type="MEP"),
        ChannelConfig(label="APB", unit="uVs", type="MEP"),
        ChannelConfig(label="MAX", unit="MSOPCT", type="Config"),
    ]

    info = StreamInfo('node_stimulator', 'Vectors', len(vec_entries), 0, 'float32', source_id)
    desc = info.desc()
    channels = info.desc().append_child("channels")

    for entry in vec_entries:
        ch = channels.append_child("channel")
        ch.append_child_value("label", entry.label)
        ch.append_child_value("unit", entry.unit)
        ch.append_child_value("type", entry.type)

    config = desc.append_child("custom_configuration")
    config.append_child_value("participant_id", "P01")
    config.append_child_value("request_algorithm", "EIG")
    config.append_child_value("request_model", "TMS")
    config.append_child_value("request_reset", str(RESET_ON_CONNECT).lower())

    outlet = StreamOutlet(info)
    print(f"Node Stimulator: Outlet created (ID: {source_id}).")

    # 2. SETUP INLET
    print("Node Stimulator: Looking for Node adaptivehbmep...")
    streams = resolve_byprop('name', 'node_adaptivehbmep')
    inlet = StreamInlet(streams[0])
    print("Node Stimulator: Connected.")

    # --- HANDSHAKE LOOP ---
    print("Node Stimulator: Attempting handshake...")
    dummy_vector = [-1.0] * len(vec_entries)

    connected = False
    sample = None

    while not connected:
        outlet.push_sample(dummy_vector)
        print("Node Stimulator: Sent handshake trigger. Waiting for reply...")

        # We capture 'sample' here. This IS the first parameter set.
        sample, timestamp = inlet.pull_sample(timeout=1.0)

        if sample:
            print(f"Node Stimulator: Handshake successful! Received: {sample}")
            connected = True
        else:
            print("Node Stimulator: No reply yet. Retrying...")

    # 3. CONTROL LOOP
    min_interval = 5.0
    last_stim_time = 0.0  # Set to 0 so we don't wait 5s on the very first run

    # Flag to ensure we process the handshake sample before pulling a new one
    first_run = True

    try:
        while True:
            # Step A: Get Data (Skip this step on first run, we already have the sample)
            if not first_run:
                print("Node Stimulator: Waiting for next parameters...")
                sample, timestamp = inlet.pull_sample(timeout=FOREVER)

            # Turn off flag so subsequent loops act normally
            first_run = False

            # Step B: Process Data
            if sample:
                # Enforce minimum interval
                time_since_last = time.time() - last_stim_time
                wait_time = min_interval - time_since_last
                if wait_time > 0:
                    time.sleep(wait_time)

                stimulate()
                # time.sleep(0.1) # Optional hardware delay

                last_stim_time = time.time()

                # Generate and Send Response
                response_vector = [random.uniform(0, 100) for _ in range(len(vec_entries) - 1)]
                response_vector = response_vector + [95.0]

                outlet.push_sample(response_vector)
                print(f"Node Stimulator Sent Results: {np.round(response_vector, 2)}")
    
    except KeyboardInterrupt:
        print("Hardware stopping...")

    finally:
        if inlet:
            inlet.close_stream()

        del outlet 


if __name__ == '__main__':
    main()
    