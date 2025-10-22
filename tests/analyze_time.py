import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from collections import defaultdict

import numpy as np
from scipy import stats
from joblib import Parallel, delayed

from rtmep import CheckpointManager
from rtmep.agent import BaseAgent

SAMPLES = list(range(3, 64))


def process_draw(draw: int, base_dir: str):
    ckpt_dir = os.path.join(
        base_dir,
        f"draw_{draw:04d}",
        "checkpoints",
    )
    ckpt = CheckpointManager(ckpt_dir)
    model = BaseAgent()
    times = defaultdict(list)

    try:
        for num_samples in SAMPLES:
            try:
                payload = ckpt.load_step(num_samples)
            except Exception:
                times = None
                break
            finally:
                state = payload["agent_state"]
                model.load_state_dict(state)
                ns = model._last_run_ns
                for metric, val in ns.items():
                    times[metric].append(float(val) / 1_000_000_000.0)
    except Exception:
        pass
    
    return draw, times


def nansem(a, axis=0):
    return stats.sem(a, axis=axis, nan_policy='omit')


if __name__ == "__main__":
    draws_space = range(200)
    base_dir = "/home/vishu/reports/gpu/hbmep-adaptive/tests/time/response_PKPK_APB__PKPK_ECR__PKPK_FCR/eig-svi"
    n_jobs = -1

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_draw)(draw, base_dir) for draw in draws_space
    )

    metrics = defaultdict(list)
    num_draws = 0
    for draw, times_dict in results:
        if not times_dict:
            continue
        for u, v in times_dict.items():
            metrics[u].append(v)
        num_draws += 1
    
    for u, v in metrics.items():
        v = np.array(v)
        metrics[u] = v

    for j, num_samples in enumerate(SAMPLES):
        output_fields = [f"#{num_samples}"]
        for metric, arr in metrics.items():
            me = np.nanmean(arr[:, j])
            se = nansem(arr[:, j])
            output_fields.append(f"[{metric}, {me:.2f}, {se:.2f}]")
        print(", ".join(output_fields))

    print(0)
