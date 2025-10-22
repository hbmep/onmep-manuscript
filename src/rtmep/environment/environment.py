import logging
from abc import ABC, abstractmethod

import numpy as np
from hbmep.util import timing
from rtmep import CheckpointManager
from rtmep import Observation
from rtmep.agent.base_agent import BaseRTAgent
from rtmep.telemetry import Telemetry, StepTimes, _now_ns

logger = logging.getLogger(__name__)


class BaseEnvironment(ABC):
    def __init__(self, *, lo=float("-inf"), hi=float("inf")):
        self.lo = float(lo)
        self.hi = float(hi)

    def reset(self) -> None:
        ...
    
    def clip(self, intensity: float) -> float:
        return float(np.clip(float(intensity), self.lo, self.hi))

    def if_stop(self) -> bool:
        return False

    @abstractmethod
    def step(self, intensity: float) -> Observation:
        ...

    def state_dict(self) -> dict:
        return {"type": self.__class__.__name__}

    def load_state_dict(self, state: dict):
        return

    @timing
    def interact(
        self,
        agent: BaseRTAgent,
        *,
        max_observations: int,                  # target number of observations
        ckpt: CheckpointManager | None = None,
        save_every: int = 1,
        resume: bool = False,                       
        resume_from: int | str | None = None,   # step number or file path
        timings_out: str | None = None,         # e.g., "/.../reports/run_123/timings.ndjson"
        timings_mode: str = "ndjson",           # "ndjson" or "csv"
        timings_flush_every: int = 50,          # flush buffer every N steps (ndjson)
        timings_truncate_on_start: bool | None = None,
        timing_tags: dict | None = None,        # attached to each row (e.g., {"agent":"EIG", "draw":7})
    ):
        """
        Run until len(agent.data) == max_observations, checkpointing at every
        `save_every` iteration.
        Resume rules:
          - If `resume_from` is int -> load that step.
          - If `resume_from` is str -> load that file path.
          - Else if `resume` and ckpt available -> load latest step.
        """
        if ckpt:
            payload = None
            try:
                if isinstance(resume_from, int):
                    payload = ckpt.load_step(resume_from)
                    logging.info(f"Resuming from checkpoint step={resume_from}")
                elif isinstance(resume_from, str):
                    payload = ckpt.load(resume_from)
                    logging.info(f"Resuming from checkpoint file: {resume_from}")
                elif resume:
                    latest = ckpt.load_latest()
                    if latest is not None:
                        payload = latest
                        logging.info(f"Resuming from latest checkpoint step={latest.get('step')}")
                if payload is not None:
                    agent.load_state_dict(payload["agent_state"])
                    self.load_state_dict(payload["env_state"])
            except FileNotFoundError as e:
                logging.warning(f"{e}. Starting fresh.")
            except Exception as e:
                logging.warning(f"Failed to resume ({e}). Starting fresh.")

        if len(agent.data) >= max_observations:
            logging.info(
                f"Already at {len(agent.data)} ≥ target {max_observations}; nothing to do."
            )
            return

        auto_truncate = not (resume or isinstance(resume_from, (int, str)))
        if timings_truncate_on_start is None:
            timings_truncate_on_start = auto_truncate
        tel = Telemetry(
            timings_out,
            timings_mode,
            timings_flush_every,
            timings_truncate_on_start,
        ) if timings_out else None
        tags = dict(timing_tags or {})

        try:
            # Episode loop
            while len(agent.data) < max_observations:
                st = StepTimes(step=len(agent.data) + 1, tags=tags.copy())
                t_iter0 = _now_ns()

                t0 = _now_ns()
                x = agent.propose()
                st.t_propose_ns = _now_ns() - t0

                t0 = _now_ns()
                x = self.clip(x)
                obs = self.step(x)
                st.t_step_ns = _now_ns() - t0

                t0 = _now_ns()
                agent_stop = agent.update(obs)
                st.t_update_ns = _now_ns() - t0
                st.agent_stop = bool(agent_stop)

                # Checkpoint after each successful append
                if ckpt and (len(agent.data) % save_every == 0):
                    t0 = _now_ns()
                    ckpt.save(
                        step=len(agent.data),
                        agent_state=agent.state_dict(),
                        env_state=self.state_dict(),
                        extra_meta={"target": int(max_observations)},
                    )
                    st.t_ckpt_ns = _now_ns() - t0

                st.t_iter_ns = _now_ns() - t_iter0
                logger.info(
                    f"num_samples: {len(agent.data)}/{max_observations}, "
                    f"propose: {st.t_propose_ns / 1_000_000_000:.2f}s, "
                    f"update: {st.t_update_ns / 1_000_000_000:.2f}s, "
                    f"iter: {st.t_iter_ns / 1_000_000_000:.2f}s, "
                )

                if tel:
                    tel.add(st)

                if agent_stop or self.if_stop():
                    break

        finally:
            if tel:
                tel.close()

        return
