import logging
from abc import ABC, abstractmethod

import numpy as np
from rtmep import CheckpointManager
from rtmep import Observation
from rtmep.agent.agent import BaseRTAgent

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

    def interact(
        self,
        agent: BaseRTAgent,
        *,
        max_observations: int,                      # target number of observations
        ckpt: CheckpointManager | None = None,
        save_every: int = 1,
        resume: bool = False,                       
        resume_from: int | str | None = None,       # step number or file path
    ) -> int:
        """
        Run until len(agent.data) == max_observations, checkpointing at every
        `save_every` iteration.
        Resume rules:
          - If `resume_from` is int -> load that step.
          - If `resume_from` is str -> load that file path.
          - Else if `resume` and ckpt available -> load latest step.
        Returns the final number of observations.
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
            return len(agent.data)

        # Initialize only if empty (don't wipe resumed state)
        if len(agent.data) == 0:
            self.reset()
            agent.reset()

        # Episode loop
        while len(agent.data) < max_observations:
            x = agent.propose()
            x = self.clip(x)
            obs = self.step(x)
            agent_stop = agent.update(obs)

            # Checkpoint after each successful append
            if ckpt and (len(agent.data) % save_every == 0):
                ckpt.save(
                    step=len(agent.data),
                    agent_state=agent.state_dict(),
                    env_state=self.state_dict(),
                    extra_meta={"target": int(max_observations)},
                )

            if agent_stop or self.if_stop():
                break

        return len(agent.data)
