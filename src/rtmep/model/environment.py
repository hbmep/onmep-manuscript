import logging
from abc import ABC, abstractmethod

from jax import random
import numpy as np
from hbmep.model import BaseModel
from hbmep.util import site
from rtmep.model.structures import Observation, ObsArrays
from rtmep.model.agent import BaseRTAgent

from rtmep.notebooks.util import load_model

logger = logging.getLogger(__name__)


class BaseRTEnvironment(ABC):
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

    def interact(self, agent, *, max_steps: int = 64) -> None:
        self.reset()
        agent.reset()
        while len(agent.data) < max_steps:
            x = self.clip(agent.propose())
            obs = self.step(x)
            agent_stop = agent.update(obs)
            if agent_stop or self.if_stop():
                break
        return


class SimulationEnv(BaseRTEnvironment):
    def __init__(
        self,
        *,
        model_dir: str,
        lo: float = float('-inf'),
        hi: float = float('inf'),
    ) -> None:
        super().__init__(lo=lo, hi=hi)
        self.model_dir = model_dir
        df, _, posterior, model, _ = load_model(model_dir)
        posterior.pop(site.obs, None)
        posterior.pop(site.mu, None)

        self.df = df.iloc[[0]].copy()
        self._row_id = self.df.index[0]
        self.model = model
        self.posterior = posterior

        self.num_draws = next(iter(self.posterior.values())).shape[0]
        self._draw = 0
        self._post_one = None
        self._refresh_cached_draw()
        self._key = random.key(0)

    def _refresh_cached_draw(self):
        self._post_one = {k: v[[self._draw]] for k, v in self.posterior.items()}

    @property
    def draw(self) -> int:
        return self._draw

    @draw.setter
    def draw(self, value: int):
        k = int(value)
        if not (0 <= k < self.num_draws):
            raise ValueError(f"draw {k} out of range [0, {self.num_draws-1}]")
        self._draw = k
        self._refresh_cached_draw()

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._key = random.key(seed)
        else:
            self._key = random.key(0)

    def step(self, intensity: float) -> Observation:
        self.df.loc[self._row_id, self.model.intensity] = intensity
        self._key, subkey = random.split(self._key)
        obs = self.model.predict(
            df=self.df,
            posterior=self._post_one,
            return_sites=[site.obs],
            key=subkey,
        )[site.obs].reshape(-1,)
        return Observation(response=obs, intensity=float(intensity), info={"draw": self._draw})
