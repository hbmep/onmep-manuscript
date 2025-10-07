import numpy as np
from jax import random, numpy as jnp
from hbmep.util import site
from rtmep import Observation
from rtmep.environment.environment import BaseEnvironment

from rtmep.notebooks.util import load_model


class SimulationEnv(BaseEnvironment):
    def __init__(
        self,
        *,
        model_dir: str,
        draw: int = 0,
        lo: float = float("-inf"),
        hi: float = float("inf"),
    ) -> None:
        super().__init__(lo=lo, hi=hi)
        self.model_dir = model_dir
        self._key = random.key(0)
        self._init_draw(draw)
        self.response = None

    def _init_draw(self, draw):
        model, df, post_one, *_ = load_model(self.model_dir, memmap_draw=draw)
        self.df = df.iloc[[0]].copy()
        self._row_id = self.df.index[0]
        self.model = model
        post_one.pop(site.obs, None)
        post_one.pop(site.mu, None)
        self._post_one = post_one
        self._draw = int(draw)

    @property
    def draw(self) -> int:
        return self._draw

    @draw.setter
    def draw(self, value: int):
        self._init_draw(value)

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._key = random.key(seed)
        else:
            self._key = random.key(0)

    def step(self, intensity: float) -> Observation:
        intensity = float(intensity)
        self.df.loc[self._row_id, self.model.intensity] = intensity
        self._key, subkey = random.split(self._key)
        obs = self.model.predict(
            df=self.df,
            posterior=self._post_one,
            return_sites=[site.obs],
            key=subkey,
        )[site.obs].reshape(-1,)
        if self.response is not None:
            mapping = dict(zip(self.model.response, obs))
            obs = np.array([mapping[res] for res in self.response])
        return Observation(response=obs, intensity=intensity, info={"draw": self._draw})

    def state_dict(self) -> dict:
        key_data = random.key_data(self._key)
        return {
            "type": self.__class__.__name__,
            "lo": self.lo, "hi": self.hi,
            "model_dir": self.model_dir,
            "draw": self._draw,
            "key_data": key_data.tolist(),
            "key_dtype": str(key_data.dtype.name),
        }

    def load_state_dict(self, state: dict):
        self.lo, self.hi = float(state["lo"]), float(state["hi"])
        self.model_dir = state["model_dir"]
        self._init_draw(int(state["draw"]))
        key_data = state.get("key_data", [0, 0])
        key_dtype = jnp.dtype(state.get("key_dtype", jnp.uint32))
        self._key = random.wrap_key_data(jnp.array(key_data, dtype=key_dtype))
