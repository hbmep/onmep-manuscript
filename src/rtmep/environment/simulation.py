import hbmep as mep
import numpy as np
from jax import random, numpy as jnp

from rtmep import Observation
from rtmep.environment.environment import BaseEnvironment

from rtmep.notebooks.util import load_model


class SimulationEnv(BaseEnvironment):
    def __init__(
        self, *, model_dir: str, draw: int = 0, participant: int = 0,
        lo=float("-inf"), hi=float("inf")
    ):
        super().__init__(lo=lo, hi=hi)
        self.model: mep.BaseModel | None = None
        self.model_dir = model_dir
        self._key = random.key(0)

        self.df = None
        self._row_id = None
        self._post_one = None
        self.response = None

        self._draw = None
        self._participant = None
        self._init_draw(draw, participant)
        return

    def _init_draw(self, draw, participant):
        model, df, post_one, *_ = load_model(self.model_dir, memmap_draw=draw)

        self.df = df.iloc[[0]].copy()
        self._row_id = self.df.index[0]
        self.model = model

        post_one.pop(mep.site.obs, None)
        post_one.pop(mep.site.mu, None)

        post_one = {u: v[:, participant:participant + 1] for u, v in post_one.items()}

        self._post_one = post_one
        self._draw = draw
        self._participant = participant
        return

    def if_stop(self) -> bool:
        return False

    @property
    def draw(self) -> int:
        return self._draw

    @draw.setter
    def draw(self, value: int):
        self._init_draw(value)
        return

    def reset(self, *, seed: int | None = None) -> None:
        self._key = random.key(0 if seed is None else seed)
        return

    def clip(self, x: float) -> float:
        return np.clip(x, self.lo, self.hi)

    def step(self, action):
        intensity, features = action
        self.df.loc[self._row_id, self.model.intensity] = intensity
        self.df.loc[self._row_id, self.model.features[-1]] = features

        self._key, subkey = random.split(self._key)
        obs = mep.predict(
            subkey,
            self.model._model,
            *mep.get_regressors(self.df, **self.model.variables),
            posterior=self._post_one,
            return_sites=[mep.site.obs],
        )[mep.site.obs].reshape(-1,)

        if self.response is not None:
            mapping = dict(zip(self.model.response, obs))
            obs = np.array([mapping[r] for r in self.response])

        return Observation(
            response=obs,
            intensity=intensity,
            features=features,
            info={"draw": self._draw},
        )

    def state_dict(self) -> dict:
        d = super().state_dict()
        key_data = random.key_data(self._key)
        d["simulation_env"] = {
            "key_data": key_data.tolist(),
            "key_dtype": str(key_data.dtype.name),
        }
        return d

    def load_state_dict(self, state: dict):
        super().load_state_dict(state)
        ed = state.get("simulation_env", {})
        key_data = ed.get("key_data", [0, 0])
        key_dtype = jnp.dtype(state.get("key_dtype", jnp.uint32))
        self._key = random.wrap_key_data(jnp.array(key_data, dtype=key_dtype))
        return
