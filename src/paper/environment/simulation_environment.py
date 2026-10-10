import logging

import hbmep as mep
import numpy as np
from jax import random, numpy as jnp, Array
from numpyro import distributions as dist

from onmep import Observation
from onmep.environment.base_environment import BaseEnvironment
from onmep.agent.base_agent import Action

from paper.model.simulate import Simulator

logger = logging.getLogger(__name__)


class SimulationEnv(BaseEnvironment):
    def __init__(
        self,
        *,
        key: Array,
        model_dir: str,
        draw: int,
        response: list[str] | None = None,
        outlier_prob: float = 0.,
        outlier_scale: float = 10.,
        **kw,
    ):
        super().__init__(**kw)
        self.name = "simulation_env"
        self.key = key

        self._model_dir = model_dir
        self._draw = draw
        self._init_draw()

        self.response = response
        self.outlier_prob = outlier_prob
        self.outlier_scale = outlier_scale
        return

    def _init_draw(self):
        model_state, post_one, df, *_ = mep.load(
            self._model_dir, memmap_draw=self._draw
        )
        model = Simulator()
        model.load_state_dict(model_state)
        df = df.iloc[[0]].reset_index(drop=True).copy()
        df[model.features[0]] = 0
        self.model = model
        self.post_one = post_one
        self.df = df
        return

    @property
    def draw(self) -> int:
        return self._draw

    @draw.setter
    def draw(self, value: int):
        self._draw = value
        self._init_draw()
        return

    @property
    def model_dir(self) -> str:
        return self._model_dir

    @model_dir.setter
    def model_dir(self, value: str):
        self._model_dir = value
        self._init_draw()
        return

    def should_stop(self):
        return False

    def _step(self, intensity: float, features: list[int]) -> Observation:
        self.df.loc[0, self.model.intensity] = intensity
        self.df.loc[0, self.model.features[-1]] = features[0]

        self.key, sub = random.split(self.key)
        obs = mep.predict(
            sub,
            self.model._model,
            *mep.get_regressors(self.df, **self.model.variables),
            posterior=self.post_one,
            return_sites=[mep.site.obs],
        )
        obs = obs[mep.site.obs].reshape(-1,)
        assert obs.shape[0] == self.model.num_response

        is_outlier = False
        if self.outlier_prob > 0:
            self.key, sub_is_outlier, sub = random.split(self.key, 3)
            is_outlier = bool(random.bernoulli(sub_is_outlier, p=self.outlier_prob))
            if is_outlier:
                obs = np.array(
                    dist.HalfNormal(self.outlier_scale).sample(
                        sub, sample_shape=(self.model.num_response,)
                    )
                ).reshape(-1,)

        if self.response is not None:
            mapping = dict(zip(self.model.response, obs))
            obs = np.array([mapping[r] for r in self.response])

        return Observation(
            intensity=intensity,
            features=jnp.asarray(features),
            response=obs,
            id=int(is_outlier),
            info={"draw": self.draw},
        )

    def step(self, action: Action) -> Observation:
        intensity = action.x
        features = action.f
        obs = self._step(intensity=intensity, features=features)
        return obs

    def state_dict(self) -> dict:
        d = super().state_dict()
        key_data = random.key_data(self.key)
        d["simulation_env"] = {
            "key_data": key_data.tolist(),
            "key_dtype": str(key_data.dtype.name),
            "_model_dir": self.model_dir,
            "_draw": self.draw,
            "response": self.response,
            "outlier_prob": self.outlier_prob,
            "outlier_scale": self.outlier_scale,
        }
        return d

    def load_state_dict(self, state: dict):
        super().load_state_dict(state)
        ed = state.get("simulation_env", {})
        key_data = ed.get("key_data", [0, 0])
        key_dtype = jnp.dtype(ed.get("key_dtype", jnp.uint32))
        self.key = random.wrap_key_data(jnp.array(key_data, dtype=key_dtype))

        self._model_dir = ed.get("_model_dir", self._model_dir)
        self._draw = ed.get("_draw", self._draw)
        self._init_draw()

        self.response = ed.get("response", self.response)
        self.outlier_prob = ed.get("outlier_prob", self.outlier_prob)
        self.outlier_scale = ed.get("outlier_scale", self.outlier_scale)
        return self
