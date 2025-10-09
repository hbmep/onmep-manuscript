import pandas as pd
from jax import random, Array, numpy as jnp
import numpy as np
import numpyro as pyro
import numpyro.distributions as dist
from numpyro.infer import MCMC
import hbmep as mep
from hbmep import BaseModel, functional as F, site
from rtmep import Observation, ObsArrays
from rtmep.telemetry import _now_ns

EPS = 1e-3


class BaseRTAgent():
    def __init__(self):
        self.data = ObsArrays()

    def reset(self) -> None:
        self.data.clear()
    
    def propose(self) -> float:
        raise NotImplementedError

    def update(self, obs: Observation) -> bool:
        raise NotImplementedError


class BaseAgent(BaseRTAgent, BaseModel):
    def __init__(
        self,
        *,
        toml_path: str | None = None,
        config: dict | None = None,
    ) -> None:
        BaseRTAgent.__init__(self)
        BaseModel.__init__(self, toml_path=toml_path, config=config)
        self.features = []
        self.mcmc = None
        self.posterior = None
        self._key = random.key(0)
        self.data = ObsArrays(R=self.num_response)
        self._compiled_shape = None
        self._last_run_ns = None

    def _on_response_changed(self, old, new):
        if hasattr(self, "data") and self.data is not None:
            self.data.resize_R(len(new))
        if hasattr(self, "_compiled_shape"):
            self._compiled_shape = None

    def reset(self) -> None:
        super().reset()
        self.mcmc = None
        self.posterior = None
        self._key = random.key(0)
        self._compiled_shape = None
        self._last_run_ns = None

    def run(
        self,
        df: pd.DataFrame = None,
        mcmc: MCMC = None,
        extra_fields: list | tuple = (),
        init_params=None,
        key: Array | None = None,
        **kw
    ) -> tuple[MCMC, dict]:
        if not len(self.data):
            raise ValueError("No observations in buffer, collect data before run().")

        n, r = self.data.capacity_shape
        shape_now = (n, r)
        if shape_now != self._compiled_shape:
            self.mcmc = None
            self._compiled_shape = shape_now

        t0 = _now_ns()
        self.key, subkey = random.split(self.key)
        mcmc = mep.run(
            key=subkey,
            model=self._model,
            intensity=self.data._X,
            features=np.empty((n, 0)),
            response=self.data._Y,
            nuts_params=self.nuts_params,
            mcmc_params=self.mcmc_params,
            mcmc=mcmc,
            extra_fields=extra_fields,
            init_params=init_params,
            num_used=self.data.n,
            **kw
        )
        posterior = mcmc.get_samples()
        self._last_run_ns = _now_ns() - t0
        return mcmc, posterior

    def state_dict(self, include_posterior: bool = True) -> dict:
        key_data = random.key_data(self._key)
        out = {
            "type": self.__class__.__name__,
            "model": super().state_dict(),
            "data": self.data.to_state(),
            "key_data": key_data.tolist(),
            "key_dtype": str(key_data.dtype.name),
            "_compiled_shape": self._compiled_shape,
            "_last_run_ns": self._last_run_ns
        }
        if include_posterior and (self.posterior is not None):
            out["posterior"] = {k: np.asarray(v) for k, v in self.posterior.items()}
        return out

    def load_state_dict(self, state: dict):
        super().load_state_dict(state.get("model", {}))
        self.data = ObsArrays.from_state(state["data"])
        key_data = state.get("key_data", [0, 0])
        key_dtype = jnp.dtype(state.get("key_dtype", jnp.uint32))
        self._key = random.wrap_key_data(jnp.array(key_data, dtype=key_dtype))
        self._compiled_shape = None
        self._last_run_ns = state.get("_last_run_ns", None)
        self.mcmc = None
        posterior = state.get("posterior", None)
        if posterior is not None:
            self.posterior = {u: jnp.asarray(v) for u, v in posterior.items()}
        else:
            self.posterior = None

    def rl(self, intensity, features, response=None, num_used=None, **kw):
        num_data = intensity.shape[0]
        mask_obs = True
        if num_used is not None:
            mask_obs = jnp.arange(intensity.shape[0]) < num_used
            mask_obs = mask_obs[:, None]
        if response is not None:
            mask_obs = mask_obs & jnp.invert(jnp.isnan(response))

        with pyro.plate(site.num_response, self.num_response):
            a = pyro.sample(site.a, dist.TruncatedNormal(50., 50., low=0.))
            b = pyro.sample(site.b, dist.HalfNormal(1.))
            g = pyro.sample(site.g, dist.HalfNormal(.1))
            h = pyro.sample(site.h, dist.HalfNormal(5.))
            v = pyro.sample(site.v, dist.HalfNormal(5.))
            c1 = pyro.sample(site.c1, dist.HalfNormal(5.))
            c2 = pyro.sample(site.c2, dist.HalfNormal(.5))

        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(site.num_response, self.num_response):
                with pyro.plate(site.num_data, num_data):
                    mu = F.rectified_logistic(intensity, a, b, g, h, v, EPS)
                    pyro.deterministic(site.mu, mu)
                    alpha, beta = self.gamma_likelihood(mu, c1, c2)
                    pyro.sample(
                        site.obs,
                        dist.Gamma(concentration=alpha, rate=beta),
                        obs=response
                    )
