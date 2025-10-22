import logging

import pandas as pd
from jax import random, Array, numpy as jnp, jit, vmap
from jax.tree_util import tree_map
import numpy as np
import numpyro as pyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, Predictive, SVI, Trace_ELBO
from numpyro.infer import autoguide
from numpyro import optim
import hbmep as mep
from hbmep import BaseModel, functional as F, site
from rtmep import Observation, ObsArrays
from rtmep.telemetry import _now_ns
from rtmep.util import execute_on_gpu

logger = logging.getLogger(__name__)
EPS = 1e-3


def _make_jitted_step(svi: SVI):
    @jit
    def _step(svi_state, *args, **kw):
        return svi.update(svi_state, *args, **kw)
    return _step


class BaseRTAgent():
    def __init__(self, *, minimum_observations_at_zero: int = 8):
        self.min_obs_at_zero = int(minimum_observations_at_zero)
        cap = 32 + (self.min_obs_at_zero // 2)
        self.data = ObsArrays(initial_capacity=cap)

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
        infer: str = "mcmc"
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

        if infer not in ("mcmc", "svi"):
            raise ValueError("infer must be 'mcmc' or 'svi'")
        self.infer = infer

        self._mcmc_state = None

        self.svi_lr = None
        self.svi_num_steps = None
        self.svi_num_samples = None
        self.svi_num_restarts = None
        self.svi_argmins = None

        self._guide = None
        self._svi = None
        self._svi_step = None
        self._svi_state = None
        self._svi_states = None

    def _svi_init(self):
        self.svi_lr = 5e-3
        self.svi_num_steps = 4000
        self.svi_num_samples = int(
            self.mcmc_params["num_chains"]
            * self.mcmc_params["num_samples"]
            // self.mcmc_params["thinning"]
        )
        self.svi_num_restarts = 100
        self.svi_argmins = []

        self._guide = autoguide.AutoMultivariateNormal(self._model)
        self._svi = SVI(
            self._model,
            self._guide,
            optim.ClippedAdam(self.svi_lr),
            loss=Trace_ELBO(num_particles=20)
        )
        self._svi_step = _make_jitted_step(self._svi)

    def _svi_warm_up(self, key, *args, **kw):
        svi_state = self._svi.init(key, *args, **kw)
        return self._svi_step(svi_state, *args, **kw)

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("_mcmc_state", None)
        d.pop("_guide", None)
        d.pop("_svi", None)
        d.pop("_svi_step", None)
        d.pop("_svi_state", None)
        d.pop("_svi_states", None)
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.reset_mcmc()
        self.reset_svi()

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
    
    def reset_mcmc(self) -> None:
        self._mcmc_state = None

    def reset_svi(self) -> None:
        self._guide = None
        self._svi = None
        self._svi_step = None
        self._svi_state = None
        self._svi_states = None

    def run_mcmc(
        self,
        df: pd.DataFrame = None,
        mcmc: MCMC = None,
        extra_fields: list | tuple = (),
        init_params=None,
        key: Array | None = None,
        **kw
    ) -> tuple[MCMC, dict]:
        t1 = _now_ns()
        if not len(self.data):
            raise ValueError("No observations in buffer, collect data before run().")

        n_cap, r_cap = self.data.capacity_shape
        shape_now = (n_cap, r_cap)
        num_used = jnp.array([[self.data.n]])
        model_kw = dict(
            intensity=self.data._X,
            features=jnp.empty((n_cap, 0)),
            response=self.data._Y,
            num_used=num_used
        )

        if shape_now != self._compiled_shape:
            self.mcmc = None
            self._compiled_shape = shape_now

        t0 = _now_ns()
        self.key, subkey = random.split(self.key)
        mcmc = mep.run(
            key=subkey,
            model=self._model,
            **model_kw,
            nuts_params=self.nuts_params,
            mcmc_params=self.mcmc_params,
            mcmc=mcmc,
            extra_fields=extra_fields,
            init_params=init_params,
            **kw
        )
        _run_ns = _now_ns() - t0
        t0 = _now_ns()
        posterior = mcmc.get_samples()
        _posterior_ns = _now_ns() - t0
        _total_ns = _now_ns() - t1
        self._last_run_ns = dict(
            run=_run_ns,
            posterior=_posterior_ns,
            total_mcmc=_total_ns
        )
        return mcmc, posterior

    def run(self, **kw):
        if self.infer == "svi":
            return self.run_svi(**kw)
        return self.run_mcmc(**kw)

    def state_dict(self, include_posterior: bool = True) -> dict:
        key_data = random.key_data(self._key)
        out = {
            "type": self.__class__.__name__,
            "model": super().state_dict(),
            "min_obs_at_zero": int(self.min_obs_at_zero),
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
        self.min_obs_at_zero = int(state.get("min_obs_at_zero", 6))      
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
            num_used = num_used[0, 0]
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

    @execute_on_gpu
    def run_svi(
        self,
        *,
        num_steps=None,
        lr=None,
        num_samples=None,
        key=None,
        num_restarts=None,
        **kw
    ):
        t1 = _now_ns()
        if not len(self.data):
            raise ValueError("No observations in buffer, collect data before run_svi().")

        n_cap, r_cap = self.data.capacity_shape
        shape_now = (n_cap, r_cap)
        num_used = jnp.array([[self.data.n]])
        model_kw = dict(
            intensity=self.data._X,
            features=jnp.empty((n_cap, 0)),
            response=self.data._Y,
            num_used=num_used
        )

        if shape_now != self._compiled_shape:
            self.reset_svi()
            self._compiled_shape = shape_now

        needs_warmup = False
        if self._svi is None:
            self._svi_init()
            needs_warmup = True

        num_steps = int(self.svi_num_steps if num_steps is None else num_steps)
        lr = float(self.svi_lr if lr is None else lr)
        num_samples = int(self.svi_num_samples if num_samples is None else num_samples)
        num_restarts = int(self.svi_num_restarts if num_restarts is None else num_restarts)

        if num_restarts <= 1:
            t0 = _now_ns()
            if needs_warmup:
                self.key, subkey = random.split(self.key)
                self._svi_state, _ = self._svi_warm_up(subkey, **model_kw)
            _init_ns = _now_ns() - t0

            t0 = _now_ns()
            svi_state = self._svi_state
            for _ in range(num_steps):
                svi_state, loss = self._svi_step(svi_state, **model_kw)
            _run_ns = _now_ns() - t0
            self._svi_state = svi_state
        
        else:
            def body_warm_up(keys):
                def one(key):
                    return self._svi_warm_up(key, **model_kw)
                states, losses = vmap(one)(keys)
                return states, losses

            t0 = _now_ns()
            if needs_warmup:
                self.key, subkey = random.split(self.key)
                subkeys = random.split(subkey, num_restarts)
                self._svi_states, _ = body_warm_up(subkeys)
            _init_ns = _now_ns() - t0

            def body_update(svi_states):
                def one(svi_state):
                    for _ in range(num_steps):
                        svi_state, loss = self._svi_step(svi_state, **model_kw)
                    return svi_state, loss
                states, losses = vmap(one)(svi_states)
                return states, losses

            t0 = _now_ns()
            svi_states = self._svi_states
            svi_states, losses = body_update(svi_states)
            _run_ns = _now_ns() - t0
            self._svi_states = svi_states

            best_idx = jnp.argmin(losses)
            # logger.info(f"SVI argmin {best_idx}/{num_restarts - 1}, loss:{losses[best_idx]:.3f}")
            self.svi_argmins.append(best_idx)
            self._svi_state = tree_map(lambda x: x[best_idx], self._svi_states)

        t0 = _now_ns()
        svi_state = self._svi_state
        _svi_params = self._svi.get_params(svi_state)
        _params_ns = _now_ns() - t0

        # Posterior draws
        t0 = _now_ns()
        self.key, subkey = random.split(self.key)
        predictive = Predictive(
            self._guide,
            params=_svi_params,
            num_samples=num_samples,
        )
        posterior = predictive(subkey)
        _posterior_ns = _now_ns() - t0
        _total_ns = _now_ns() - t1
        self._last_run_ns = dict(
            init=_init_ns,
            run=_run_ns,
            params=_params_ns,
            posterior=_posterior_ns,
            total_svi=_total_ns
        )
        return None, posterior
