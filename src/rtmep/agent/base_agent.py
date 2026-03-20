import os
import logging

import hbmep as mep
from jax import random, numpy as jnp, jit
from jax.scipy.special import logsumexp
import numpyro as pyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, SVI, Trace_ELBO, autoguide
from numpyro.infer.util import log_likelihood
from numpyro import optim

from rtmep import Observation, ObsArrays
from rtmep.telemetry import _now_ns
from rtmep.util import execute_on_gpu

logger = logging.getLogger(__name__)

EPS = 1e-3


def _make_jitted_step(svi: SVI):
    @jit
    def _step(svi_state, **model_kw):
        return svi.update(svi_state, **model_kw)
    return _step


class BaseRTAgent:
    def __init__(
        self, *, D: int | None = None, R: int | None = None,
        initial_capacity: int = 32, N0: int = 8
    ):
        self.N0 = N0
        self.data = ObsArrays(D=D, R=R, initial_capacity=initial_capacity)
        return

    def reset(self):
        self.data.reset()
        return

    def propose(self):
        raise NotImplementedError

    def update(self, obs: Observation) -> bool:
        raise NotImplementedError

    def state_dict(self, include_posterior: bool = True) -> dict:
        raise NotImplementedError

    def load_state_dict(self, state: dict):
        raise NotImplementedError


class BaseAgent(mep.BaseModel, BaseRTAgent):
    def __init__(
        self, *,
        key: random.key = random.key(0),
        toml_path: str | None = None,
        config: dict | None = None,
        infer: str = "mcmc",
        initial_capacity: int = 32,
        ess_threshold: float = 0.5,
        K: int = 1,
        N0: int = 8,
    ) -> None:
        mep.BaseModel.__init__(self, key=key, toml_path=toml_path, config=config)

        self.K = K
        self.D = len(self.features)
        if self.D > 1:
            raise NotImplementedError("Not implemented for multiple features.")
        BaseRTAgent.__init__(
            self, D=self.D, R=self.num_response,
            initial_capacity=initial_capacity, N0=N0
        )

        if infer not in ("mcmc", "svi", "sequential", "smc"):
            raise ValueError("infer must be 'mcmc', 'svi', 'sequential' or 'smc'")
        self.infer = infer

        self.key, self._key = random.split(self.key)

        self._compiled_shape = None
        self._last_run_ns = None
        self.posterior: dict | None = None
        self._logw = None
        self.ess_threshold = ess_threshold

        # Inference state
        self._mcmc = {"kernel": None, "state": None}
        self._svi = {"guide": None, "obj": None, "step": None, "state": None}
        return

    def _ensure_compiled_shape(self):
        shape_now = self.data.capacity_shape
        if shape_now != self._compiled_shape:
            self._compiled_shape = shape_now
            self.reset_mcmc()
            self.reset_svi()
        return

    def _model_kwargs(self):
        num_used = jnp.array([[self.data.n]])
        return dict(
            intensity=self.data._X,
            features=self.data._F,
            response=self.data._Y,
            num_used=num_used,
        )

    def _require_data(self, msg: str):
        if not len(self.data):
            raise ValueError(msg)
        return

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("_mcmc", None)
        d.pop("_svi", None)
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._mcmc = {"kernel": None, "state": None}
        self._svi = {"guide": None, "obj": None, "step": None, "state": None}
        return

    def _on_response_changed(self, old, new):
        if getattr(self, "data", None) is not None:
            self.data.resize_R(len(new))
        self._compiled_shape = None
        return

    def reset(self):
        super().reset()
        self._key = random.key(0)
        self._compiled_shape = None
        self._last_run_ns = None
        self.posterior = None
        self._logw = None
        self.reset_mcmc()
        self.reset_svi()
        return

    def reset_mcmc(self):
        self._mcmc["kernel"] = None
        self._mcmc["state"] = None
        return

    def reset_svi(self):
        self._svi["guide"] = None
        self._svi["obj"] = None
        self._svi["step"] = None
        self._svi["state"] = None
        return

    def run_mcmc(self, *, extra_fields=(), init_params=None, **kw):
        t1 = _now_ns()
        self._require_data("No observations in buffer, collect data before run_mcmc().")
        self._ensure_compiled_shape()

        model_kw = self._model_kwargs()

        if self._mcmc["kernel"] is None:
            self._mcmc["kernel"] = NUTS(self._model, **self.nuts_params)

        mcmc = MCMC(self._mcmc["kernel"], **self.mcmc_params)

        t0 = _now_ns()
        self.key, subkey = random.split(self.key)
        mcmc.run(subkey, **model_kw, extra_fields=extra_fields, init_params=init_params, **kw)
        run_ns = _now_ns() - t0

        t0 = _now_ns()
        posterior = mcmc.get_samples()
        post_ns = _now_ns() - t0

        self._mcmc["state"] = mcmc.last_state
        self._last_run_ns = dict(run=run_ns, posterior=post_ns, total=_now_ns() - t1)

        self.posterior = posterior
        S = next(iter(posterior.values())).shape[0]
        self._logw = jnp.zeros(S)
        return mcmc, posterior

    def _init_svi(self):
        self._ensure_compiled_shape()
        if self._svi["obj"] is not None:
            return

        guide = autoguide.AutoMultivariateNormal(self._model)
        svi = SVI(
            self._model,
            guide,
            optim.ClippedAdam(5e-3),
            loss=Trace_ELBO(num_particles=200),
        )
        self._svi["guide"] = guide
        self._svi["obj"] = svi
        self._svi["step"] = _make_jitted_step(svi)
        return

    @execute_on_gpu
    def run_svi(self, *, num_steps=2000, num_samples=500, **kw):
        self._require_data("No observations in buffer, collect data before run_svi().")
        self._init_svi()

        model_kw = self._model_kwargs()
        svi: SVI = self._svi["obj"]
        step_fn = self._svi["step"]

        self.key, subkey = random.split(self.key)
        svi_state = svi.init(subkey, **model_kw)

        for _ in range(num_steps):
            svi_state, _ = step_fn(svi_state, **model_kw)

        self._svi["state"] = svi_state
        params = svi.get_params(svi_state)

        self.key, subkey = random.split(self.key)
        predictive = pyro.infer.Predictive(self._svi["guide"], params=params, num_samples=num_samples)
        posterior = predictive(subkey)

        self.posterior = posterior
        return None, posterior

    def sequential_update(self):
        n = self.data.n - 1

        model_kw = dict(
            intensity=self.data._X[n:n+1],
            features=self.data._F[n:n+1],
            response=self.data._Y[n:n+1],
            num_used=jnp.array([[1]]),
        )

        ll = log_likelihood(self._model, self.posterior, **model_kw)
        u = ll[mep.site.obs]
        loglik = u.sum(tuple(range(1, u.ndim))).reshape(-1)

        logw = self._logw + loglik
        logw = logw - logsumexp(logw)
        self._logw = logw

        w = jnp.exp(logw)
        ess = 1.0 / jnp.sum(w * w)

        S = next(iter(self.posterior.values())).shape[0]
        thresh = self.ess_threshold * S if self.ess_threshold <= 1.0 else self.ess_threshold
        if ess < thresh:
            return self.run_mcmc()

        return None, self.posterior

    def run(self, **kw):
        if self.infer == "mcmc":
            return self.run_mcmc(**kw)
        if self.infer == "svi":
            return self.run_svi(**kw)
        if self.infer == "sequential":
            if self.posterior is None:
                return self.run_mcmc(**kw)
            return self.sequential_update()
        if self.infer == "smc":
            if self.posterior is None:
                return self.run_mcmc(**kw)
            return self.sequential_update()
        raise ValueError("infer must be 'mcmc', 'svi', 'sequential' or 'smc'")

    def state_dict(self, include_posterior: bool = True) -> dict:
        key_data = random.key_data(self._key)
        out = {
            "type": self.__class__.__name__,
            "model": super().state_dict(),
            "D": self.D,
            "N0": self.N0,
            "ess_threshold": self.ess_threshold,
            "data": self.data.to_state(),
            "key_data": key_data.tolist(),
            "key_dtype": str(key_data.dtype.name),
            "_compiled_shape": self._compiled_shape,
            "_last_run_ns": self._last_run_ns,
        }
        if include_posterior and (self.posterior is not None):
            import numpy as np
            out["posterior"] = {k: np.asarray(v) for k, v in self.posterior.items()}
            out["_logw"] = None if self._logw is None else np.asarray(self._logw)
        return out

    def load_state_dict(self, state: dict):
        super().load_state_dict(state.get("model", {}))
        self.D = state.get("D", self.D)
        self.N0 = state.get("N0", self.N0)
        self.data = ObsArrays.from_state(state["data"])
        self.ess_threshold = state.get("ess_threshold", self.ess_threshold)

        key_data = state.get("key_data", [0, 0])
        key_dtype = jnp.dtype(state.get("key_dtype", jnp.uint32))
        self._key = random.wrap_key_data(jnp.array(key_data, dtype=key_dtype))

        # Force re-init of compiled/inference state
        self._compiled_shape = None
        self._last_run_ns = state.get("_last_run_ns", None)
        self.reset_mcmc()
        self.reset_svi()

        posterior = state.get("posterior", None)
        self.posterior = (
            {u: jnp.asarray(v) for u, v in posterior.items()}
            if posterior is not None else None
        )
        lw = state.get("_logw", None)
        self._logw = jnp.asarray(lw) if lw is not None else None
        return

    def plot_curves(self, output_path: str | None = None):
        import numpy as np
        if self.posterior is None:
            raise ValueError("No posterior available. Run inference first.")
        df = self.data.to_df(**self.variables)
        num_used = self.data.n
        posterior = {u: np.array(v) for u, v in self.posterior.items()}
        posterior[mep.site.mu] = posterior[mep.site.mu][:, :num_used]
        prediction_df = self.make_prediction_dataset(df, num_points=100)
        ppd = self.predict(df=prediction_df, posterior=posterior)

        if not output_path and not self.build_dir:
            return

        if output_path is None:
            output_path = os.path.join(self.build_dir, "curves.pdf")

        posterior_var = mep.site.a
        prediction_var = mep.site.mu
        threshold = (
            posterior[posterior_var]
            if posterior is not None and posterior_var in posterior.keys()
            else None
        )
        figures = mep.plot(
            df=df,
            **self.variables,
            prediction_df=prediction_df,
            prediction=ppd[prediction_var],
            threshold=threshold,
        )
        figures = [u for u, _ in figures]
        mep.make_pdf(figures=figures, output_path=output_path)
        return

    def rl(self, intensity, features, response=None, num_used=None, **kw):
        num_data = intensity.shape[0]
        num_features = [self.K]

        mask_obs = True
        if num_used is not None:
            num_used = num_used[0, 0]
            mask_obs = (jnp.arange(num_data) < num_used)[:, None]
        if response is not None:
            mask_obs = mask_obs & jnp.invert(jnp.isnan(response))

        with pyro.plate(mep.site.num_response, self.num_response):
            with pyro.plate(mep.site.num_features[0], num_features[0]):
                a = pyro.sample(mep.site.a, dist.TruncatedNormal(50., 50., low=0.))
                b = pyro.sample(mep.site.b, dist.HalfNormal(1.))
                
                g = pyro.sample(mep.site.g, dist.HalfNormal(.1))    # L
                h = pyro.sample(mep.site.h, dist.HalfNormal(5.))    # H
                v = pyro.sample(mep.site.v, dist.HalfNormal(5.))    # ell

                c1 = pyro.sample(mep.site.c1, dist.HalfNormal(5.))
                c2 = pyro.sample(mep.site.c2, dist.HalfNormal(.5))

        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(mep.site.num_response, self.num_response):
                with pyro.plate(mep.site.num_data, num_data):
                    mu = mep.functional.rectified_logistic(
                        intensity,
                        a[*features.T],
                        b[*features.T],
                        g[*features.T],
                        h[*features.T],
                        v[*features.T],
                        EPS,
                    )
                    pyro.deterministic(mep.site.mu, mu)
                    alpha, beta = self.gamma_likelihood(mu, c1[*features.T], c2[*features.T])
                    pyro.sample(mep.site.obs, dist.Gamma(concentration=alpha, rate=beta), obs=response)
