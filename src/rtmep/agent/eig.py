import numpy as np
import jax
from jax import random, numpy as jnp
from jax.scipy.special import logsumexp, gammaln
from hbmep import site, functional as F

from rtmep import Observation
from rtmep.agent.base_agent import BaseAgent
from rtmep.model.simulate import EPS
from rtmep.util import execute_on_gpu

KEYS = (site.a, site.b, site.g, site.h, site.v, site.c1, site.c2)


@execute_on_gpu
def loglik_gamma(y, alpha, beta):
    """
    Joint Gamma(site.alpha, site.beta) log-likelihood across R muscles (conditionally independent).
    All inputs broadcast to (..., R) -> returns (...)
    """
    return jnp.sum(
        (
            alpha * jnp.log(beta)
            - gammaln(alpha)
            + (alpha - 1.) * jnp.log(y)
            - beta * y
        ),
        axis=-1
    )


@execute_on_gpu
def _gamma_params_single(self, x, post_one):
    """
    Compute (mu, alpha, beta) at scalar x for ONE posterior draw.
    """
    x_arr = jnp.array([[x]])                            # shape (1, 1)
    mu = F.rectified_logistic(x_arr, *(post_one[u] for u in KEYS[:-2]), EPS)
    alpha, beta = self.gamma_likelihood(mu, post_one[site.c1], post_one[site.c2])
    return mu[0], alpha[0], beta[0]                     # each (R,)


@execute_on_gpu
def _gamma_params_all(self, x, posterior):
    """
    Vectorized over S draws -> returns alpha (S, R), beta (S, R)
    """
    S = posterior[site.a].shape[0]
    def body_one_draw(i):
        post_one = {k: posterior[k][i] for k in KEYS}               # each (R,)
        _, alpha, beta = _gamma_params_single(self, x, post_one)
        return alpha, beta
    alpha_all, beta_all = jax.vmap(body_one_draw)(jnp.arange(S))    # each (S, R)
    return alpha_all, beta_all


@execute_on_gpu
def eig_mc_for_x_gamma(self, x, posterior, key, num_outer=None):
    """
    Uses
        U ~ Gamma(site.alpha, 1) (in shape-rate parameterization)
        -> U / site.beta ~ Gamma(site.alpha, site.beta)
    """
    S = posterior[site.a].shape[0]
    if (num_outer is None) or (num_outer >= S):
        idx = jnp.arange(S)
    else:
        idx = jax.random.choice(key, S, (num_outer,), replace=False)
    S_outer = idx.shape[0]
    alpha_all, beta_all = _gamma_params_all(self, x, posterior)     # (S, R)
    keys = jax.random.split(key, S_outer)

    def body_one_draw(k, i):
        alpha_s, beta_s = alpha_all[i], beta_all[i]                 # (R,)
        y_s = jax.random.gamma(key=k, a=alpha_s) / beta_s           # (R,)
        loglik_s = loglik_gamma(y_s, alpha_s, beta_s)
        def body_ll_j(alp_j, bet_j):
            return loglik_gamma(y_s, alp_j, bet_j)
        loglik_all = jax.vmap(body_ll_j)(alpha_all, beta_all)       # (S,)
        log_pred = logsumexp(loglik_all) - jnp.log(S)
        return loglik_s - log_pred

    gains = jax.vmap(body_one_draw)(keys, idx)
    return gains.mean()


@execute_on_gpu
def eig_mc_grid_gamma(self, xs, posterior, key, num_outer=None, batch_size=100):
    n = xs.shape[0]
    keys = random.split(key, n)
    outs = []
    if batch_size is None:
        batch_size = n
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        vals_batch = (
            jax.vmap(lambda xx, kk: eig_mc_for_x_gamma(
                self, xx, posterior, kk, num_outer
            ))
            (xs[start:stop], keys[start:stop])
        )
        outs.append(vals_batch)
    return jnp.concatenate(outs, axis=0)


class EIG(BaseAgent):
    def __init__(
        self,
        *,
        toml_path: str | None = None,
        config: dict | None = None,
        infer: str = "mcmc",
        min_observations: int = 3,
        grid_lo: float = 0.0,
        grid_hi: float = 100.0,
        grid_n: int = 201,
        method: str ="mc",
        num_outer: int | None = None,
        max_step: float | None = None,
        batch_size: int | None = 25,
    ):
        super().__init__(toml_path=toml_path, config=config, infer=infer)
        self.min_obs = int(min_observations)
        self.grid_n = grid_n
        self.grid = jnp.linspace(float(grid_lo), float(grid_hi), int(grid_n))
        self._perm = None
        self._perm_i = 0
        self.method = method
        self.num_outer = num_outer
        self.max_step = max_step
        self.batch_size = batch_size
        self._last_x = None

    def _local_grid_fixed_n(self, local_n=101):
        if (self.max_step is None) or (self._last_x is None):
            return self.grid
        lo = jnp.maximum(self.grid.min(), self._last_x - self.max_step)
        hi = jnp.minimum(self.grid.max(), self._last_x + self.max_step)
        return jnp.linspace(lo, hi, int(local_n))

    @execute_on_gpu
    def _choose_next_x(self, posterior):
        self._key, sub = random.split(self._key)
        eig_vals = eig_mc_grid_gamma(
            self, self.grid, posterior, sub, num_outer=self.num_outer,
            batch_size=self.batch_size
        )

        if (self.max_step is not None) and (self._last_x is not None):
            lo = jnp.maximum(self.grid.min(), self._last_x - self.max_step)
            hi = jnp.minimum(self.grid.max(), self._last_x + self.max_step)
            mask = (self.grid >= lo) & (self.grid <= hi)
            k_local = int(jnp.argmax(jnp.where(mask, eig_vals, -jnp.inf)))
        else:
            k_local = int(jnp.argmax(eig_vals))

        x_next = float(self.grid[k_local]) 
        self._last_x = x_next
        return x_next, eig_vals

    def reset(self, *args, **kw):
        super().reset(*args, **kw)
        self._perm = None
        self._perm_i = 0
        self._last_x = None

    def random_proposal(self):
        needs_zero = len(self.data) < self.min_obs_at_zero
        if needs_zero:
            return float(0)

        if (self._perm is None) or (self._perm_i >= self.grid.shape[0]):
            self._key, sub = random.split(self._key)
            self._perm = random.permutation(sub, self.grid.shape[0])
            self._perm_i = 0
        idx = int(self._perm[self._perm_i])
        self._perm_i += 1
        return float(self.grid[idx])

    @execute_on_gpu
    def eig_gamma_proposal(self):
        needs_zero = len(self.data) < self.min_obs_at_zero
        if needs_zero:
            return float(0)

        needs_random = (
            len(self.data) - self.min_obs_at_zero < self.min_obs
            or self.posterior is None
        )
        if needs_random:
            return self.random_proposal()

        x, _ = self._choose_next_x(self.posterior)
        self._last_x = x
        return x

    def update_on_each(self, obs: Observation) -> bool:
        self.data.append(intensity=obs.intensity, response=obs.response)
        if len(self.data) >= self.min_obs_at_zero + self.min_obs:
            self.mcmc, self.posterior = self.run()
        return False

    def state_dict(self, include_posterior: bool = True) -> dict:
        d = super().state_dict(include_posterior=include_posterior)
        perm_list = None if self._perm is None else np.asarray(self._perm, dtype=int).tolist()

        d["eig"] = {
            "min_obs": int(self.min_obs),
            "grid_lo": float(self.grid.min()),
            "grid_hi": float(self.grid.max()),
            "grid_n": int(self.grid_n),
            "method": self.method,
            "num_outer": None if self.num_outer is None else int(self.num_outer),
            "max_step": None if self.max_step is None else float(self.max_step),
            "batch_size": None if self.batch_size is None else int(self.batch_size),
            "_last_x": self._last_x,
            "perm": perm_list,
            "perm_i": int(self._perm_i),
        }
        return d

    def load_state_dict(self, state: dict):
        super().load_state_dict(state)
        eig = state.get("eig", {})
        grid_lo = eig.get("grid_lo", None)
        grid_hi = eig.get("grid_hi", None)
        grid_n = eig.get("grid_n", None)
        if (grid_lo is not None) and (grid_hi is not None) and (grid_n is not None):
            self.grid_n = int(grid_n)
            self.grid = jnp.linspace(float(grid_lo), float(grid_hi), int(grid_n))

        self.min_obs = int(eig.get("min_obs", self.min_obs))
        self.method = eig.get("method", self.method)
        _num_outer = eig.get("num_outer", self.num_outer)
        self.num_outer = None if _num_outer is None else int(_num_outer)
        _max_step = eig.get("max_step", self.max_step)
        self.max_step = None if _max_step is None else float(_max_step)
        _bs = eig.get("batch_size", self.batch_size)
        self.batch_size = None if _bs is None else int(_bs)
        self._last_x = eig.get("_last_x")

        # Restore permutation
        perm_list = eig.get("perm", None)
        self._perm_i = int(eig.get("perm_i", 0))
        if perm_list is None:
            self._perm = None
            self._perm_i = 0
        else:
            perm_arr = np.asarray(perm_list, dtype=int)
            if perm_arr.shape[0] != self.grid.shape[0]:
                self._perm = None
                self._perm_i = 0
            else:
                self._perm = jnp.array(perm_arr)
                self._perm_i = int(min(self._perm_i, self.grid.shape[0]))
