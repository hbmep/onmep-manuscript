import hbmep as mep
from jax import random, numpy as jnp, vmap
from jax.scipy.special import logsumexp

from rtmep.agent import BaseAgent
from rtmep.agent.eig_gamma import loglik_gamma, _params_all
from rtmep.agent.eig import EIG
from rtmep.util import execute_on_gpu


def _prior_var(param: jnp.ndarray, w: jnp.ndarray):
    """
    param: (S, R) prior samples
    w: (S,) normalized weights
    """
    mean = jnp.sum(w[:, None] * param, axis=0)          # (R,)
    mean2 = jnp.sum(w[:, None] * (param ** 2), axis=0)  # (R,)
    return mean2 - mean ** 2                            # (R,)


def evr_one(
    self: BaseAgent,
    x: float,
    k: int,
    posterior: dict[str, jnp.ndarray],
    key: random.key,
    num_outer: int | None = None
):
    """
    U(x) = sum_r E_y[ Var_prior[a_r] - Var_post[a_r | y, x] ]

    where the expectation over y is approximated using outer Monte Carlo,
    and Var_post is computed via importance reweighting of existing draws.
    """
    # Posterior
    a = posterior[mep.site.a]   # (S, K, R)
    a = a[:, k, :]              # (S, R)
    S = a.shape[0]
    key, key_idx, key_y = random.split(key, 3)

    logw_prior = self._logw
    if (logw_prior is None) or (logw_prior.shape[0] != S):
        logw_prior = jnp.zeros(S)
    logw_prior = logw_prior - logsumexp(logw_prior)
    w_prior = jnp.exp(logw_prior)

    # Prior variance
    var_prior = _prior_var(a, w=w_prior)    # (R,)
    alpha_all, beta_all = _params_all(self, x, k, posterior)    # (S, R)

    def body_one_draw(kkey, i):
        alpha_s, beta_s = alpha_all[i], beta_all[i]     # (R,)
        y_s = random.gamma(kkey, a=alpha_s) / beta_s    # (R,)
        loglik_all = vmap(
            lambda alp_j, bet_j: loglik_gamma(y_s, alp_j, bet_j)
        )(alpha_all, beta_all)  # (S,)

        # Importance weights for updated posterior p(theta | x, y):
        # w_post,j ∝ w_prior,j * p(y | theta_j, x)
        logw_post = logw_prior + loglik_all
        logw_post = logw_post - logsumexp(logw_post)
        w_post = jnp.exp(logw_post)                     # (S,)

        # Weighted mean and variance over draws, per muscle
        # a: (S, R), w_post: (S,)
        mean = jnp.sum(w_post[:, None] * a, axis=0)         # (R,)
        mean2 = jnp.sum(w_post[:, None] * (a ** 2), axis=0) # (R,)
        var_post = mean2 - mean ** 2                        # (R,)
        return var_prior - var_post                         # (R,)

    if (num_outer is None) or (num_outer >= S):
        idx = jnp.arange(S)
        keys = random.split(key_y, S)
        reductions = vmap(body_one_draw)(keys, idx)
        reduction = jnp.sum(w_prior[:, None] * reductions, axis=0)
    else:
        idx = random.choice(key_idx, S, (num_outer,), replace=True, p=w_prior)
        keys = random.split(key_y, num_outer)
        reductions = vmap(body_one_draw)(keys, idx)
        reduction = reductions.mean(axis=0)

    return jnp.sum(reduction)


@execute_on_gpu
def evr_grid(
    self: BaseAgent, xs, ks, posterior, key, num_outer=None, batch_size=64
):
    """Return evr_vals with shape (K, N)."""
    N = xs.shape[0]
    K = self.K
    keys_k = random.split(key, K)

    def eval_one_k(k_idx, kk):
        keys_x = random.split(kk, N)

        def one_x(xx, kx):
            return evr_one(
                self, xx, k_idx, posterior, kx, num_outer=num_outer
            )

        outs = []
        bs = N if batch_size is None else batch_size
        for start in range(0, N, bs):
            stop = min(start + bs, N)
            vals = vmap(one_x)(xs[start:stop], keys_x[start:stop])
            outs.append(vals)
        return jnp.concatenate(outs, axis=0)    # (N,)

    vals = vmap(eval_one_k)(ks, keys_k)         # (K, N)
    return vals


class EVR(EIG):
    def __init__(self, *args, **kw):
        kw.pop("policy", None)  # ignored for EVR
        super().__init__(*args, **kw)

    @execute_on_gpu
    def _choose_next_action(self, posterior: dict):
        self._key, sub = random.split(self._key)
        vals = evr_grid(
            self,
            self.grid,
            self.grid_f,
            posterior,
            sub,
            num_outer=self.num_outer,
            batch_size=self.batch_size,
        )

        if (self.max_step is not None) and (self._last_x is not None):
            lo = jnp.maximum(self.grid.min(), self._last_x - self.max_step)
            hi = jnp.minimum(self.grid.max(), self._last_x + self.max_step)
            hi = jnp.minimum(hi, self.grid_max)
            mask_x = (self.grid >= lo) & (self.grid <= hi)  # (N,)
            masked = jnp.where(mask_x[None, :], vals, -jnp.inf)
            flat = jnp.argmax(masked)
        else:
            flat = jnp.argmax(vals)

        N = self.grid.shape[0]
        k_best = int(flat // N)
        ix_best = int(flat % N)
        x_best = self.grid[ix_best]

        self._last_x = x_best
        return x_best, k_best, vals
