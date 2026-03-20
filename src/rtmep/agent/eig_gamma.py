import hbmep as mep
from jax import random, numpy as jnp, vmap
from jax.scipy.special import logsumexp, gammaln

from rtmep.util import execute_on_gpu
from rtmep.agent.base_agent import BaseAgent, EPS

KEYS = (
    mep.site.a, mep.site.b, mep.site.g, mep.site.h, mep.site.v,
    mep.site.c1, mep.site.c2
)
F = mep.functional.rectified_logistic


def loglik_gamma(y, alpha, beta):
    """
    Joint Gamma(site.alpha, site.beta) log-likelihood across
    R muscles (conditionally independent). Inputs broadcast to
    (..., R) -> (...).
    """
    return jnp.sum(
        alpha * jnp.log(beta)
        - gammaln(alpha)
        + (alpha - 1.0) * jnp.log(y)
        - beta * y,
        axis=-1,
    )


def _params_all(self: BaseAgent, x: float, k: int, posterior: dict):
    mu = F(x, *(posterior[u][:, k, :] for u in KEYS[:-2]), EPS)     # (S, R)
    alpha, beta = self.gamma_likelihood(
        mu,
        posterior[mep.site.c1][:, k, :],
        posterior[mep.site.c2][:, k, :]
    )   # (S, R)
    return alpha, beta


def eig_one(
    self: BaseAgent,
    x: float,
    k: int,
    posterior: dict[str, jnp.ndarray],
    key: random.key,
    num_outer: int | None = None
):
    """
    Monte Carlo MI estimate for candidate action (x, k):
        E_{θ,y}[ log p(y|θ,x,k) - log p(y|x,k) ]

    Uses:
        z ~ Gamma(site.alpha, 1) (in shape-rate parameterization)
        z / site.beta ~ Gamma(site.alpha, site.beta)
    """
    S = posterior[mep.site.a].shape[0]
    key, key_idx, key_y = random.split(key, 3)

    logw = self._logw
    if (logw is None) or (logw.shape[0] != S):
        logw = jnp.zeros(S)
    logw = logw - logsumexp(logw)
    w = jnp.exp(logw)

    alpha_all, beta_all = _params_all(self, x, k, posterior)  # (S, R)

    def body_one_draw(kkey, i):
        alpha_s, beta_s = alpha_all[i], beta_all[i]     # (R,)
        y_s = random.gamma(kkey, a=alpha_s) / beta_s    # (R,)
        loglik_s = loglik_gamma(y_s, alpha_s, beta_s)   # ()
        loglik_all = vmap(
            lambda alp_j, bet_j: loglik_gamma(y_s, alp_j, bet_j)
        )(alpha_all, beta_all)  # (S,)
        log_pred = logsumexp(loglik_all + logw)
        return loglik_s - log_pred

    if (num_outer is None) or (num_outer >= S):
        idx = jnp.arange(S)
        keys = random.split(key_y, S)
        gains = vmap(body_one_draw)(keys, idx)          # (S,)
        gain = jnp.sum(w * gains)                       # ()
    else:
        idx = random.choice(key_idx, S, (num_outer,), replace=True, p=w)
        keys = random.split(key_y, num_outer)
        gains = vmap(body_one_draw)(keys, idx)          # (S,)
        gain = gains.mean()                             # ()

    return gain


@execute_on_gpu
def eig_grid(
    self: BaseAgent, xs, ks, posterior, key, num_outer=None, batch_size=64
):
    """Return eig_vals with shape (K, N)."""
    N = xs.shape[0]
    K = self.K
    keys_k = random.split(key, K)

    def eval_one_k(k_idx, kk):
        keys_x = random.split(kk, N)

        def one_x(xx, kx):
            return eig_one(
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
