import logging

import hbmep as mep
import numpy as np
from jax import random, numpy as jnp

from rtmep.util import execute_on_gpu
from rtmep import Observation
from rtmep.agent.base_agent import BaseAgent
from rtmep.agent.eig_gamma import eig_grid

logger = logging.getLogger(__name__)


class EIG(BaseAgent):
    def __init__(
        self, *,
        key: random.key = random.key(0),
        toml_path: str | None = None,
        config: dict | None = None,
        infer: str = "mcmc",
        initial_capacity: int = 32,
        ess_threshold: float = 0.5,
        K: int = 1,     # possible values of features[0]
        N0: int = 8,    # num observations at 0
        N1: int = 3,    # num random observations for warmup/kickstart
        grid_lo: float = 0.0,
        grid_hi: float = 100.0,
        grid_max: float = 100.0,    # TODO: Needed for LSL
        grid_n: int = 101,
        num_outer: int | None = None,
        batch_size: int | None = 64,
        max_step: float | None = None,
        policy: str = "eig",    # "eig", "random"
        update_every: int = 1,
    ):
        super().__init__(
            key=key,
            toml_path=toml_path,
            config=config,
            infer=infer,
            initial_capacity=initial_capacity,
            ess_threshold=ess_threshold,
            K=K,
            N0=N0,
        )

        if policy not in ("eig", "random"):
            raise ValueError("policy must be 'eig' or 'random'")

        self.grid_n = grid_n
        self.grid = jnp.linspace(grid_lo, grid_hi, grid_n, dtype=self.data.dtype_x)
        self.grid_f = jnp.arange(self.K, dtype=self.data.dtype_f)
        self.grid_max = grid_max

        self.N1 = N1
        self.num_outer = num_outer
        self.batch_size = batch_size
        self.max_step = max_step
        self.policy = policy
        self.update_every = max(1, update_every)

        self._last_x = None
        self._perm_x = None
        self._perm_i = 0

        self._n0_total = self.N0 * self.K
        self._n1_total = self.N1 * self.K
        self._n_start = self._n0_total + self._n1_total

        self._t = 0
        self._rand_k_i = 0
        return

    def _in_zero_phase(self) -> bool:
        return self._t < self._n0_total

    def _in_warm_phase(self) -> bool:
        return (self._t >= self._n0_total) and (self._t < self._n_start)

    def _random_propose(self) -> float:
        if (self._perm_x is None) or (self._perm_i >= self.grid.shape[0]):
            self._key, sub = random.split(self._key)
            self._perm_x = random.permutation(sub, self.grid.shape[0])
            self._perm_i = 0
        idx = self._perm_x[self._perm_i]
        self._perm_i += 1
        return self.grid[idx]

    @execute_on_gpu
    def _choose_next_action(self, posterior: dict):
        self._key, sub = random.split(self._key)
        vals = eig_grid(
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

    def eig_propose(self):
        if self._in_zero_phase():
            k = self._t % self.K
            k = jnp.array([k])
            return 0., k

        if self._in_warm_phase():
            k = (self._t - self._n0_total) % self.K
            k = jnp.array([k])
            x = self._random_propose()
            self._last_x = x
            return x, k

        if (self.policy == "random") or (self.posterior is None):
            k = self._rand_k_i % self.K
            k = jnp.array([k])
            self._rand_k_i += 1
            x = self._random_propose()
            self._last_x = x
            return x, k

        x, k, _ = self._choose_next_action(self.posterior)
        k = jnp.array([k])
        return x, k
    
    def propose(self):
        return self.eig_propose()

    def update(self, obs: Observation) -> bool:
        return self.update_on_each(obs)

    def update_on_each(self, obs: Observation) -> bool:
        self.data.append(intensity=obs.intensity, features=obs.features, response=obs.response)
        self._t = self.data.n

        if self._t >= self._n_start:
            n_after_warm = self._t - self._n_start
            if (self.posterior is None) or (n_after_warm % self.update_every == 0):
                self.mcmc, self.posterior = self.run()

        return False

    def reset(self, *args, **kw):
        super().reset(*args, **kw)
        self._last_x = None
        self._perm_x = None
        self._perm_i = 0
        self._t = self.data.n
        self._rand_k_i = 0
        return

    def state_dict(self, include_posterior: bool = True) -> dict:
        d = super().state_dict(include_posterior=include_posterior)

        perm_list = None
        if self._perm_x is not None:
            perm_list = np.asarray(self._perm_x, dtype=int).tolist()

        d["eig"] = {
            "N1": self.N1,
            "grid_lo": np.asarray(self.grid).min(),
            "grid_hi": np.asarray(self.grid).max(),
            "grid_n": self.grid_n,
            "num_outer": None if self.num_outer is None else self.num_outer,
            "batch_size": None if self.batch_size is None else self.batch_size,
            "max_step": None if self.max_step is None else self.max_step,
            "policy": str(self.policy),
            "update_every": self.update_every,
            "_last_x": None if self._last_x is None else self._last_x,
            "_perm_x": perm_list,
            "_perm_i": self._perm_i,
            "_t": self._t,
            "_rand_k_i": self._rand_k_i,
        }
        return d

    def load_state_dict(self, state: dict):
        super().load_state_dict(state)

        ed = state.get("eig", {})
        self.N1 = ed["N1"]
        grid_lo = ed["grid_lo"]
        grid_hi = ed["grid_hi"]
        grid_n = ed["grid_n"]
        self.grid_n = grid_n
        self.grid = jnp.linspace(grid_lo, grid_hi, grid_n)

        self.num_outer = ed["num_outer"]
        self.batch_size = ed["batch_size"]
        self.max_step = ed["max_step"]
        self.policy = ed["policy"]
        self.update_every = ed["update_every"]
        self._last_x = ed["_last_x"]

        perm_list = ed["_perm_x"]
        perm_i = int(ed.get("_perm_i", 0))

        if perm_list is None:
            self._perm_x = None
            self._perm_i = 0
        else:
            arr = np.asarray(perm_list, dtype=int)
            self._perm_x = arr
            self._perm_i = min(perm_i, arr.shape[0])

        self._n0_total = self.N0 * self.K
        self._n1_total = self.N1 * self.K
        self._n_start = self._n0_total + self._n1_total

        self._t = int(ed.get("_t", self.data.n))
        if self._t != self.data.n:
            logger.warning(
                f"State mismatch: _t={self._t} "
                f"but data.n={self.data.n}. Using data.n."
            )
            self._t = self.data.n
        self._rand_k_i = ed["_rand_k_i"]
        return
