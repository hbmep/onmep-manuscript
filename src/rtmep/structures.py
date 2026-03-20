import time
import logging
from dataclasses import dataclass, field

import numpy as np
import jax.numpy as jnp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Observation:
    intensity: float                # ()
    features: jnp.ndarray           # (D,)
    response: jnp.ndarray           # (R,)
    t_sec: float = field(default_factory=lambda: time.time())
    info: dict = field(default_factory=dict)

    def __repr__(self):
        f_str = ", ".join([f"{x}" for x in self.features])
        r_str = ", ".join([f"{x:.2f}" for x in self.response])
        return (
            f"Obs(int={self.intensity:.2f}, feat=[{f_str}], "
            f"resp=[{r_str}], t={self.t_sec:.2f})"
        )

    def to_string(self):
        return self.__repr__()


class ObsArrays:
    """
    Growable (X, F, Y) buffer with shapes:
      X: (N, 1)         intensity
      F: (N, D)         features
      Y: (N, R)         response
    """
    def __init__(
        self,
        *,
        D: int | None = None,
        R: int | None = None,
        initial_capacity: int = 32,
    ):
        self.D = D
        self.R = R

        self.dtype_x = jnp.float64
        self.dtype_f = jnp.int32
        self.dtype_y = jnp.float64

        self._cap = initial_capacity
        self._n = 0     # num_data

        self._X = jnp.zeros((self._cap, 1), dtype=self.dtype_x)
        self._F = jnp.empty(
            (self._cap, 0 if self.D is None else self.D),
            dtype=self.dtype_f
        )
        self._Y = jnp.empty(
            (self._cap, 0 if self.R is None else self.R),
            dtype=self.dtype_y
        )
        return

    def __len__(self):
        return self._n

    @property
    def n(self):    # num_data
        return self._n

    @property
    def capacity(self):
        return self._cap

    @property
    def shape(self):
        return (self._n, self.D, self.R)

    @property
    def capacity_shape(self):
        return (self._cap, self.D, self.R)

    @property
    def X(self):
        return self._X[:self._n, :]

    @property
    def F(self):
        return self._F[:self._n, :]

    @property
    def Y(self):
        return self._Y[:self._n, :]

    def reset(self):
        self._n = 0
        return

    def _grow(self, new_cap: int):
        new_cap = max(new_cap, self._cap * 2)
        logger.info(f"Capacity changed from {self._cap} to {new_cap}")

        X_new = jnp.zeros((new_cap, 1), dtype=self.dtype_x)
        X_new = X_new.at[:self._n, :].set(self._X[:self._n, :])

        if self.D is None:
            F_new = jnp.empty((new_cap, 0), dtype=self.dtype_f)
        else:
            F_new = jnp.empty((new_cap, self.D), dtype=self.dtype_f)
            if self._n:
                F_new = F_new.at[:self._n, :].set(self._F[:self._n, :])

        if self.R is None:
            Y_new = jnp.empty((new_cap, 0), dtype=self.dtype_y)
        else:
            Y_new = jnp.empty((new_cap, self.R), dtype=self.dtype_y)
            if self._n:
                Y_new = Y_new.at[:self._n, :].set(self._Y[:self._n, :])

        self._X = X_new
        self._F = F_new
        self._Y = Y_new
        self._cap = new_cap
        return

    def append(self, *, intensity: float, features: jnp.ndarray, response: jnp.ndarray):
        x = jnp.asarray(intensity, dtype=self.dtype_x)
        f = jnp.asarray(features, dtype=self.dtype_f).reshape(-1)
        r = jnp.asarray(response, dtype=self.dtype_y).reshape(-1)
        
        if self.D is None:
            self.D = f.size
            self._F = jnp.empty((self._cap, self.D), dtype=self.dtype_f)
        elif f.size != self.D:
            raise ValueError(f"Features length {f.size} != fixed D {self.D}")

        if self.R is None:
            self.R = r.size
            self._Y = jnp.empty((self._cap, self.R), dtype=self.dtype_y)
        elif r.size != self.R:
            raise ValueError(f"Response length {r.size} != fixed R {self.R}")

        if self._n >= self._cap:
            self._grow(self._cap * 2)

        self._X = self._X.at[self._n, 0].set(x)
        self._F = self._F.at[self._n, :].set(f)
        self._Y = self._Y.at[self._n, :].set(r)
        self._n += 1
        return

    def resize_R(self, R_new: int):
        R_new = R_new
        if self.R == R_new:
            return

        Y_new = jnp.empty((self._cap, R_new), dtype=self.dtype_y)
        if self.R is not None:
            copy_cols = min(self.R, R_new)
            if copy_cols and self._n:
                Y_new = Y_new.at[:self._n, :copy_cols].set(self._Y[:self._n, :copy_cols])

        self._Y = Y_new
        self.R = R_new
        return

    # TODO
    def resize_F(self, F_new: int):
        raise NotImplementedError

    def to_df(self, *, intensity: str, features: list[str], response: list[str], **kw):
        import pandas as pd
        if self.R is None:
            return pd.DataFrame(columns=[intensity] + list(features) + list(response))
        if len(features) != self.D:
            raise ValueError(f"feature columns({len(features)}) != D({self.D})")
        if len(response) != self.R:
            raise ValueError(f"response columns({len(response)}) != R({self.R})")

        X_host = np.asarray(self.X)
        F_host = np.asarray(self.F)
        Y_host = np.asarray(self.Y)
        data = {intensity: X_host.ravel()}
        for j, col in enumerate(features):
            data[col] = F_host[:, j]
        for j, col in enumerate(response):
            data[col] = Y_host[:, j]
        return pd.DataFrame(data)

    def to_state(self) -> dict:
        return {
            "D": self.D,
            "R": self.R,
            "_cap": self._cap,
            "_n": self._n,
            "X": np.asarray(self.X),
            "F": np.asarray(self.F),
            "Y": np.asarray(self.Y),
        }

    @classmethod
    def from_state(cls, state: dict):
        cap = state["_cap"]
        n = state["_n"]
        D = state["D"]
        R = state["R"]

        obj = cls(D=D, R=R, initial_capacity=cap)
        obj._n = n

        # Restore arrays
        x_src = jnp.asarray(state["X"], dtype=obj.dtype_x)
        f_src = jnp.asarray(state["F"], dtype=obj.dtype_f)
        y_src = jnp.asarray(state["Y"], dtype=obj.dtype_y)

        assert x_src.shape == (n, 1)
        f_cols = 0 if D is None else D
        assert f_src.shape == (n, f_cols)
        y_cols = 0 if R is None else R
        assert y_src.shape == (n, y_cols)

        obj._X = obj._X.at[:n, :].set(x_src)
        if f_cols:
            obj._F = obj._F.at[:n, :].set(f_src)
        else:
            obj._F = jnp.empty((obj._cap, 0), dtype=obj.dtype_f)
        if y_cols:
            obj._Y = obj._Y.at[:n, :].set(y_src)
        else:
            obj._Y = jnp.empty((obj._cap, 0), dtype=obj.dtype_y)

        return obj
