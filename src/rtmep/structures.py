import time
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Observation:
    response: jnp.ndarray   # (R,)
    intensity: float
    t_sec: float = field(default_factory=lambda: time.time())
    info: dict = field(default_factory=dict)


class ObsArrays:
    """
    Growable (X, Y) buffer with shapes:
      X: (N, 1)     (intensity)
      Y: (N, R)     (response)
    Appends are O(1) amortized via capacity doubling.
    """
    def __init__(
        self,
        *,
        R: int | None = None,
        initial_capacity: int = 32,
    ):
        self.R: int | None = None if R is None else int(R)
        self.dtype = jnp.dtype(jnp.float64)
        self._cap = int(initial_capacity)
        self._n = 0

        self._X = jnp.empty((self._cap, 1), dtype=self.dtype)
        self._Y = jnp.empty((self._cap, 0 if self.R is None else self.R), dtype=self.dtype)

    def __len__(self) -> int:
        return self._n

    @property
    def n(self): return self._n

    @property
    def capacity(self): return self._cap

    @property
    def shape(self): return (self._n, self.R)

    @property
    def capacity_shape(self): return (self._cap, self.R)

    @property
    def X(self) -> jax.Array:
        return self._X[:self._n, :]

    @property
    def Y(self) -> jax.Array:
        return self._Y[:self._n, :]

    def clear(self) -> None:
        self._n = 0

    def _grow(self, new_cap: int) -> None:
        new_cap = max(new_cap, self._cap * 2)
        X_new = jnp.empty((new_cap, 1), dtype=self.dtype)
        X_new = X_new.at[:self._n, :].set(self._X[:self._n, :])

        if self.R is None:
            Y_new = jnp.empty((new_cap, 0), dtype=self.dtype)
        else:
            Y_new = jnp.empty((new_cap, self.R), dtype=self.dtype)
            Y_new = Y_new.at[:self._n, :].set(self._Y[:self._n, :])

        self._X = X_new
        self._Y = Y_new
        self._cap = new_cap

    def append(self, *, intensity: float, response: jnp.ndarray) -> None:
        x = jnp.asarray(intensity, dtype=self.dtype)
        r = jnp.asarray(response, dtype=self.dtype).reshape(-1)

        if self.R is None:
            self.R = int(r.size)
            Y_new = jnp.empty((self._cap, self.R), dtype=self.dtype)
            self._Y = Y_new
        elif int(r.size) != self.R:
            raise ValueError(f"Response length {r.size} != fixed R {self.R}")

        if self._n >= self._cap:
            self._grow(self._cap * 2)

        self._X = self._X.at[self._n, 0].set(x)
        self._Y = self._Y.at[self._n, :].set(r)
        self._n += 1

    def resize_R(self, R_new: int) -> None:
        R_new = int(R_new)
        if self.R == R_new:
            return

        Y_new = jnp.empty((self._cap, R_new), dtype=self.dtype)
        if self.R is not None:
            copy_cols = min(self.R, R_new)
            if copy_cols and self._n:
                Y_new = Y_new.at[:self._n, :copy_cols].set(self._Y[:self._n, :copy_cols])

        self._Y = Y_new
        self.R = R_new

    def to_df(self, *, intensity: str, response: list[str], **kw):
        import pandas as pd
        if self.R is None:
            return pd.DataFrame(columns=[intensity] + list(response))
        if len(response) != self.R:
            raise ValueError(f"response columns({len(response)}) != R({self.R})")

        X_host = np.asarray(self.X)
        Y_host = np.asarray(self.Y)
        data = {intensity: X_host.ravel()}
        for j, col in enumerate(response):
            data[col] = Y_host[:, j]
        return pd.DataFrame(data)

    def to_state(self) -> dict:
        return {
            "R": self.R,
            "dtype": str(np.dtype(self.dtype.name).name),
            "_cap": self._cap,
            "_n": self._n,
            "X": np.asarray(self.X),
            "Y": np.asarray(self.Y),
        }

    @classmethod
    def from_state(cls, state: dict):
        dtype = jnp.dtype(state.get("dtype", jnp.float64))
        cap = int(state["_cap"])
        n = int(state["_n"])
        R = None if state["R"] is None else int(state["R"])

        obj = cls(R=R, initial_capacity=cap)
        obj._n = n

        # Restore arrays
        x_src = jnp.asarray(state["X"], dtype=dtype)
        assert x_src.shape == (n, 1), f"X shape mismatch: {x_src.shape} vs {(n, 1)}"
        obj._X = obj._X.at[:n, :].set(x_src)

        y_cols = 0 if R is None else R
        y_src = jnp.asarray(state["Y"], dtype=dtype)
        assert y_src.shape == (n, y_cols), f"Y shape mismatch: {y_src.shape} vs {(n, y_cols)}"
        if y_cols:
            obj._Y = obj._Y.at[:n, :].set(y_src)
        else:
            obj._Y = jnp.empty((obj._cap, 0), dtype=dtype)

        return obj
