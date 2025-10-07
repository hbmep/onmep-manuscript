import time
from dataclasses import dataclass, field

import numpy as np
import jax.numpy as jnp


@dataclass(frozen=True)
class Observation:
    response: np.ndarray        # shape (num_response,)
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
    def __init__(self, *, R: int | None = None, dtype=float, initial_capacity: int = 32):
        self.R: int | None = None if R is None else int(R)
        self.dtype = dtype
        self._cap = int(initial_capacity)
        self._n = 0
        self._X = np.empty((self._cap, 1), dtype=dtype)
        self._Y = np.empty((self._cap, 0 if self.R is None else self.R), dtype=dtype)

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
    def X(self) -> np.ndarray:
        return self._X[:self._n, :]

    @property
    def Y(self) -> np.ndarray:
        return self._Y[:self._n, :]

    def clear(self) -> None:
        self._n = 0

    def _grow(self, new_cap: int) -> None:
        new_cap = max(new_cap, self._cap * 2)
        X_new = np.empty((new_cap, 1), dtype=self.dtype)
        X_new[:self._n, :] = self._X[:self._n, :]
        self._X = X_new

        if self.R is None:
            # Still unknown, keep 0 columns until first append sets R
            Y_new = np.empty((new_cap, 0), dtype=self.dtype)
        else:
            Y_new = np.empty((new_cap, self.R), dtype=self.dtype)
            Y_new[:self._n, :] = self._Y[:self._n, :]
        self._Y = Y_new
        self._cap = new_cap

    def append(self, *, intensity: float, response: np.ndarray) -> None:
        r = np.asarray(response, dtype=self.dtype).reshape(-1)
        if self.R is None:
            self.R = int(r.size)
            # Allocate Y with the right column count and copy existing rows
            Y_new = np.empty((self._cap, self.R), dtype=self.dtype)
            if self._n:     # normally 0 here
                Y_new[:self._n, :] = self._Y[:self._n, :]
            self._Y = Y_new
        elif int(r.size) != self.R:
            raise ValueError(f"Response length {r.size} != fixed R {self.R}")

        if self._n >= self._cap:
            self._grow(self._cap * 2)

        self._X[self._n, 0] = float(intensity)
        self._Y[self._n, :] = r
        self._n += 1

    def as_jax(self):
        return jnp.asarray(self.X), jnp.asarray(self.Y)

    def to_dataframe(self, *, intensity: str, response: list[str], **kw):
        import pandas as pd
        if self.R is None:
            return pd.DataFrame(columns=[intensity] + list(response))
        if len(response) != self.R:
            raise ValueError(f"response columns({len(response)}) != R({self.R})")
        data = {intensity: self.X.ravel()}
        for j, col in enumerate(response):
            data[col] = self.Y[:, j]
        return pd.DataFrame(data)

    def resize_R(self, R_new: int) -> None:
        R_new = int(R_new)
        if self.R == R_new:
            return
        Y_new = np.empty((self._cap, R_new), dtype=self.dtype)
        # Copy over as many existing response columns as we can
        old_cols = self._Y.shape[1]
        copy_cols = min(old_cols, R_new)
        if copy_cols and self._n:
            Y_new[:self._n, :copy_cols] = self._Y[:self._n, :copy_cols]
        self._Y = Y_new
        self.R = R_new

    def to_state(self) -> dict:
        return {
            "R": self.R,
            "_cap": self._cap,
            "_n": self._n,
            "_X": self._X[:self._n, :].copy(),
            "_Y": self._Y[:self._n, :].copy(),
            "dtype": str(np.dtype(self.dtype).name),
        }

    @classmethod
    def from_state(cls, state: dict):
        dtype = np.dtype(state.get("dtype", float))
        cap   = int(state["_cap"])
        n   = int(state["_n"])
        R   = None if state["R"] is None else int(state["R"])

        obj = cls(R=R, dtype=dtype, initial_capacity=cap)
        obj._n   = n

        # Restore arrays
        x_src = state["_X"]
        y_src = state["_Y"]
        assert x_src.shape == (n, 1), f"X shape mismatch: {x_src.shape} vs {(n, 1)}"
        obj._X[:n, :] = x_src

        y_cols = 0 if R is None else R
        assert y_src.shape == (n, y_cols), f"Y shape mismatch: {y_src.shape} vs {(n, y_cols)}"
        if y_cols:
            obj._Y[:n, :] = y_src

        return obj
