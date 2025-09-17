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
        self._cap = max(1, int(initial_capacity))
        self._n = 0
        self._X = np.empty((self._cap, 1), dtype=dtype)
        self._Y = np.empty((self._cap, 0 if self.R is None else self.R), dtype=dtype)

    def __len__(self) -> int:
        return self._n

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
