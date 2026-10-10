from typing import Union

import hbmep as mep
from pydantic import BaseModel, field_validator


class MCMC_kw(BaseModel):
    num_chains: int = 4
    num_warmup: int = 400
    num_samples: int = 400
    thinning: int = 1
    progress_bar: bool = False


class NUTS_kw(BaseModel):
    target_accept_prob: float = 0.8
    max_tree_depth: Union[int, tuple[int, int]] = (10, 10)
    dense_mass: bool | list[tuple[str, ...]] = False

    @field_validator("max_tree_depth", mode="before")
    @classmethod
    def parse_tuple(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("(") and v.endswith(")"):
                v = v[1:-1]
            if "," in v:
                return tuple(int(x.strip()) for x in v.split(","))
            return int(v)
        return v


class BaseAgent_kw(BaseModel):
    infer: str = "mcmc"
    ess_threshold: float = 0.5
    independent_update: bool = True
    initial_capacity: int = 32
    K: int = 1
    N0: int = 8
    N1: int = 3
    grid_n: int = 101
    grid_min: float = 0.
    grid_max: float = 100.
    grid_max_eff: float = 100.
    share_zero: bool = False
    share_warm: bool = False


class Uniform_kw(BaseModel):
    use_acquisition: bool = False
    update_every: int = 1


class EIG_kw(BaseModel):
    batch_size: int | None = 51


class EVR_kw(BaseModel):
    batch_size: int | None = 51
    target_parameter: str | None = mep.site.a


class PEST_kw(BaseModel):
    update_every: int = 1
    method: str = "mle1"
    mep_threshold: float = 0.05
    # run_hbmep_anyway: bool = True
    a_grid_n: int = 1001
    b_grid_n: int = 501


class FocusedEIG_kw(BaseModel):
    batch_size: int | None = 51
    focus_parameters: tuple[str, ...] = (
        mep.site.a,
        mep.site.b,
        mep.site.h,
    )
    kernel: str = "gaussian_cov"
    num_conditional: float | None = None
