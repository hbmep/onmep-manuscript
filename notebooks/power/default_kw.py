from pydantic import BaseModel, Field

from paper.default_kw import (
    MCMC_kw,
    NUTS_kw,
    BaseAgent_kw,
    Uniform_kw,
    EIG_kw as _EIG_kw,
    EVR_kw as _EVR_kw,
    PEST_kw,
    FocusedEIG_kw as _FocusedEIG_kw,
)


class EIG_kw(_EIG_kw):
    batch_size: int = 21


class EVR_kw(_EVR_kw):
    batch_size: int = 21


class FocusedEIG_kw(_FocusedEIG_kw):
    batch_size: int = 21
