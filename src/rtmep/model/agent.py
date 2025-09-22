import pandas as pd
from jax import random, Array
import numpy as np
from numpyro.infer import MCMC
import hbmep as mep
from hbmep import BaseModel
from hbmep.util import timing
from rtmep.model.structures import Observation, ObsArrays


class BaseRTAgent():
    def __init__(self):
        self.data = ObsArrays()

    def reset(self) -> None:
        self.data.clear()
    
    def propose(self) -> float:
        raise NotImplementedError

    def update(self, obs: Observation) -> bool:
        raise NotImplementedError


class BaseAgent(BaseRTAgent, BaseModel):
    def __init__(
        self,
        *,
        toml_path: str | None = None,
        config: dict | None = None,
    ) -> None:
        BaseRTAgent.__init__(self)
        BaseModel.__init__(self, toml_path=toml_path, config=config)
        self._key = random.key(0)
        self.posterior = None
        self.mcmc = None

    def reset(self) -> None:
        super().reset()
        self._key = random.key(0)
        self.posterior = None
        self.mcmc = None

    @timing
    def run(
        self,
        df: pd.DataFrame = None,
        mcmc: MCMC = None,
        extra_fields: list | tuple = (),
        init_params=None,
        key: Array | None = None,
        **kw
    ) -> tuple[MCMC, dict]:
        if df is not None:
            return super().run(
                df=df,
                mcmc=mcmc,
                extra_fields=extra_fields,
                init_params=init_params,
                key=key,
                **kw
            )
        
        if not len(self.data):
            raise ValueError("No observations in buffer, collect data before run().")

        return mep.run(
            key=self.key if key is None else key,
            model=self._model,
            intensity=self.data.X,
            features=np.empty((self.data.X.shape[0], 0)),
            response=self.data.Y,
            nuts_params=self.nuts_params,
            mcmc_params=self.mcmc_params,
            mcmc=mcmc,
            extra_fields=extra_fields,
            init_params=init_params,
            **kw
        )
