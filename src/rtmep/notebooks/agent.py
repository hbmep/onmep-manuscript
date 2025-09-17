from jax import random
import numpy as np
import numpyro as pyro
from numpyro import distributions as dist

import hbmep as mep
from hbmep import site, functional as F, smooth_functional as SF
from rtmep import (
    Observation,
    BaseAgent
)


class RTAgent(BaseAgent):
    def __init__(self, *args, max_observations=24, **kw) -> None:
        BaseAgent.__init__(self, *args, **kw)
        self.max_obs = int(max_observations)

    def random_proposal(self):
        self._key, subkey = random.split(self._key)
        return float(random.randint(subkey, shape=(), minval=0, maxval=100))

    def update_on_max_obs(self, obs: Observation) -> bool:
        self.data.append(intensity=obs.intensity, response=obs.response)
        if len(self.data) < self.max_obs:
            return False
        else:
            self.mcmc, self.posterior = self.run()
            return True

    def rl(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]

        # Mask missing observations
        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        # Priors
        with pyro.plate(site.num_response, self.num_response):
            a = pyro.sample(
                site.a, dist.TruncatedNormal(50., 50., low=0)
            )
            b = pyro.sample(site.b, dist.HalfNormal(1.))

            g = pyro.sample(site.g, dist.HalfNormal(.1))
            h = pyro.sample(site.h, dist.HalfNormal(5.))
            v = pyro.sample(site.v, dist.HalfNormal(5.))

            c1 = pyro.sample(site.c1, dist.HalfNormal(5.))
            c2 = pyro.sample(site.c2, dist.HalfNormal(.5))

        # Observation model
        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(site.num_response, self.num_response):
                with pyro.plate(site.num_data, num_data):
                    mu, alpha, beta = self.gamma_likelihood(
                        SF.rectified_logistic,
                        intensity,
                        (a, b, g, h, v,),
                        c1, c2
                    )
                    pyro.deterministic(site.mu, mu)

                    # Observations
                    y_ = pyro.sample(
                        site.obs,
                        dist.Gamma(concentration=alpha, rate=beta),
                        obs=response
                    )
