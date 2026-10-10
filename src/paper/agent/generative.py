import hbmep as mep
from jax import numpy as jnp, lax
import numpyro as pyro
import numpyro.distributions as dist

from onmep.agent.generative import GenerativeMixin as _GenerativeMixin
from onmep.agent.likelihood import EPS


class GenerativeMixin(_GenerativeMixin):
    ...
