from functools import partial

import hbmep as mep
from jax import random, numpy as jnp
from jax.scipy.special import gammaln, logsumexp

from onmep.agent.likelihood import LikelihoodMixin as _LikelihoodMixin, EPS, LOG2PI


class LikelihoodMixin(_LikelihoodMixin):
    ...