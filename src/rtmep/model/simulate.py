import logging

import hbmep as mep
import numpy as np
import jax
import jax.numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist

logger = logging.getLogger(__name__)
EPS = 1e-3


class Simulator(mep.BaseModel):
    def __init__(self, *args, **kw):
        super(Simulator, self).__init__(*args, **kw)
        self.simulate = False
        self.use_mixture = False

    def hb_rl(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        # Mask missing observations
        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        # Hyper-priors
        a_scale = pyro.sample(mep.site.a.scale, dist.HalfNormal(50.))
        b_scale = pyro.sample(mep.site.b.scale, dist.HalfNormal(1.))

        h_scale = pyro.sample(mep.site.h.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(mep.site.g.scale, dist.HalfNormal(.1))
        v_scale = pyro.sample(mep.site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(mep.site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(mep.site.c2.scale, dist.HalfNormal(.5))

        # Priors
        with pyro.plate(mep.site.num_response, self.num_response):
            # Hyper-priors
            a_loc = pyro.sample(mep.site.a.loc, dist.TruncatedNormal(50., 50., low=0))
            b_loc = pyro.sample(mep.site.b.loc, dist.TruncatedNormal(0.5, 1.))
            h_loc = pyro.sample(mep.site.h.loc, dist.TruncatedNormal(3., 5.))

            with pyro.plate_stack(mep.site.num_features, num_features, rightmost_dim=-2):
                a = pyro.sample(mep.site.a, dist.TruncatedNormal(a_loc, a_scale, low=0))
                b = pyro.sample(mep.site.b, dist.TruncatedNormal(b_loc, b_scale, low=0))
                h = pyro.sample(mep.site.h, dist.TruncatedNormal(h_loc, h_scale, low=0))

                g = pyro.sample(mep.site.g, dist.HalfNormal(g_scale))
                v = pyro.sample(mep.site.v, dist.HalfNormal(v_scale))

                c1 = pyro.sample(mep.site.c1, dist.HalfNormal(c1_scale))
                c2 = pyro.sample(mep.site.c2, dist.HalfNormal(c2_scale))

        # Outlier probability
        if self.use_mixture:
            q = pyro.sample(mep.site.outlier_prob, dist.Uniform(0., 0.01))

        # Observation model
        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(mep.site.num_response, self.num_response):
                with pyro.plate(mep.site.num_data, num_data):
                    mu = pyro.deterministic(
                        mep.site.mu,
                        mep.functional.rectified_logistic(
                            intensity,
                            a[*features.T],
                            b[*features.T],
                            g[*features.T],
                            h[*features.T],
                            v[*features.T],
                            0.0 if self.simulate else EPS
                        )
                    )
                    alpha, beta = self.gamma_likelihood(
                        mu, c1[*features.T], c2[*features.T],
                    )

                    # Mixture distribution
                    if self.use_mixture:
                        mixing_distribution = dist.Categorical(
                            probs=jnp.stack([1 - q, q], axis=-1)
                        )
                        component_distributions=[
                            dist.Gamma(concentration=alpha, rate=beta),
                            dist.HalfNormal(
                                scale=(g[*features.T] + h[*features.T])
                            )
                        ]
                        Mixture = dist.MixtureGeneral(
                            mixing_distribution=mixing_distribution,
                            component_distributions=component_distributions
                        )

                    # Observations
                    y_ = pyro.sample(
                        mep.site.obs,
                        (
                            Mixture if self.use_mixture
                            else dist.Gamma(concentration=alpha, rate=beta)
                        ),
                        obs=response
                    )

                    if self.use_mixture:
                        log_probs = Mixture.component_log_probs(y_)
                        pyro.deterministic(
                            "p", log_probs - jax.nn.logsumexp(
                                log_probs, axis=-1, keepdims=True
                            )
                        )
