import logging

import hbmep as mep
import numpy as np
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

    @staticmethod
    def _priors(*, num_features, num_response, include_c2):
        # Hyper-priors
        a_scale = pyro.sample(mep.site.a.scale, dist.HalfNormal(50.))
        b_scale = pyro.sample(mep.site.b.scale, dist.HalfNormal(1.))

        h_scale = pyro.sample(mep.site.h.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(mep.site.g.scale, dist.HalfNormal(.1))
        v_scale = pyro.sample(mep.site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(mep.site.c1.scale, dist.HalfNormal(5.))

        if include_c2:
            c2_scale = pyro.sample(mep.site.c2.scale, dist.HalfNormal(.5))

        fn_params = {}
        lik_params = {}

        # Priors
        with pyro.plate(mep.site.num_response, num_response):
            # Hyper-priors
            a_loc = pyro.sample(mep.site.a.loc, dist.TruncatedNormal(50., 50., low=0))
            b_loc = pyro.sample(mep.site.b.loc, dist.TruncatedNormal(0.5, 1., low=0))
            h_loc = pyro.sample(mep.site.h.loc, dist.TruncatedNormal(3., 5., low=0))

            with pyro.plate_stack(mep.site.num_features, num_features, rightmost_dim=-2):
                fn_params[mep.site.a] = pyro.sample(
                    mep.site.a, dist.TruncatedNormal(a_loc, a_scale, low=0)
                )
                fn_params[mep.site.b] = pyro.sample(
                    mep.site.b, dist.TruncatedNormal(b_loc, b_scale, low=0)
                )
                fn_params[mep.site.h] = pyro.sample(
                    mep.site.h, dist.TruncatedNormal(h_loc, h_scale, low=0)
                )

                fn_params[mep.site.g] = pyro.sample(mep.site.g, dist.HalfNormal(g_scale))
                fn_params[mep.site.v] = pyro.sample(mep.site.v, dist.HalfNormal(v_scale))

                lik_params[mep.site.c1] = pyro.sample(mep.site.c1, dist.HalfNormal(c1_scale))
                if include_c2:
                    lik_params[mep.site.c2] = pyro.sample(mep.site.c2, dist.HalfNormal(c2_scale))

        return fn_params, lik_params

    @staticmethod
    def _observe(
        *,
        features,
        response,
        use_mixture,
        num_response,
        num_data,
        mu,
        lik,
        lik_params,
    ):
        # Mask missing observations
        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        # Outlier probability
        if use_mixture:
            if response is None:
                q = 0.
            else:
                q = pyro.sample(mep.site.outlier_prob, dist.Uniform(0., 0.01))

        # Observation model
        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(mep.site.num_response, num_response):
                with pyro.plate(mep.site.num_data, num_data):
                    pyro.deterministic(mep.site.mu, mu)
                    if lik == "gamma":
                        alpha, beta = mep.BaseModel.gamma_likelihood(
                            mu,
                            lik_params[mep.site.c1][*features.T],
                            lik_params[mep.site.c2][*features.T]
                        )
                        fg = dist.Gamma(concentration=alpha, rate=beta)
                    elif lik == "lognormal":
                        loc = jnp.log(mu)
                        scale = lik_params[mep.site.c1][*features.T]
                        fg = dist.LogNormal(loc=loc, scale=scale)
                    else:
                        raise ValueError

                    # Mixture distribution
                    if use_mixture:
                        mixing_distribution = dist.Categorical(
                            probs=jnp.stack([1 - q, q], axis=-1)
                        )
                        component_distributions = [
                            fg, dist.HalfNormal(scale=(10.))
                        ]
                        dist_ = dist.MixtureGeneral(
                            mixing_distribution=mixing_distribution,
                            component_distributions=component_distributions,
                        )
                    else:
                        dist_ = fg

                    # Observations
                    pyro.sample(mep.site.obs, dist_, obs=response)

    def rl_gamma(self, intensity, features, response=None, **kw):
        include_c2 = True
        lik = "gamma"

        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        fn_params, lik_params = self._priors(
            num_features=num_features,
            num_response=self.num_response,
            include_c2=include_c2,
        )

        mu = mep.functional.rectified_logistic(
            intensity,
            **{u: v[*features.T] for u, v in fn_params.items()},
            eps=0.0 if self.simulate else EPS
        )

        self._observe(
            features=features,
            response=response,
            use_mixture=self.use_mixture,
            num_response=self.num_response,
            num_data=num_data,
            mu=mu,
            lik=lik,
            lik_params=lik_params,
        )

    def rl_lognormal(self, intensity, features, response=None, **kw):
        include_c2 = False
        lik = "lognormal"

        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        fn_params, lik_params = self._priors(
            num_features=num_features,
            num_response=self.num_response,
            include_c2=include_c2,
        )

        mu = mep.functional.rectified_logistic(
            intensity,
            **{u: v[*features.T] for u, v in fn_params.items()},
            eps=0.0 if self.simulate else EPS
        )

        self._observe(
            features=features,
            response=response,
            use_mixture=self.use_mixture,
            num_response=self.num_response,
            num_data=num_data,
            mu=mu,
            lik=lik,
            lik_params=lik_params,
        )

    def state_dict(self) -> dict:
        state = super().state_dict()
        state.update({
            "simulate": self.simulate,
        })
        return state

    def load_state_dict(self, state: dict):
        super().load_state_dict(state)
        self.simulate = state.get("simulate", self.simulate)
        return self
