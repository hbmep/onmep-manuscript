import hbmep as mep
import numpy as np
from jax import numpy as jnp, nn
import numpyro as pyro
import numpyro.distributions as dist

EPS = 1e-3


class HBestimation(mep.BaseModel):
    def __init__(self, *args, **kw):
        super(HBestimation, self).__init__(*args, **kw)

    @staticmethod
    def _sample_priors(features, num_response):
        num_features = np.max(features, axis=0) + 1
        num_fixed = 1
        num_delta = num_features[1] - 1

        # Fixed
        a_fixed_loc = pyro.sample("a_fixed_loc", dist.TruncatedNormal(50., 50., low=0))
        a_fixed_scale = pyro.sample("a_fixed_scale", dist.HalfNormal(50.))

        with pyro.plate(mep.site.num_response, num_response):
            with pyro.plate("num_fixed", num_fixed):
                with pyro.plate(mep.site.num_features[0], num_features[0]):
                    a_fixed = pyro.sample(
                        "a_fixed",
                        dist.TruncatedNormal(a_fixed_loc, a_fixed_scale, low=0)
                    )

        # Delta
        a_delta_loc = pyro.sample("a_delta_loc", dist.Normal(0., 50.))
        a_delta_scale = pyro.sample("a_delta_scale", dist.HalfNormal(50.))

        with pyro.plate(mep.site.num_response, num_response):
            with pyro.plate("num_delta", num_delta):
                with pyro.plate(mep.site.num_features[0], num_features[0]):
                    a_delta = pyro.sample(
                        "a_delta",
                        dist.Normal(a_delta_loc, a_delta_scale)
                    )

                    # Penalty for negative a
                    penalty_for_negative_a = (
                        jnp.fabs(a_fixed + a_delta) - (a_fixed + a_delta)
                    )
                    pyro.factor("penalty_for_negative_a", -penalty_for_negative_a)

        b_scale = pyro.sample(mep.site.b.scale, dist.HalfNormal(1.))

        h_scale = pyro.sample(mep.site.h.scale, dist.HalfNormal(5.))
        g_scale = pyro.sample(mep.site.g.scale, dist.HalfNormal(.1))
        v_scale = pyro.sample(mep.site.v.scale, dist.HalfNormal(5.))

        c1_scale = pyro.sample(mep.site.c1.scale, dist.HalfNormal(5.))
        c2_scale = pyro.sample(mep.site.c2.scale, dist.HalfNormal(.5))

        fn_params = {}
        lik_params = {}

        # Priors
        with pyro.plate(mep.site.num_response, num_response):
            with pyro.plate(mep.site.num_features[1], num_features[1]):
                with pyro.plate(mep.site.num_features[0], num_features[0]):
                    fn_params[mep.site.a] = pyro.deterministic(
                        mep.site.a, jnp.concatenate([a_fixed, a_fixed + a_delta], axis=1)
                    )
                    fn_params[mep.site.b] = pyro.sample(mep.site.b, dist.HalfNormal(b_scale)
                    )

                    fn_params[mep.site.g] = pyro.sample(mep.site.g, dist.HalfNormal(g_scale))
                    fn_params[mep.site.h] = pyro.sample(mep.site.h, dist.HalfNormal(h_scale))
                    fn_params[mep.site.v] = pyro.sample(mep.site.v, dist.HalfNormal(v_scale))

                    lik_params[mep.site.c1] = pyro.sample(mep.site.c1, dist.HalfNormal(c1_scale))
                    lik_params[mep.site.c2] = pyro.sample(mep.site.c2, dist.HalfNormal(c2_scale))

        return fn_params, lik_params

    def _model(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_response = self.num_response

        fn_params, lik_params = self._sample_priors(features, num_response)

        mu = mep.functional.rectified_logistic(
            intensity,
            **{u: v[*features.T] for u, v in fn_params.items()},
            eps=EPS,
        )

        # Mask missing observations
        mask_obs = True
        if response is not None:
            mask_obs = np.invert(np.isnan(response))

        # Observation model
        with pyro.handlers.mask(mask=mask_obs):
            with pyro.plate(mep.site.num_response, num_response):
                with pyro.plate(mep.site.num_data, num_data):
                    pyro.deterministic(mep.site.mu, mu)
                    alpha, beta = mep.BaseModel.gamma_likelihood(
                        mu,
                        lik_params[mep.site.c1][*features.T],
                        lik_params[mep.site.c2][*features.T]
                    )
                    dist_ = dist.Gamma(concentration=alpha, rate=beta)
                    pyro.sample(mep.site.obs, dist_, obs=response)
