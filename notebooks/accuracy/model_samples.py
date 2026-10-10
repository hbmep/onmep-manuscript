import hbmep as mep
import numpy as np
from jax import numpy as jnp
import numpyro as pyro
import numpyro.distributions as dist

EPS = 1e-3


class HBestimation(mep.BaseModel):
    def __init__(self, *args, **kw):
        super(HBestimation, self).__init__(*args, **kw)
        self.log_error_threshold = 0

    @staticmethod
    def _sample_priors(features, num_features, num_response):
        b_scale = pyro.sample(mep.site.b.scale, dist.LogNormal(0., 1.))
        c1_scale = pyro.sample(mep.site.c1.scale, dist.LogNormal(0., 1.))

        with pyro.plate(mep.site.num_response, num_response):
            with pyro.plate(mep.site.num_features[0], num_features[0]):
                b_log_raw = pyro.sample("b_log_raw", dist.Normal(0., 1.))
                b = jnp.exp(b_scale * b_log_raw)

                c1_raw = pyro.sample(mep.site.c1.raw, dist.HalfNormal(1.))
                c1 = c1_scale * c1_raw
    
        params = {mep.site.b: b[*features.T], mep.site.c1: c1[*features.T]}
        return params

    @staticmethod
    def _sample_delta(features, num_features, num_response):
        num_fixed = 1
        num_delta = num_response - 1

        a_fixed_loc = pyro.sample("a_fixed_loc", dist.Normal(jnp.log2(24.), 2.))
        a_fixed_scale = pyro.sample("a_fixed_scale", dist.HalfNormal(2.))

        with pyro.plate("num_fixed", num_fixed):
            with pyro.plate(mep.site.num_features[0], num_features[0]):
                a_fixed_raw = pyro.sample("a_fixed_raw", dist.Normal(0., 1.))
                a_fixed = a_fixed_loc + a_fixed_scale * a_fixed_raw

        with pyro.plate("num_delta", num_delta):
            a_delta_loc = pyro.sample("a_delta_loc", dist.Normal(0., 2.))
            a_delta_scale = pyro.sample("a_delta_scale", dist.HalfNormal(2.))

            with pyro.plate(mep.site.num_features[0], num_features[0]):
                a_delta_raw = pyro.sample("a_delta_raw", dist.Normal(0., 1.))
                a_delta = a_delta_loc + a_delta_scale * a_delta_raw
                a_fixed_plus_delta = a_fixed + a_delta

        a = pyro.deterministic(
            mep.site.a,
            jnp.concatenate([a_fixed, a_fixed_plus_delta], axis=-1),
        )

        params = {mep.site.a: a[*features.T]}
        return params

    def power_decay(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        params = {}

        params.update(self._sample_delta(
            features=features,
            num_features=num_features,
            num_response=self.num_response,
        ))

        params.update(self._sample_priors(
            features=features,
            num_features=num_features,
            num_response=self.num_response,
        ))

        with pyro.plate(mep.site.num_response, self.num_response):
            with pyro.plate(mep.site.num_data, num_data):
                mu = -params[mep.site.b] * (intensity - params[mep.site.a])
                mu += self.log_error_threshold
                pyro.deterministic(mep.site.mu, mu)
                pyro.sample(
                    mep.site.obs,
                    dist.Normal(loc=mu, scale=params[mep.site.c1]),
                    obs=response,
                )

    def broken_power_decay(self, intensity, features, response=None, **kw):
        num_data = intensity.shape[0]
        num_features = np.max(features, axis=0) + 1

        params = {}

        params.update(self._sample_delta(
            features=features,
            num_features=num_features,
            num_response=self.num_response,
        ))

        params.update(self._sample_priors(
            features=features,
            num_features=num_features,
            num_response=self.num_response,
        ))

        # Late to early form
        with pyro.plate(mep.site.num_response, self.num_response):
            with pyro.plate(mep.site.num_features[0], num_features[0]):
                b_early_frac = pyro.sample("b_early_frac", dist.Beta(5., 2.))
                tau = pyro.sample("tau", dist.Normal(jnp.log2(24.), 2.))

        b_late = params[mep.site.b]
        b_early = b_late / b_early_frac[*features.T]

        with pyro.plate(mep.site.num_response, self.num_response):
            with pyro.plate(mep.site.num_data, num_data):
                hinge_x = mep.functional.smooth_max(intensity - tau[*features.T], eps=EPS)
                hinge_a = mep.functional.smooth_max(params[mep.site.a] - tau[*features.T], eps=EPS)

                mu = (
                    -b_early * (intensity - params[mep.site.a])
                    - (b_late - b_early) * (hinge_x - hinge_a)
                )
                mu += self.log_error_threshold
                pyro.deterministic(mep.site.mu, mu)

                pyro.sample(
                    mep.site.obs,
                    dist.Normal(loc=mu, scale=params[mep.site.c1]),
                    obs=response,
                )

    def state_dict(self):
        d = super().state_dict()
        d["log_error_threshold"] = self.log_error_threshold
        return d
    
    def load_state_dict(self, state):
        d = super().load_state_dict(state)
        self.log_error_threshold = d["log_error_threshold"]
        return self
