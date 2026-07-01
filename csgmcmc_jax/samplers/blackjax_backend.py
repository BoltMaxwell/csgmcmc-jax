"""blackjax-backed SG-MCMC updates.

Thin wrappers around ``blackjax.sgmcmc.diffusions`` so the update math is owned by
blackjax. The factories return pure update closures matching ``jax_backend`` so the
training driver can swap backends with no other change.

Conventions (blackjax / Ma et al. 2015):
  * SGLD  : position += step_size * g + sqrt(2 * temperature * step_size) * noise
  * SGHMC : position += step_size * momentum                                 (old momentum)
            momentum += -alpha*step_size*momentum + step_size * g
                        + sqrt(step_size*temperature*(2*alpha - step_size*temperature*beta)) * noise
``g`` is the log-posterior gradient (ascent direction).
"""

from blackjax.sgmcmc import diffusions


def make_sgld():
    """Overdamped Langevin (SGLD) step.

    Numerically identical to ``jax_backend.make_sgld`` for the same inputs, which
    the equivalence test relies on.
    """
    one_step = diffusions.overdamped_langevin()

    def step(rng_key, position, g, step_size, temperature):
        return one_step(rng_key, position, g, step_size, temperature)

    return step


def make_sghmc(alpha=0.01, beta=0.0):
    """SGHMC step with blackjax's friction parameterization.

    Note ``alpha`` here is the *friction* coefficient (Ma et al.), NOT the
    ``1 - momentum`` knob used by the original csgmcmc code path in
    ``jax_backend.make_sghmc`` -- the two SGHMC backends are valid but distinct
    parameterizations and are compared on speed/quality, not bit-equivalence.
    """
    one_step = diffusions.sghmc(alpha, beta)

    def step(rng_key, position, momentum, g, step_size, temperature):
        return one_step(rng_key, position, momentum, g, step_size, temperature)

    return step
