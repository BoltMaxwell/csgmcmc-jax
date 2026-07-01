"""Cyclical SG-MCMC samplers: shared schedule + two interchangeable backends.

Both backends expose the same factory interface so the training driver can swap
them without changing the loop:

    make_sgld()                 -> step(rng, position, g, step_size, temperature)
    make_sghmc(**hyperparams)   -> step(rng, position, momentum, g, step_size, temperature)

where ``g`` is the (estimated) gradient of the log-posterior (ascent direction);
the driver is responsible for forming ``g = -(grad_loss + weight_decay * position)``
and for the cyclical ``step_size`` / ``temperature``.
"""

from . import blackjax_backend, jax_backend  # noqa: F401
from .cyclical import ScheduleState, build_schedule  # noqa: F401

BACKENDS = {"blackjax": blackjax_backend, "jax": jax_backend}
