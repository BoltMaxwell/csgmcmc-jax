"""Cyclical step-size schedule (Zhang et al., 2020).

Vendored from the blackjax sampling-book ``cyclical_sgld`` recipe and made
jit-friendly: ``schedule_fn(step_id)`` returns jax scalars so it can be called
from inside ``lax.scan``.

Each cycle has length ``cycle_length = num_training_steps // num_cycles``. The
step size follows a cosine decay within the cycle; the first ``exploration_ratio``
fraction of the cycle is the *exploration* phase (plain SGD, no noise) and the
remainder is the *sampling* phase (SGLD/SGHMC, with injected noise).
"""

from typing import NamedTuple

import jax.numpy as jnp


class ScheduleState(NamedTuple):
    step_size: jnp.ndarray  # cosine-annealed step size for this step
    do_sample: jnp.ndarray  # bool: True during the sampling phase of the cycle


def build_schedule(
    num_training_steps,
    num_cycles=4,
    initial_step_size=1e-3,
    exploration_ratio=0.25,
):
    """Return ``schedule_fn(step_id) -> ScheduleState``.

    ``step_id`` may be a Python int or a traced jax integer.
    """
    cycle_length = num_training_steps // num_cycles

    def schedule_fn(step_id):
        phase = (step_id % cycle_length) / cycle_length
        do_sample = phase >= exploration_ratio
        cos_out = jnp.cos(jnp.pi * phase) + 1.0
        step_size = 0.5 * cos_out * initial_step_size
        return ScheduleState(step_size, do_sample)

    return schedule_fn
