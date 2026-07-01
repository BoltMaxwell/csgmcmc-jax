"""Sampler correctness tests (runnable directly or via pytest).

    JAX_PLATFORMS=cpu python tests/test_samplers.py
"""

import jax
import jax.numpy as jnp
import numpy as np

from csgmcmc_jax.samplers import build_schedule
from csgmcmc_jax.samplers.blackjax_backend import make_sgld as bjx_sgld, make_sghmc as bjx_sghmc
from csgmcmc_jax.samplers.jax_backend import make_sgld as jax_sgld, make_sghmc as jax_sghmc


def _toy_position(key):
    k1, k2 = jax.random.split(key)
    return {"a": jax.random.normal(k1, (5, 7)), "b": jax.random.normal(k2, (3,))}


def test_sgld_backends_bit_equivalent():
    """blackjax and hand-rolled SGLD must coincide for identical inputs -- this is
    what makes the speed comparison apples-to-apples."""
    key = jax.random.key(0)
    pos = _toy_position(key)
    grad = jax.tree_util.tree_map(lambda x: jnp.sin(x), pos)
    rng = jax.random.key(42)
    out_b = bjx_sgld()(rng, pos, grad, 0.01, 0.7)
    out_j = jax_sgld()(rng, pos, grad, 0.01, 0.7)
    for kk in pos:
        assert np.allclose(out_b[kk], out_j[kk], atol=1e-6, rtol=1e-6), kk
    print("[ok] SGLD backends bit-equivalent")


def test_sghmc_backends_finite_and_deterministic():
    """SGHMC backends use different (both valid) parameterizations; we only require
    each to be deterministic and finite."""
    key = jax.random.key(1)
    pos = _toy_position(key)
    mom = jax.tree_util.tree_map(jnp.zeros_like, pos)
    grad = jax.tree_util.tree_map(lambda x: jnp.cos(x), pos)
    rng = jax.random.key(7)
    for name, step in [("blackjax", bjx_sghmc(alpha=0.1)), ("jax", jax_sghmc(alpha=0.9))]:
        p1, m1 = step(rng, pos, mom, grad, 0.01, 0.5)
        p2, m2 = step(rng, pos, mom, grad, 0.01, 0.5)
        leaves = jax.tree_util.tree_leaves((p1, m1))
        assert all(bool(jnp.isfinite(x).all()) for x in leaves), name
        assert all(np.allclose(a, b) for a, b in zip(
            jax.tree_util.tree_leaves(p1), jax.tree_util.tree_leaves(p2))), name
    print("[ok] SGHMC backends finite + deterministic")


def test_sgld_zero_temperature_is_noiseless():
    """temperature=0 (exploration phase) -> pure gradient step, no noise."""
    pos = _toy_position(jax.random.key(2))
    grad = jax.tree_util.tree_map(jnp.ones_like, pos)
    out = jax_sgld()(jax.random.key(9), pos, grad, 0.1, 0.0)
    expected = jax.tree_util.tree_map(lambda p, g: p + 0.1 * g, pos, grad)
    for kk in pos:
        assert np.allclose(out[kk], expected[kk], atol=1e-7), kk
    print("[ok] zero-temperature SGLD is a plain gradient step")


def test_schedule_phases_and_cosine():
    sched = build_schedule(1000, num_cycles=4, initial_step_size=0.5, exploration_ratio=0.9)
    states = [sched(i) for i in range(1000)]
    do_sample = np.array([bool(s.do_sample) for s in states])
    step_sizes = np.array([float(s.step_size) for s in states])
    # 10% of each cycle samples -> ~10% overall
    assert abs(do_sample.mean() - 0.10) < 0.02, do_sample.mean()
    # step size starts at the cycle peak and decays to ~0 at the cycle end
    assert np.isclose(step_sizes[0], 0.5, atol=1e-6)
    assert step_sizes[249] < 1e-3  # end of first 250-step cycle
    print("[ok] schedule: ~10% sampling, cosine peak->0 per cycle")


if __name__ == "__main__":
    test_sgld_backends_bit_equivalent()
    test_sghmc_backends_finite_and_deterministic()
    test_sgld_zero_temperature_is_noiseless()
    test_schedule_phases_and_cosine()
    print("\nall sampler tests passed")
