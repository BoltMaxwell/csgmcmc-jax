"""Toy 25-Gaussian demo (JAX port of ``experiments/mog25.m`` / blackjax notebook).

Reproduces the sampling-book result: plain SGLD gets trapped in a mode, while
cyclical SGLD explores the full mixture. Runs the cyclical sampler with *both*
backends (``blackjax`` and the hand-rolled ``jax``) and reports how many of the
25 modes each run visits.

Run:
    python -m csgmcmc_jax.toy_mog                 # quick run + mode counts
    python -m csgmcmc_jax.toy_mog --plot          # also save figs/ (needs scipy/mpl)
"""

import argparse
import functools

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from .samplers import build_schedule
from .samplers.blackjax_backend import make_sgld as bjx_sgld
from .samplers.jax_backend import make_sgld as jax_sgld

# --- Target: mixture of 25 isotropic Gaussians on a 5x5 grid (notebook setup) ---
_GRID = np.asarray([-4, -2, 0, 2, 4], dtype=np.float32)
MU = jnp.asarray(np.stack(np.meshgrid(_GRID, _GRID), -1).reshape(-1, 2))
_LAMBDA = 1.0 / MU.shape[0]
_SIGMA = 0.03 * jnp.eye(2)


def logprob_fn(x):
    return jnp.sum(
        jnp.log(_LAMBDA)
        + jsp.special.logsumexp(
            jax.scipy.stats.multivariate_normal.logpdf(x, MU, _SIGMA), axis=-1
        )
    )


grad_logprob = jax.grad(logprob_fn)


def _explore_step(position, step_size):
    """Exploration phase: plain gradient ascent on logprob (no noise)."""
    return position + step_size * grad_logprob(position)


def run_cyclical(backend, key, num_steps=50_000, num_cycles=30,
                 init_step_size=0.09, exploration_ratio=0.25, temperature=1.0):
    """Cyclical SGLD. Returns (all_positions, do_sample_mask) over the whole run."""
    sgld_step = {"blackjax": bjx_sgld, "jax": jax_sgld}[backend]()
    schedule_fn = build_schedule(num_steps, num_cycles, init_step_size, exploration_ratio)

    def sample_branch(args):
        rng, pos, ss = args
        return sgld_step(rng, pos, grad_logprob(pos), ss, temperature)

    def explore_branch(args):
        _, pos, ss = args
        return _explore_step(pos, ss)

    def body(carry, step_id):
        pos, rng = carry
        rng, sub = jax.random.split(rng)
        sched = schedule_fn(step_id)
        pos = jax.lax.cond(sched.do_sample, sample_branch, explore_branch,
                           (sub, pos, sched.step_size))
        return (pos, rng), (pos, sched.do_sample)

    init_key, scan_key = jax.random.split(key)
    init_pos = -10 + 20 * jax.random.uniform(init_key, shape=(2,))
    (_, _), (positions, mask) = jax.lax.scan(
        body, (init_pos, scan_key), jnp.arange(num_steps)
    )
    return positions, mask


def run_plain_sgld(key, num_steps=50_000, init_step_size=0.05, gamma=0.55, temperature=1.0):
    """Decreasing-step-size SGLD (the baseline that gets stuck)."""
    sgld_step = jax_sgld()
    steps = jnp.arange(1, num_steps + 1)
    schedule = init_step_size * steps ** (-gamma)

    def body(carry, ss):
        pos, rng = carry
        rng, sub = jax.random.split(rng)
        pos = sgld_step(sub, pos, grad_logprob(pos), ss, temperature)
        return (pos, rng), pos

    init_key, scan_key = jax.random.split(key)
    init_pos = -10 + 20 * jax.random.uniform(init_key, shape=(2,))
    (_, _), positions = jax.lax.scan(body, (init_pos, scan_key), schedule)
    return positions


def count_modes(samples, radius=0.5):
    """Number of the 25 modes that have at least one sample within ``radius``."""
    samples = np.asarray(samples)
    mu = np.asarray(MU)
    d = np.linalg.norm(samples[:, None, :] - mu[None, :, :], axis=-1)  # (N, 25)
    nearest = d.argmin(axis=1)
    within = d.min(axis=1) < radius
    return len(np.unique(nearest[within]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--cycles", type=int, default=30)
    p.add_argument("--seed", type=int, default=20200203)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    key = jax.random.key(args.seed)
    k_plain, k_bjx, k_jax = jax.random.split(key, 3)

    run_cyc = jax.jit(functools.partial(run_cyclical, num_steps=args.steps,
                                        num_cycles=args.cycles), static_argnums=0)

    plain = jax.block_until_ready(jax.jit(run_plain_sgld)(k_plain))
    bjx_pos, bjx_mask = jax.block_until_ready(run_cyc("blackjax", k_bjx))
    jax_pos, jax_mask = jax.block_until_ready(run_cyc("jax", k_jax))

    bjx_samples = np.asarray(bjx_pos)[np.asarray(bjx_mask)]
    jax_samples = np.asarray(jax_pos)[np.asarray(jax_mask)]

    print(f"plain SGLD            : visited {count_modes(plain):2d} / 25 modes")
    print(f"cyclical SGLD (blackjax): visited {count_modes(bjx_samples):2d} / 25 modes "
          f"({bjx_samples.shape[0]} samples)")
    print(f"cyclical SGLD (jax)     : visited {count_modes(jax_samples):2d} / 25 modes "
          f"({jax_samples.shape[0]} samples)")

    if args.plot:
        _plot(plain, bjx_samples, jax_samples)


def _plot(plain, bjx_samples, jax_samples):
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("figs", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (title, s) in zip(
        axes,
        [("plain SGLD", plain), ("cyclical SGLD (blackjax)", bjx_samples),
         ("cyclical SGLD (jax)", jax_samples)],
    ):
        ax.plot(s[:, 0], s[:, 1], "k.", ms=0.5, alpha=0.3)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_title(title)
        ax.set_aspect("equal")
    fig.tight_layout()
    out = "figs/toy_mog_jax.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
