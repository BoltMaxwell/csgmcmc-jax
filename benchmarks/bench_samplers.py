"""Speed comparison: blackjax vs hand-rolled SG-MCMC backends.

Two synchronized measurements (per the jax-fast-code contract: warm up the jit,
then time steady state with ``block_until_ready``):

  1. **Isolated sampler update** on a ResNet18-sized parameter pytree -- isolates the
     pure backend overhead (this is where the two implementations actually differ).
  2. **Full training step** (augment + forward + backward + update) on a real batch --
     realistic wall-clock, where the ResNet dominates and the backend is a rounding
     error. Reporting both is the honest answer to "which backend is faster".

CPU here is a development box; run on the GPU/cluster for representative numbers.

    JAX_PLATFORMS=cpu PYTHONPATH=. python benchmarks/bench_samplers.py --iters 30
"""

import argparse
import statistics
import time

import equinox as eqx
import jax
import jax.numpy as jnp

from csgmcmc_jax.data import make_augment
from csgmcmc_jax.models.resnet import make_resnet18
from csgmcmc_jax.samplers import build_schedule
from csgmcmc_jax.train_cifar import build_epoch_step, make_update


def _time(fn, args, iters, warmup=3):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3  # ms


def bench_isolated_update(iters):
    key = jax.random.key(0)
    model, _ = make_resnet18(key)
    params, _ = eqx.partition(model, eqx.is_inexact_array)
    grads = jax.tree_util.tree_map(lambda x: 0.01 * x, params)
    momentum = jax.tree_util.tree_map(jnp.zeros_like, params)
    rng = jax.random.key(1)

    print("\n== isolated sampler update (ResNet18-sized pytree) ==")
    for method in ("csgld", "csghmc"):
        row = []
        for backend in ("blackjax", "jax"):
            update = make_update(method, backend, alpha=0.9, weight_decay=5e-4)
            fn = eqx.filter_jit(update)
            ms = _time(fn, (rng, params, momentum, grads, 0.1, 1e-9), iters)
            row.append((backend, ms))
        print(f"  {method:7s}  " + "   ".join(f"{b}: {ms:.3f} ms" for b, ms in row))


def bench_full_step(iters, batch_size):
    key = jax.random.key(0)
    model, state = make_resnet18(key)
    params, static = eqx.partition(model, eqx.is_inexact_array)
    momentum = jax.tree_util.tree_map(jnp.zeros_like, params)
    x = jax.random.normal(jax.random.key(2), (batch_size, 3, 32, 32))
    y = jax.random.randint(jax.random.key(3), (batch_size,), 0, 10)
    augment = make_augment("cifar10")
    schedule_fn = build_schedule(100_000, 4, 0.5, 0.9)

    print(f"\n== full training step (batch={batch_size}, fwd+bwd+update) ==")
    for method in ("csgld", "csghmc"):
        row = []
        for backend in ("blackjax", "jax"):
            update = make_update(method, backend, alpha=0.9, weight_decay=5e-4)
            step = build_epoch_step(static, augment, schedule_fn, update, 1e-9)
            fn = eqx.filter_jit(step)
            carry = (params, momentum, state, jax.random.key(4))
            inp = (jnp.asarray(60_000), x, y)  # step_id in the sampling phase
            ms = _time(fn, (carry, inp), iters)
            row.append((backend, ms))
        print(f"  {method:7s}  " + "   ".join(f"{b}: {ms:.3f} ms" for b, ms in row))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--full", action="store_true", help="also run the full-step benchmark (slow on CPU)")
    args = p.parse_args()
    print(f"backend: {jax.default_backend()}  devices: {jax.devices()}")
    bench_isolated_update(args.iters)
    if args.full:
        bench_full_step(args.iters, args.batch_size)


if __name__ == "__main__":
    main()
