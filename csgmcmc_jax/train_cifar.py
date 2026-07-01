"""Cyclical SG-MCMC training on CIFAR-10/100 (port of ``experiments/cifar_*.py``).

Faithful to the original csgmcmc dynamics: a single cosine cyclical step size runs
throughout, and the exploration/sampling phases differ *only* by whether Langevin
noise is injected -- implemented by gating the sampler ``temperature`` to 0 during
exploration. This keeps SGHMC momentum active across both phases, exactly as the
original ``update_params`` does.

The whole epoch is a single ``lax.scan`` so it compiles to one fused region (the
right shape for GPU). The sampler backend (``blackjax`` vs hand-rolled ``jax``) and
method (``csgld`` vs ``csghmc``) are chosen at build time.

    python -m csgmcmc_jax.train_cifar --data_path PATH --dir OUT \
        --method csgld --backend jax --dataset cifar10
"""

import argparse
import os
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .data import load_cifar, make_augment
from .models.resnet import make_resnet18
from .samplers import build_schedule
from .samplers.blackjax_backend import make_sgld as bjx_sgld, make_sghmc as bjx_sghmc
from .samplers.jax_backend import make_sgld as jax_sgld, make_sghmc as jax_sghmc

_SGLD = {"blackjax": bjx_sgld, "jax": jax_sgld}
_SGHMC = {"blackjax": bjx_sghmc, "jax": jax_sghmc}


def make_loss_fn(static):
    def loss_fn(params, state, x, y):
        model = eqx.combine(params, static)
        batched = jax.vmap(model, axis_name="batch", in_axes=(0, None), out_axes=(0, None))
        logits, new_state = batched(x, state)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        return loss, new_state

    return eqx.filter_value_and_grad(loss_fn, has_aux=True)


def make_update(method, backend, *, alpha, weight_decay, bjx_friction=1.0):
    """Return ``update(key, params, momentum, grads, step_size, temperature)``.

    ``grads`` is the raw loss gradient; weight decay (the Gaussian prior) and the
    sign flip to the log-posterior ascent direction are applied here, uniformly for
    both backends and both phases.

    For blackjax cSGHMC, ``bjx_friction`` is the Ma-et-al. friction coefficient.
    NOTE: blackjax's SGHMC damping is ``(1 - friction*step_size)`` -- coupled to the
    step size -- whereas the original csgmcmc uses a *fixed* ``(1 - alpha)`` damping.
    Across the cyclical cosine schedule no single friction reproduces the original's
    heavy step-independent damping (empirically friction=0.1 -> 70% BMA, friction=1.0
    -> 46% BMA on CIFAR-10, vs ~95.5% for the hand-rolled backend). blackjax cSGHMC
    therefore needs its own hyperparameter search and does NOT drop-in reproduce the
    paper; it is a genuinely different discretization. (blackjax cSGLD, by contrast,
    is bit-equivalent to the hand-rolled backend and reproduces the paper.)
    """
    if method == "csgld":
        step = _SGLD[backend]()

        def update(key, params, momentum, grads, step_size, temperature):
            g = jax.tree_util.tree_map(lambda gg, p: -(gg + weight_decay * p), grads, params)
            params = step(key, params, g, step_size, temperature)
            return params, momentum
    elif method == "csghmc":
        step = _SGHMC["jax"](alpha) if backend == "jax" else _SGHMC["blackjax"](alpha=bjx_friction)

        def update(key, params, momentum, grads, step_size, temperature):
            g = jax.tree_util.tree_map(lambda gg, p: -(gg + weight_decay * p), grads, params)
            params, momentum = step(key, params, momentum, g, step_size, temperature)
            return params, momentum
    else:
        raise ValueError(method)
    return update


def build_epoch_step(static, augment, schedule_fn, update, temperature_eff):
    grad_fn = make_loss_fn(static)

    def step(carry, inp):
        params, momentum, state, key = carry
        step_id, x_raw, y = inp
        key, aug_key, noise_key = jax.random.split(key, 3)
        x = augment(aug_key, x_raw)
        (loss, state), grads = grad_fn(params, state, x, y)
        sched = schedule_fn(step_id)
        temperature = jnp.where(sched.do_sample, temperature_eff, 0.0)
        params, momentum = update(noise_key, params, momentum, grads, sched.step_size, temperature)
        return (params, momentum, state, key), loss

    return step


def _batch_epoch(rng, x, y, batch_size):
    """Shuffle and split into (num_batches, batch_size, ...), dropping remainder."""
    n = (x.shape[0] // batch_size) * batch_size
    perm = np.asarray(jax.random.permutation(rng, x.shape[0]))[:n]
    xb = x[perm].reshape(-1, batch_size, *x.shape[1:])
    yb = y[perm].reshape(-1, batch_size)
    return xb, yb


def train(
    data,
    *,
    method="csgld",
    backend="jax",
    epochs=200,
    num_cycles=4,
    batch_size=64,
    lr_0=0.5,
    weight_decay=5e-4,
    temperature=1.0 / 50000,
    alpha=None,
    bjx_friction=1.0,
    exploration_ratio=0.9,
    samples_per_cycle=3,
    seed=1,
    save_dir=None,
    log_every=1,
):
    """Run training; return list of saved samples ``(model, state)`` (and write them
    to ``save_dir`` if given)."""
    if alpha is None:
        alpha = 1.0 if method == "csgld" else 0.9

    key = jax.random.key(seed)
    key, model_key = jax.random.split(key)
    model, state = make_resnet18(model_key, num_classes=data["num_classes"])
    params, static = eqx.partition(model, eqx.is_inexact_array)
    momentum = jax.tree_util.tree_map(jnp.zeros_like, params)

    x_train, y_train = data["x_train"], data["y_train"]
    datasize = x_train.shape[0]
    num_batches = datasize // batch_size
    total_steps = epochs * num_batches
    temperature_eff = temperature / datasize  # original noise = sqrt(2*lr*alpha*T/N)

    augment = make_augment_for(data)
    schedule_fn = build_schedule(total_steps, num_cycles, lr_0, exploration_ratio)
    update = make_update(method, backend, alpha=alpha, weight_decay=weight_decay,
                         bjx_friction=bjx_friction)
    step = build_epoch_step(static, augment, schedule_fn, update, temperature_eff)
    run_epoch = eqx.filter_jit(partial(jax.lax.scan, step))

    epochs_per_cycle = max(epochs // num_cycles, 1)
    samples = []
    for epoch in range(epochs):
        key, shuffle_key, epoch_key = jax.random.split(key, 3)
        xb, yb = _batch_epoch(shuffle_key, x_train, y_train, batch_size)
        step_ids = epoch * num_batches + jnp.arange(xb.shape[0])
        carry = (params, momentum, state, epoch_key)
        carry, losses = run_epoch(carry, (step_ids, jnp.asarray(xb), jnp.asarray(yb)))
        params, momentum, state, _ = carry
        if epoch % log_every == 0:
            print(f"epoch {epoch:3d}  loss {float(jnp.mean(losses)):.4f}")

        in_cycle = epoch % epochs_per_cycle
        if in_cycle >= epochs_per_cycle - samples_per_cycle:
            model_s = eqx.combine(params, static)
            samples.append((model_s, state))
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                path = os.path.join(save_dir, f"sample_{len(samples) - 1:03d}.eqx")
                eqx.tree_serialise_leaves(path, (model_s, state))
    return samples


def make_augment_for(data):
    # data dict doesn't carry its name; infer by num_classes for the right stats.
    dataset = "cifar100" if data["num_classes"] == 100 else "cifar10"
    return make_augment(dataset)


def main():
    p = argparse.ArgumentParser(description="Cyclical SG-MCMC on CIFAR (JAX/Equinox)")
    p.add_argument("--data_path", required=True)
    p.add_argument("--dir", dest="save_dir", required=True)
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--method", default="csgld", choices=["csgld", "csghmc"])
    p.add_argument("--backend", default="jax", choices=["jax", "blackjax"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--cycles", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--temperature", type=float, default=1.0 / 50000)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--bjx_friction", type=float, default=1.0,
                   help="blackjax cSGHMC friction (steady-state step = step_size*g/friction)")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    data = load_cifar(args.data_path, args.dataset)
    train(
        data,
        method=args.method,
        backend=args.backend,
        epochs=args.epochs,
        num_cycles=args.cycles,
        batch_size=args.batch_size,
        lr_0=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        alpha=args.alpha,
        bjx_friction=args.bjx_friction,
        seed=args.seed,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
