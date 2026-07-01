# csgmcmc-jax

JAX/Equinox port of **Cyclical Stochastic Gradient MCMC** (Zhang et al.,
*ICLR 2020*), forked from [ruqizhang/csgmcmc](https://github.com/ruqizhang/csgmcmc).

The cyclical idea: a cosine step-size schedule split into `M` cycles. Each cycle
does **exploration** (large step, no noise — discovers new modes) then **sampling**
(small step, Langevin noise injected — characterizes a mode), and snapshots a few
models per cycle. The snapshots form a Bayesian model-average ensemble.

This port follows the
[blackjax sampling-book cyclical-SGLD recipe](https://blackjax-devs.github.io/sampling-book/algorithms/cyclical_sgld.html)
and implements the SG-MCMC update with **two interchangeable backends** so they can
be compared:

| backend | source |
|---|---|
| `blackjax` | wraps `blackjax.sgmcmc.diffusions.overdamped_langevin` / `sghmc` |
| `jax` | hand-rolled pure JAX, faithful to the original csgmcmc math |

## Layout

```
csgmcmc_jax/
  data.py                 manual numpy CIFAR-10/100 loader + on-device crop/flip aug
  models/resnet.py        Equinox ResNet18 (stateful BatchNorm, mode="batch")
  samplers/
    cyclical.py           build_schedule (cosine step size + do_sample flag)
    blackjax_backend.py   blackjax diffusion wrappers
    jax_backend.py        hand-rolled SGLD/SGHMC
  train_cifar.py          cSGLD/cSGHMC driver (one lax.scan per epoch)
  ensemble.py             softmax-averaged BMA accuracy over saved samples
  toy_mog.py              25-Gaussian demo (both backends)
benchmarks/bench_samplers.py   synchronized blackjax-vs-handrolled timing
tests/                    sampler equivalence + CIFAR pipeline smoke tests
```

## Install

```bash
pip install -e .          # jax, equinox, blackjax, optax, numpy
pip install -e '.[plot]'  # + matplotlib/scipy for the toy figure
```

## Toy 25-Gaussian demo

```bash
python -m csgmcmc_jax.toy_mog --plot     # writes figs/toy_mog_jax.png
```
Plain SGLD stays trapped in one mode; cyclical SGLD (both backends) visits all 25.

## CIFAR training + ensemble

Download CIFAR once (not on the hot path) and point `--data_path` at the parent dir
of `cifar-10-batches-py/` (or `cifar-100-python/`):

```bash
python -m csgmcmc_jax.train_cifar --data_path DATA --dir runs/csgld \
    --method csgld --backend jax --dataset cifar10        # cSGLD
python -m csgmcmc_jax.train_cifar --data_path DATA --dir runs/csghmc \
    --method csghmc --backend blackjax --alpha 0.9         # cSGHMC

python -m csgmcmc_jax.ensemble --data_path DATA --dir runs/csgld --dataset cifar10
```

Defaults reproduce the paper run: 200 epochs, 4 cycles, `lr_0=0.5`, `wd=5e-4`,
`temperature=1/50000`, noise in the last 10% of each cycle (`exploration_ratio=0.9`),
3 snapshots/cycle → 12 ensemble members.

> **CPU note.** This repo's dev box is CPU-only — fine for the toy demo, smoke tests,
> and the benchmark harness, but **full 200-epoch ResNet18 training belongs on a
> GPU/cluster** (one CPU training step is ~0.6 s).

## Faithfulness to the original

- Exploration vs sampling differ **only** by whether Langevin noise is injected
  (gated via `temperature → 0` in exploration), so SGHMC momentum runs continuously —
  matching the original `update_params`, rather than swapping optimizers.
- The original noise scale `sqrt(2·lr·alpha·T/N)` maps onto the diffusion kernels via
  `g = -(grad + wd·p)`, `step_size = lr`, `temperature = T/N`. With this mapping the
  **two SGLD backends are bit-equivalent** (asserted in `tests/test_samplers.py`).
- cSGHMC uses two *different but valid* parameterizations (the original's `(1-α)`
  momentum buffer vs blackjax's Ma-et-al. friction). These are **not** interchangeable:
  blackjax's damping `(1 - friction·step_size)` is step-size-coupled, while the
  original's `(1-α)` is fixed — so across the cyclical schedule no single friction
  reproduces the original's damping. blackjax cSGHMC needs its own hyperparameter
  search and does not drop-in reproduce the paper (see results below).

## Results (CIFAR-10, ResNet18, 200 epochs / 4 cycles, 12-member ensemble, H100)

| config | BMA accuracy | wall-clock |
|---|---|---|
| cSGLD / jax | **95.61%** | 641 s |
| cSGLD / blackjax | **95.63%** | 655 s |
| cSGHMC / jax | **95.53%** | 657 s |
| cSGHMC / blackjax | 46–70% (friction-dependent, see above) | 649 s |

Three configs reproduce the paper. **cSGLD jax vs blackjax (95.61 vs 95.63%)** is the
clean apples-to-apples check — identical accuracy, as the bit-equivalence test
predicts. Per-run wall-clock is within noise across backends: the SG-MCMC update is a
negligible fraction of the ResNet forward/backward, so the backend choice does not
affect speed.

## Backend speed comparison

From `benchmarks/bench_samplers.py` (CPU dev box — indicative only):

- **Isolated update** on a ResNet18-sized pytree: blackjax ≈ hand-rolled (~34 ms,
  within noise) — the update is the same tree-map math.
- **Full training step** (fwd+bwd+update, batch 64): ~610 ms, dominated by the
  ResNet; backend difference is within noise.

**Takeaway:** the backend choice has negligible wall-clock impact, so the hand-rolled
`jax` backend is preferable — same speed, no blackjax dependency. Re-run on the
target GPU for representative numbers.

## Tests

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. python tests/test_samplers.py      # fast
JAX_PLATFORMS=cpu PYTHONPATH=. python tests/test_cifar_smoke.py   # ~1-2 min (compiles ResNet18)
```
