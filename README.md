# csgmcmc-jax

A [JAX](https://github.com/jax-ml/jax) reimplementation of **Cyclical Stochastic
Gradient MCMC** (cSG-MCMC) for Bayesian deep learning — the PyTorch original
rebuilt on [Equinox](https://github.com/patrick-kidger/equinox) (models) and
[BlackJAX](https://github.com/blackjax-devs/blackjax) (samplers).

This is a reimplementation written with the assistance of **[Claude Code](https://claude.com/claude-code)**
(Anthropic). It ports the method — it does not introduce it. All credit for the
algorithm and the original code goes to the authors:

> Ruqi Zhang, Chunyuan Li, Jianyi Zhang, Changyou Chen, Andrew Gordon Wilson.
> **Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning.** ICLR 2020 (Oral).
> [paper](https://arxiv.org/abs/1902.03932) · [original code](https://github.com/ruqizhang/csgmcmc)

```bibtex
@article{zhang2020csgmcmc,
  title={Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning},
  author={Zhang, Ruqi and Li, Chunyuan and Zhang, Jianyi and Chen, Changyou and Wilson, Andrew Gordon},
  journal={International Conference on Learning Representations},
  year={2020}
}
```

This repository contains **only the JAX reimplementation**. For the original PyTorch
and MATLAB sources — the reference training scripts, the model zoo, the MATLAB toy
experiment, and cached figures — see the original repo:
**https://github.com/ruqizhang/csgmcmc**.

### Scope of this port

| ported here | not ported — see [original repo](https://github.com/ruqizhang/csgmcmc) |
|---|---|
| cSGLD + cSGHMC training, cyclical schedule, BMA ensemble | — |
| ResNet18 (Equinox) | the other architectures (VGG, DenseNet, GoogLeNet, SENet, PNASNet, ResNeXt, MobileNet, …) |
| toy 25-Gaussian demo (`toy_mog.py`) | the original MATLAB toy (`mog25.m`, `csgld.m`, `sgld.m`) + `plot_density.ipynb` |
| CIFAR-10 / CIFAR-100 data + augmentation | — |

## Method

The cyclical idea: a cosine step-size schedule split into `M` cycles. Each cycle
does **exploration** (large step, no noise — discovers new modes) then **sampling**
(small step, Langevin noise injected — characterizes a mode), and snapshots a few
models per cycle. The snapshots form a Bayesian model-average ensemble.

The cyclical driver follows the
[BlackJAX sampling-book cyclical-SGLD recipe](https://blackjax-devs.github.io/sampling-book/algorithms/cyclical_sgld.html),
and the SG-MCMC update is provided by **two interchangeable backends** so they can be
compared:

| backend | source |
|---|---|
| `jax` | hand-rolled pure JAX, faithful to the original csgmcmc math (reproduces the paper) |
| `blackjax` | wraps `blackjax.sgmcmc.diffusions.overdamped_langevin` / `sghmc` |

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
- The canonical cSGHMC is the original's `(1-α)` momentum-buffer form, faithfully
  reproduced by the `jax` backend (~95.5% BMA). The **`blackjax` cSGHMC is an added,
  exploratory variant** — the BlackJAX sampling-book implements only cyclical *SGLD*
  and explicitly leaves cyclical SGHMC "as an exercise for the reader," so this
  combination of BlackJAX's `sghmc` diffusion with the cyclical schedule appears in
  neither source. It is **not** a drop-in equivalent: BlackJAX's damping
  `(1 - friction·step_size)` is step-size-coupled (the original's `(1-α)` is fixed),
  and rewriting the update in terms of the position increment shows its *effective*
  learning rate is `step_size²` rather than `step_size` — so at the paper's
  `lr_0=0.5` it underfits and needs its own hyperparameter search. Treat it as a
  characterization of BlackJAX's primitive, not a reproduction of the paper.

## Results (CIFAR-10, ResNet18, 200 epochs / 4 cycles, 12-member ensemble, H100)

| config | BMA accuracy | wall-clock |
|---|---|---|
| cSGLD / jax | **95.61%** | 641 s |
| cSGLD / blackjax | **95.63%** | 655 s |
| cSGHMC / jax | **95.53%** | 657 s |
| cSGHMC / blackjax *(untuned added variant)* | 46–70% (friction-dependent, see above) | 649 s |

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
