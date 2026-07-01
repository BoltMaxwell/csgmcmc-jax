"""Bayesian model averaging over saved cSG-MCMC samples (port of
``experiments/cifar_ensemble.py``).

Each sample is a serialized ``(model, state)``. Predictions are softmax-averaged
across samples (the BMA predictive), then argmax-ed for accuracy.

    python -m csgmcmc_jax.ensemble --data_path PATH --dir SAMPLES --dataset cifar10
"""

import argparse
import glob
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .data import load_cifar
from .models.resnet import make_resnet18


@eqx.filter_jit
def _probs(model, state, x):
    """Softmax probabilities for a batch, in inference mode (running BN stats)."""
    model = eqx.nn.inference_mode(model)
    batched = jax.vmap(model, axis_name="batch", in_axes=(0, None), out_axes=(0, None))
    logits, _ = batched(x, state)
    return jax.nn.softmax(logits, axis=-1)


def predict_mean(model, state, x_test, batch_size=200):
    """Mean softmax probs of one sample over the test set."""
    n = x_test.shape[0]
    out = []
    for i in range(0, n, batch_size):
        out.append(np.asarray(_probs(model, state, jnp.asarray(x_test[i:i + batch_size]))))
    return np.concatenate(out, axis=0)


def evaluate(samples, x_test, y_test, batch_size=200):
    """Return (bma_accuracy, per_sample_accuracies)."""
    y_test = np.asarray(y_test)
    bma = np.zeros((x_test.shape[0], int(y_test.max()) + 1), dtype=np.float64)
    per_sample = []
    for model, state in samples:
        probs = predict_mean(model, state, x_test, batch_size)
        per_sample.append(float((probs.argmax(1) == y_test).mean()))
        bma += probs
    bma /= len(samples)
    return float((bma.argmax(1) == y_test).mean()), per_sample


def load_samples(save_dir, num_classes):
    """Deserialize all ``sample_*.eqx`` files into ``(model, state)`` skeletons."""
    paths = sorted(glob.glob(os.path.join(save_dir, "sample_*.eqx")))
    if not paths:
        raise FileNotFoundError(f"no sample_*.eqx files in {save_dir}")
    samples = []
    for path in paths:
        skeleton = make_resnet18(jax.random.key(0), num_classes=num_classes)
        samples.append(eqx.tree_deserialise_leaves(path, skeleton))
    return samples


def main():
    p = argparse.ArgumentParser(description="cSG-MCMC ensemble evaluation")
    p.add_argument("--data_path", required=True)
    p.add_argument("--dir", dest="save_dir", required=True)
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    args = p.parse_args()

    data = load_cifar(args.data_path, args.dataset)
    samples = load_samples(args.save_dir, data["num_classes"])
    acc, per_sample = evaluate(samples, data["x_test"], data["y_test"])
    print(f"loaded {len(samples)} samples")
    print(f"per-sample acc: mean {np.mean(per_sample):.4f}  range "
          f"[{min(per_sample):.4f}, {max(per_sample):.4f}]")
    print(f"BMA ensemble accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
