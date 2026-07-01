"""End-to-end smoke test of the CIFAR train -> ensemble pipeline on tiny synthetic
data (no real CIFAR download needed). Slow on CPU (~1-2 min: ResNet18 compiles).

    JAX_PLATFORMS=cpu PYTHONPATH=. python tests/test_cifar_smoke.py
"""

import numpy as np

from csgmcmc_jax.ensemble import evaluate
from csgmcmc_jax.train_cifar import train


def _fake_data(seed=0, n_train=64, n_test=32, num_classes=10):
    rng = np.random.default_rng(seed)
    return {
        "x_train": rng.standard_normal((n_train, 3, 32, 32)).astype(np.float32),
        "y_train": rng.integers(0, num_classes, n_train).astype(np.int32),
        "x_test": rng.standard_normal((n_test, 3, 32, 32)).astype(np.float32),
        "y_test": rng.integers(0, num_classes, n_test).astype(np.int32),
        "num_classes": num_classes,
    }


def test_train_then_ensemble_runs():
    data = _fake_data()
    samples = train(data, method="csgld", backend="jax", epochs=2, num_cycles=1,
                    batch_size=32, samples_per_cycle=2, seed=0, log_every=99)
    assert len(samples) == 2
    acc, per_sample = evaluate(samples, data["x_test"], data["y_test"])
    assert 0.0 <= acc <= 1.0
    assert len(per_sample) == 2
    print(f"[ok] pipeline ran; BMA acc={acc:.3f} (chance ~0.1 on random data)")


def test_serialised_samples_roundtrip(tmp_path=None):
    import tempfile

    from csgmcmc_jax.ensemble import load_samples

    data = _fake_data(seed=1)
    d = tempfile.mkdtemp() if tmp_path is None else str(tmp_path)
    train(data, method="csghmc", backend="blackjax", epochs=1, num_cycles=1,
          batch_size=32, samples_per_cycle=1, seed=1, save_dir=d, log_every=99)
    samples = load_samples(d, num_classes=10)
    assert len(samples) == 1
    acc, _ = evaluate(samples, data["x_test"], data["y_test"])
    assert 0.0 <= acc <= 1.0
    print(f"[ok] serialise->load->evaluate roundtrip; acc={acc:.3f}")


if __name__ == "__main__":
    test_train_then_ensemble_runs()
    test_serialised_samples_roundtrip()
    print("\nCIFAR smoke tests passed")
