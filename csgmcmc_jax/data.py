"""Manual CIFAR-10/100 loader (no torchvision / TF dependency).

Reads the official python-pickle files, normalizes per-channel (means/stds from
``experiments/config.py``), and provides on-device train augmentation matching
torchvision's ``RandomCrop(32, padding=4)`` + ``RandomHorizontalFlip()``.

Expected layout under ``root``:
    cifar10  -> root/cifar-10-batches-py/{data_batch_1..5,test_batch}
    cifar100 -> root/cifar-100-python/{train,test}

Download once (not on the hot path)::
    cifar10 : https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    cifar100: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
"""

import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

MEAN = {
    "cifar10": (0.4914, 0.4822, 0.4465),
    "cifar100": (0.5071, 0.4867, 0.4408),
}
STD = {
    "cifar10": (0.2023, 0.1994, 0.2010),
    "cifar100": (0.2675, 0.2565, 0.2761),
}
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


def _unpickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def _normalize(images_u8, dataset):
    """uint8 (N,3,32,32) -> normalized float32."""
    x = images_u8.astype(np.float32) / 255.0
    mean = np.asarray(MEAN[dataset], np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(STD[dataset], np.float32).reshape(1, 3, 1, 1)
    return (x - mean) / std


def load_cifar(root, dataset="cifar10"):
    """Return dict with normalized NCHW float32 arrays + labels (numpy)."""
    if dataset == "cifar10":
        d = os.path.join(root, "cifar-10-batches-py")
        train = [_unpickle(os.path.join(d, f"data_batch_{i}")) for i in range(1, 6)]
        x_tr = np.concatenate([b[b"data"] for b in train]).reshape(-1, 3, 32, 32)
        y_tr = np.concatenate([np.asarray(b[b"labels"]) for b in train])
        test = _unpickle(os.path.join(d, "test_batch"))
        x_te = np.asarray(test[b"data"]).reshape(-1, 3, 32, 32)
        y_te = np.asarray(test[b"labels"])
    elif dataset == "cifar100":
        d = os.path.join(root, "cifar-100-python")
        train, test = _unpickle(os.path.join(d, "train")), _unpickle(os.path.join(d, "test"))
        x_tr = np.asarray(train[b"data"]).reshape(-1, 3, 32, 32)
        y_tr = np.asarray(train[b"fine_labels"])
        x_te = np.asarray(test[b"data"]).reshape(-1, 3, 32, 32)
        y_te = np.asarray(test[b"fine_labels"])
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    return {
        "x_train": _normalize(x_tr, dataset),
        "y_train": y_tr.astype(np.int32),
        "x_test": _normalize(x_te, dataset),
        "y_test": y_te.astype(np.int32),
        "num_classes": NUM_CLASSES[dataset],
    }


def _pad_value(dataset):
    """Per-channel value of a normalized zero pixel (torchvision zero-pads in raw
    [0,1] space *before* normalization)."""
    mean = np.asarray(MEAN[dataset], np.float32)
    std = np.asarray(STD[dataset], np.float32)
    return jnp.asarray((-mean / std).reshape(3, 1, 1))


def make_augment(dataset, pad=4):
    """Return ``augment(key, images)`` -> augmented batch (on device, jittable).

    Per image: zero-pad (in raw space) by ``pad`` then random 32x32 crop, and a
    random horizontal flip with p=0.5.
    """
    pad_value = _pad_value(dataset)

    def augment_one(key, img):
        kc, kf = jax.random.split(key)
        c, h, w = img.shape
        # zero-pad in normalized space using the correct per-channel constant
        # (a raw zero pixel maps to -mean/std after normalization).
        padded = jnp.broadcast_to(pad_value, (c, h + 2 * pad, w + 2 * pad))
        padded = padded.at[:, pad:pad + h, pad:pad + w].set(img)
        top = jax.random.randint(kc, (), 0, 2 * pad + 1)
        left = jax.random.randint(kc, (), 0, 2 * pad + 1)
        cropped = jax.lax.dynamic_slice(padded, (0, top, left), (c, h, w))
        flip = jax.random.bernoulli(kf)
        return jnp.where(flip, cropped[:, :, ::-1], cropped)

    def augment(key, images):
        keys = jax.random.split(key, images.shape[0])
        return jax.vmap(augment_one)(keys, images)

    return augment
