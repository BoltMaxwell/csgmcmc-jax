"""Download + extract the official CIFAR-10/100 python pickles.

Idempotent: skips download/extract if the target files already exist.

    python -m csgmcmc_jax.download --dataset cifar10 --root ./data
    python -m csgmcmc_jax.download --dataset all --root ./data
"""

import argparse
import hashlib
import os
import tarfile
import urllib.request

_SPECS = {
    "cifar10": {
        "url": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "md5": "c58f30108f718f92721af3b95e74349a",
        "marker": "cifar-10-batches-py/test_batch",
    },
    "cifar100": {
        "url": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
        "md5": "eb9058c3a382ffc7106e4002c42a8d85",
        "marker": "cifar-100-python/test",
    },
}


def _md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(dataset, root):
    """Download + extract ``dataset`` under ``root``; return the extracted dir."""
    spec = _SPECS[dataset]
    os.makedirs(root, exist_ok=True)
    marker = os.path.join(root, spec["marker"])
    if os.path.exists(marker):
        print(f"[{dataset}] already present at {os.path.dirname(marker)}")
        return os.path.dirname(marker)

    tar_path = os.path.join(root, os.path.basename(spec["url"]))
    if not (os.path.exists(tar_path) and _md5(tar_path) == spec["md5"]):
        print(f"[{dataset}] downloading {spec['url']} ...")
        urllib.request.urlretrieve(spec["url"], tar_path)
        got = _md5(tar_path)
        if got != spec["md5"]:
            raise RuntimeError(f"md5 mismatch for {tar_path}: got {got}, want {spec['md5']}")
        print(f"[{dataset}] md5 OK ({got})")

    print(f"[{dataset}] extracting ...")
    with tarfile.open(tar_path) as tf:
        tf.extractall(root)  # archive is the trusted CIFAR distribution
    os.remove(tar_path)
    print(f"[{dataset}] ready at {os.path.dirname(marker)}")
    return os.path.dirname(marker)


def main():
    p = argparse.ArgumentParser(description="Download CIFAR-10/100 python pickles")
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "all"])
    p.add_argument("--root", default="./data")
    args = p.parse_args()
    names = ["cifar10", "cifar100"] if args.dataset == "all" else [args.dataset]
    for name in names:
        download(name, args.root)


if __name__ == "__main__":
    main()
