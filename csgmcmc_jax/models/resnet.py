"""ResNet18 for CIFAR in Equinox (port of ``models/resnet.py``).

Stateful ``eqx.nn.BatchNorm`` in ``mode="batch"`` (batch stats during training,
running stats at inference -- matching PyTorch) with ``momentum=0.9`` (PyTorch's
0.1). Modules operate on a single example ``(C, H, W)``; batch them with

    batched = jax.vmap(model, axis_name="batch", in_axes=(0, None), out_axes=(0, None))
    logits, state = batched(x, state)

Build params + state together::

    model, state = make_resnet18(key, num_classes=10)
"""

from typing import List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

_BN_MOMENTUM = 0.9  # == PyTorch BatchNorm momentum 0.1


def _bn(channels):
    return eqx.nn.BatchNorm(channels, axis_name="batch", momentum=_BN_MOMENTUM, mode="batch")


class BasicBlock(eqx.Module):
    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    conv2: eqx.nn.Conv2d
    bn2: eqx.nn.BatchNorm
    sc_conv: Optional[eqx.nn.Conv2d]
    sc_bn: Optional[eqx.nn.BatchNorm]

    expansion = 1

    def __init__(self, in_planes, planes, stride, key):
        k1, k2, ks = jax.random.split(key, 3)
        self.conv1 = eqx.nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, use_bias=False, key=k1)
        self.bn1 = _bn(planes)
        self.conv2 = eqx.nn.Conv2d(planes, planes, 3, stride=1, padding=1, use_bias=False, key=k2)
        self.bn2 = _bn(planes)
        if stride != 1 or in_planes != self.expansion * planes:
            self.sc_conv = eqx.nn.Conv2d(
                in_planes, self.expansion * planes, 1, stride=stride, use_bias=False, key=ks
            )
            self.sc_bn = _bn(self.expansion * planes)
        else:
            self.sc_conv = None
            self.sc_bn = None

    def __call__(self, x, state):
        out, state = self.bn1(self.conv1(x), state)
        out = jax.nn.relu(out)
        out, state = self.bn2(self.conv2(out), state)
        if self.sc_conv is not None:
            shortcut, state = self.sc_bn(self.sc_conv(x), state)
        else:
            shortcut = x
        return jax.nn.relu(out + shortcut), state


class ResNet(eqx.Module):
    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm
    blocks: List[BasicBlock]
    linear: eqx.nn.Linear

    def __init__(self, num_blocks, num_classes, key):
        keys = jax.random.split(key, 6)
        self.conv1 = eqx.nn.Conv2d(3, 64, 3, stride=1, padding=1, use_bias=False, key=keys[0])
        self.bn1 = _bn(64)
        blocks = []
        in_planes = 64
        for planes, n, stride, lk in zip(
            (64, 128, 256, 512), num_blocks, (1, 2, 2, 2), keys[1:5]
        ):
            strides = [stride] + [1] * (n - 1)
            bkeys = jax.random.split(lk, len(strides))
            for s, bk in zip(strides, bkeys):
                blocks.append(BasicBlock(in_planes, planes, s, bk))
                in_planes = planes * BasicBlock.expansion
        self.blocks = blocks
        self.linear = eqx.nn.Linear(512 * BasicBlock.expansion, num_classes, key=keys[5])

    def __call__(self, x, state):
        out, state = self.bn1(self.conv1(x), state)
        out = jax.nn.relu(out)
        for block in self.blocks:
            out, state = block(out, state)
        out = jnp.mean(out, axis=(1, 2))  # global average pool -> (C,)
        return self.linear(out), state


def ResNet18(num_classes=10, *, key):
    return ResNet([2, 2, 2, 2], num_classes, key)


def make_resnet18(key, num_classes=10):
    """Return ``(model, state)`` ready for training."""
    return eqx.nn.make_with_state(ResNet18)(num_classes=num_classes, key=key)
