import functools as ft
from math import ceil, sqrt
from typing import NamedTuple, Union

import jax
import jax.numpy as jnp
import jax.random as jrd
import jax.tree_util as jtu


def long_width_axis_fn(w):
    if w.ndim > 1:
        return -1 if w.shape[-1] >= w.shape[-2] else -2
    else:
        return None


def sqrt_width_rank_fn(w, axis=-1):
    if w.ndim > 1:
        d = min(1, ceil(sqrt(w.shape[axis])))
        return d
    else:
        return 1


def width_ratio_rank_fn(w, scale=0.0, minimum=8, maximum=8, axis=-1):
    """Returns the rank closest to the ratio of the specified axis to the
    specified scale within the specified range if the input is
    multi-dimensional, but always returns the unit dimensionality of 1 for
    scalars."""
    if w.ndim > 1:
        d = min(max(minimum, w.shape[axis], int(w.shape[axis] * scale)), maximum)
        return d
    else:
        return 1


class Decomp(NamedTuple):
    "https://openreview.net/forum?id=d71n4ftoCBy"
    u: jax.Array
    v: jax.Array
    x: jax.Array
    y: jax.Array
    i: int


class Offset(NamedTuple):
    b: jax.Array


Adapter = Union[Decomp, Offset]


def init(rng, params, rank_fn=sqrt_width_rank_fn, axis_fn=long_width_axis_fn):
    """Initialize low-rank adapters. Ignore tree leaves w that are not instances
    of jax.Array or where rank_fn(w) is None."""
    dimen = jtu.tree_map(axis_fn, params, is_leaf=lambda a: isinstance(a, jax.Array))
    ranks = jtu.tree_map(
        lambda a, i: rank_fn(a, axis=i),
        params,
        dimen,
        is_leaf=lambda a: isinstance(a, jax.Array),
    )

    def make_adapter(w, r, i):
        nonlocal rng
        if r is None:
            return None
        elif w.ndim > 1:
            rng, key = jrd.split(rng)
            w = w.swapaxes(-1, i)
            m, n = w.shape[-2:]
            k = min(r, n)
            u, x = jrd.normal(key, [2, m, k], dtype=w.dtype) / sqrt(m)
            v, y = jnp.zeros([2, k, n], dtype=w.dtype)
            return Decomp(u, v, x, y, i)
        elif w.ndim == 1:
            return Offset(jnp.zeros_like(w))

    return jtu.tree_map(
        make_adapter, params, ranks, dimen, is_leaf=lambda a: isinstance(a, jax.Array)
    )


def fuse(params, lora, precision=jax.lax.Precision.HIGHEST):
    "Fuse low-rank adapters into full-rank parameter trees."

    def adapt(w, a):
        if a is None:
            return w
        elif isinstance(a, Decomp):
            dot = ft.partial(
                jax.lax.dot,
                precision=precision,
                preferred_element_type=w.dtype,
            )
            return w + (dot(a.u, a.v) * dot(a.x, a.y)).swapaxes(-1, a.i)
        elif isinstance(a, Offset):
            return w + a.b
        else:
            return w

    return jtu.tree_map(adapt, params, lora)
