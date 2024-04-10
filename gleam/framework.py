from typing import Tuple, Union

import numpy as np
from scipy.stats import norm

try:
    import torch
except ImportError:
    pass


TensorType = Union[np.ndarray, "torch.Tensor"]
TensorTypeOrScalar = Union[TensorType, float]


def assert_same_type(*xs: TensorTypeOrScalar):
    T = type(xs[0])
    assert all(isinstance(x, T) for x in xs), "All inputs are not of same type."


def log(x: TensorTypeOrScalar) -> TensorType:
    if isinstance(x, (int, float, np.ndarray)):
        return np.log(x)
    return x.log()


def exp(x: TensorTypeOrScalar) -> TensorType:
    if isinstance(x, (int, float, np.ndarray)):
        return np.exp(x)
    return x.exp()


def sqrt(x: TensorTypeOrScalar) -> TensorType:
    if isinstance(x, (int, float, np.ndarray)):
        return np.sqrt(x)
    return x.sqrt()


def pow(x: TensorTypeOrScalar, d: float) -> TensorType:
    if isinstance(x, (int, float, np.ndarray)):
        return np.power(x, d)
    return x.pow(d)


def repeat(x: TensorType, repeats: int, axis: int):
    if isinstance(x, np.ndarray):
        return np.repeat(x, repeats=repeats, axis=axis)
    repeat_shape = [1 for _ in range(x.shape[0])]
    repeat_shape[axis] = repeats
    return x.repeat(tuple(repeat_shape))


def ones_like(x: TensorType) -> TensorType:
    if isinstance(x, np.ndarray):
        return np.ones_like(x)
    return torch.ones_like(x)


def concat(xs: TensorType, dim: int = 0):
    assert_same_type(xs)
    x = xs[0]

    if isinstance(x, np.ndarray):
        return np.concatenate(xs, axis=dim)
    return torch.concat(xs, dim=dim)


def where(
    condition: TensorType, input: TensorTypeOrScalar, other: TensorTypeOrScalar
) -> TensorType:
    if isinstance(condition, (int, float, np.ndarray)):
        return np.where(condition, input, other)

    return torch.where(condition, input, other)


def _to_numpy(x: TensorTypeOrScalar) -> np.ndarray:
    if isinstance(x, (int, float)):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    return x.numpy()


def to_numpy(*xs: TensorTypeOrScalar) -> Tuple[np.ndarray, ...]:
    if len(xs) > 1:
        return [_to_numpy(x) for x in xs]
    else:
        return _to_numpy(xs[0])


def to_torch(x: TensorTypeOrScalar) -> Tuple["torch.Tensor"]:
    assert_same_type(x)
    return torch.Tensor(x)


class linalg:
    def inv(x: TensorType):
        if isinstance(x, np.ndarray):
            return np.linalg.inv(x)
        return torch.linalg.inv(x)


class dist:
    class normal:
        def pdf(x: TensorTypeOrScalar, loc=0, scale=1):
            if isinstance(x, (int, float, np.ndarray)):
                return norm.pdf(x, loc, scale)
            dist = torch.distributions.Normal(loc, scale)
            return dist.log_prob(x).exp()

        def cdf(x, loc=0, scale=1):
            if isinstance(x, (int, float, np.ndarray)):
                return norm.cdf(x, loc, scale)
            dist = torch.distributions.Normal(loc, scale)
            return dist.cdf(x)

        def icdf(x, loc=0, scale=1):
            if isinstance(x, (int, float, np.ndarray)):
                return norm.ppf(x, loc, scale)
            dist = torch.distributions.Normal(loc, scale)
            return dist.icdf(x)
