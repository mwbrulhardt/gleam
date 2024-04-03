from typing import Tuple, Union, overload

import numpy as np
from scipy.stats import norm

try:
    import torch
except ImportError:
    pass


TensorType = Union[np.ndarray, "torch.Tensor"]
TensorTypeOrScalar = TensorType | float


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


def ones_like(x: TensorType) -> TensorType:
    if isinstance(x, np.ndarray):
        return np.ones_like(x)
    return torch.ones_like(x)


def concat(*xs: TensorType, dim: int = 0):
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


@overload
def to_numpy(x: TensorTypeOrScalar) -> np.ndarray:
    if isinstance(x, (int, float)):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    return x.numpy()


def to_numpy(*xs: TensorTypeOrScalar) -> Tuple[np.ndarray]:
    assert_same_type(xs)
    x = xs[0]
    if isinstance(x, (int, float)):
        return (np.array(x) for x in xs)
    elif isinstance(x, np.ndarray):
        return xs
    return (x.numpy() for x in xs)


def to_torch(*xs: TensorTypeOrScalar) -> Tuple["torch.Tensor"]:
    assert_same_type(xs)
    x = xs[0]
    if isinstance(x, (int, float)):
        return (torch.tensor(x) for x in xs)
    elif isinstance(x, np.ndarray):
        return (torch.from_numpy(x) for x in xs)
    return xs


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
            return dist.pdf(x, loc, scale)

        def cdf(x, loc=0, scale=1):
            if isinstance(x, (int, float, np.ndarray)):
                return norm.cdf(x, loc, scale)
            dist = torch.distributions.Normal(loc, scale)
            return dist.cdf(x, loc, scale)

        def icdf(x, loc=0, scale=1):
            if isinstance(x, (int, float, np.ndarray)):
                return norm.ppf(x, loc, scale)
            dist = torch.distributions.Normal(loc, scale)
            return dist.icdf(x, loc, scale)
