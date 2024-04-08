from typing import TYPE_CHECKING, Tuple, Union, overload

import numpy as np
from scipy.stats import norm

try:
    import torch
except ImportError:
    pass

if TYPE_CHECKING:
    import torch

TensorType = Union[np.ndarray, "torch.Tensor"]
TensorTypeOrScalar = TensorType | float


def assert_same_type(x: TensorTypeOrScalar, *xs: TensorTypeOrScalar):
    T = type(x)
    if len(xs) < 0:
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


@overload
def concat(*xs: "torch.Tensor") -> "torch.Tensor":
    ...


@overload
def concat(*xs: np.ndarray) -> np.ndarray:
    ...


def concat(*xs, dim: int = 0) -> TensorType:
    if all(isinstance(x, np.ndarray) for x in xs):
        return np.concatenate(xs, axis=dim)
    return torch.concat(xs, dim=dim)


def where(
    condition: TensorType | bool, input: TensorTypeOrScalar, other: TensorTypeOrScalar
) -> TensorTypeOrScalar:
    if (
        isinstance(condition, bool)
        and isinstance(input, (float, int))
        and isinstance(other, (float, int))
    ):
        return input if condition else other
    elif (
        isinstance(condition, np.ndarray)
        and isinstance(input, (float, int, np.ndarray))
        and isinstance(other, (float, int, np.ndarray))
    ):
        return np.where(condition, input, other)
    elif (
        not isinstance(condition, (np.ndarray, bool))
        and not isinstance(input, np.ndarray)
        and not isinstance(other, np.ndarray)
    ):
        return torch.where(condition, input, other)
    raise Exception("Mismatch in types.")


def _to_numpy(x: TensorTypeOrScalar) -> np.ndarray:
    if isinstance(x, (int, float)):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    return x.numpy()


def to_numpy(*xs: TensorTypeOrScalar) -> Tuple[np.ndarray, ...]:
    return tuple(_to_numpy(x) for x in xs)


def _to_torch(x: TensorTypeOrScalar) -> "torch.Tensor":
    if isinstance(x, (int, float)):
        return torch.tensor(x)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x


def to_torch(*xs: TensorTypeOrScalar) -> Tuple["torch.Tensor", ...]:
    return tuple(_to_torch(x) for x in xs)


class linalg:
    @staticmethod
    def inv(x: TensorType) -> TensorType:
        if isinstance(x, np.ndarray):
            return np.linalg.inv(x)
        return torch.linalg.inv(x)


class dist:
    class normal:
        @staticmethod
        def pdf(
            x: TensorTypeOrScalar,
            loc: TensorTypeOrScalar = 0,
            scale: TensorTypeOrScalar = 1,
        ) -> TensorTypeOrScalar:
            if isinstance(x, (int, float, np.ndarray)):
                return norm.pdf(x, loc, scale)
            dist = torch.distributions.Normal(loc, scale)
            return dist.log_prob(x).exp()

        @staticmethod
        def cdf(
            x: TensorTypeOrScalar,
            loc: TensorTypeOrScalar = 0,
            scale: TensorTypeOrScalar = 1,
        ) -> TensorTypeOrScalar:
            if isinstance(x, (int, float, np.ndarray)):
                return norm.cdf(x, loc, scale)
            dist = torch.distributions.Normal(loc, scale)
            return dist.cdf(x)

        @staticmethod
        def icdf(
            x: TensorTypeOrScalar,
            loc: TensorTypeOrScalar = 0,
            scale: TensorTypeOrScalar = 1,
        ) -> TensorTypeOrScalar:
            if isinstance(x, (int, float, np.ndarray)):
                return norm.ppf(x, loc, scale)
            dist = torch.distributions.Normal(loc, scale)
            return dist.icdf(x)
