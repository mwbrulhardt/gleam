import numpy as np
from scipy.stats import norm


class Framework:
    """A class for encapsulating both the numpy and torch framework into one common api."""

    _instances = {}

    def __init__(self, backend: str = "numpy"):
        self._backend = backend
        if backend == "numpy":
            self._framework = np
        elif backend == "torch":
            import torch

            self._framework = torch

    @classmethod
    def get_framework(cls, backend: str) -> "Framework":
        if backend not in cls._instances:
            cls._instances[backend] = Framework(backend)
        return cls._instances[backend]

    def log(self, x):
        return self._framework.log(x)

    def exp(self, x):
        return self._framework.exp(x)

    def norm_cdf(self, x, loc=0, scale=1):
        if self._backend == "torch":
            dist = self._framework.distributions.Normal(loc, scale)
            return dist.cdf(x)
        return norm.cdf(x, loc, scale)

    def norm_icdf(self, x, loc=0, scale=1):
        if self._backend == "torch":
            dist = self._framework.distributions.Normal(loc, scale)
            return dist.icdf(x)
        return norm.ppf(x, loc, scale)

    def norm_pdf(self, x, loc=0, scale=1):
        if self._backend == "torch":
            dist = self._framework.distributions.Normal(loc, scale)
            return dist.pdf(x)
        return norm.pdf(x, loc, scale)


def resolve(x) -> Framework:
    if "torch.Tensor" in str(type(x)):
        return Framework.get_framework("torch")
    return Framework.get_framework("numpy")
