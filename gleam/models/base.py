from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

import gleam.black_scholes as bs
from gleam import framework as fw


class Estimator(ABC):
    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError()


class ImpliedVolatilityModel(ABC):
    @abstractmethod
    def get_ivs(self, *args, **kwargs):
        raise NotImplementedError()


class PricingModel(ABC):
    @abstractmethod
    def get_prices(self, *args, **kwargs):
        raise NotImplementedError()


@dataclass
class Slice:
    """Option price slice."""

    prices: List[float,]
    strikes: List[float,]
    ttm: float
    spot: float
    forwards: Optional[float] = (None,)
    rates: Optional[float] = (None,)
    w: int = (1,)


class OptionMarketData:
    """
    Class representing option market price lattice.

    Attributes:
        prices (List[List[float]]): List of lists containing option prices.
        strikes (List[List[float]]): List of lists. Each inner list contains monotonically increasing strikes corresponding to a maturity.
        ttm (List[float]): Time to maturity (yearfrac).
        spot (float): Spot price.
        forwards (Optional[List[float]]): List of forward prices. Default is None.
        rates (Optional[List[float]]): List of interest rates. Default is None.
        w (int): Parameter w. Default is 1.
        framework (Optional[str]): Framework used for calculations. Default is 'numpy'.
    """

    def __init__(
        self,
        prices: List[List[float,]],
        strikes: List[List[float,]],
        ttm: List[float],
        spot: float,
        forwards: Optional[List[float]] = None,
        rates: Optional[List[float]] = None,
        w: int = 1,
        framework: Optional[str] = "numpy",
    ):
        """
        Initialize OptionMarketData with provided parameters.

        Args:
            prices (List[List[float]]): List of lists containing option prices.
            strikes (List[List[float]]): List of lists.
                Each inner list contains monotonically increasing strikes corresponding to a maturity.
            ttm (List[float]): Time to maturity (yearfrac).
            spot (float): Spot price.
            forwards (Optional[List[float]]): List of forward prices. Default is None.
            rates (Optional[List[float]]): List of interest rates. Default is None.
            w (int): Parameter w. Default is 1.
            framework (Optional[str]): Framework used for calculations. Default is 'numpy'.

        Raises:
            AssertionError: If input data does not meet requirements.
        """
        # Check formatting.
        assert all(
            [t1 > t0 for t0, t1 in zip(ttm[:-1], ttm[1:])]
        ), "Time to maturity has to be monotonically increasing."
        assert all(
            [
                all(
                    [k1 > k0 for k0, k1 in zip(strikes[i][:-1], strikes[i][1:])]
                    for i in range(len(strikes))
                )
            ]
        ), "Each slice of strikes has to be monotonically increasing."

        for price_slice, strike_slice in zip(prices, strikes):
            assert len(price_slice) == len(
                strike_slice
            ), "Length of price slice and strike slice must coincide."

        assert len(prices) == len(strikes) == len(ttm)
        if rates is not None:
            assert len(ttm) == len(rates)
        if forwards is not None:
            assert len(ttm) == len(forwards)

        self._prices = prices
        self._spot = spot
        self._strikes = strikes
        self._ttm = ttm
        self._forwards = forwards

        self._w = w

        self._framework = framework

        if rates is None:
            self._rates = len(strikes) * [
                0.0,
            ]
        else:
            self._rates = rates

    @property
    def prices(
        self,
    ):
        """List of option prices."""
        return [self.to(price) for price in self._prices]

    @property
    def strikes(self):
        """List of strikes."""
        return [self.to(strike_slice) for strike_slice in self._strikes]

    @property
    def xstrikes(self):
        """List of strikes adjusted for discount factors."""
        return [D * K for D, K in zip(self.discount_factors, self.strikes)]

    @property
    def xprices(self):
        """List of option prices adjusted for forward prices."""
        return [V / F for V, F in zip(self.prices, self.forwards)]

    @property
    def rates(self):
        """List of interest rates."""

        return [
            fw.repeat(self.to(r), repeats=len(K), axis=0)
            for r, K in zip(self._rates, self.strikes)
        ]

    @property
    def discount_factors(self):
        """List of discount factors."""

        return [fw.exp(-1.0 * r * T) for r, T in zip(self._rates, self.ttm)]

    @property
    def forwards(self):
        """List of forward prices."""
        if self._forwards is None:
            return [self._spot / D for D in self.discount_factors]
        return self._forwards

    @property
    def ttm(self):
        """List of time to maturity (annualized)."""
        return [
            fw.repeat(self.to(T), repeats=len(K), axis=0)
            for T, K in zip(self._ttm, self.strikes)
        ]

    @property
    def implied_vols(self) -> List[np.array]:
        """List of implied volatilities."""
        return [
            bs.iv(V=V, S=self._spot, K=K, tau=tau, r=r, w=self._w)
            for (V, K, tau, r) in zip(
                self.prices,
                self.strikes,
                self.ttm,
                self.rates,
            )
        ]

    @property
    def delta_plus(
        self,
    ):
        dp_list = list()
        for p, k in zip(self.prices, self.strikes):
            if len(p) >= 3:
                dp = (p[2:] - p[1:-1]) / (k[2:] - k[1:-1])
                dp_list.append(dp)
            else:
                dp_list.append(None)
        return dp_list

    @property
    def delta_minus(
        self,
    ):
        dm_list = list()
        for p, k in zip(self.prices, self.strikes):
            if len(p) >= 3:
                dm = (p[1:-1] - p[:-2]) / (k[1:-1] - k[:-2])
                dm_list.append(dm)
            else:
                dm_list.append(None)
        return dm_list

    @property
    def sofd(
        self,
    ) -> List[np.array]:
        """List of second order finite differences."""
        butterflies = list()
        for dm, dp in zip(self.delta_plus, self.delta_minus):
            if dm is None:
                bfs = None
            else:
                bfs = dp - dm
            butterflies.append(bfs)
        return butterflies

    def to(self, x: fw.TensorType):
        """
        Convert input to the specified framework.

        Args:
            x (fw.TensorType): Input to convert.

        Returns:
            fw.TensorType: Converted input.
        """
        if self._framework == "torch":
            return fw.to_torch(x)
        elif self._framework == "numpy":
            return np.array(x)
        elif self._framework == "list":
            return x
        else:
            raise NotImplementedError("Framework %s not implemented" % self._framework)

    def torch(self):
        """Change framework to torch."""
        self._framework = "torch"

    def numpy(self):
        """Change framework to numpy."""
        self._framework = "numpy"
