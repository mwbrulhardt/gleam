"""
Realised Model DLV (Discrete Local Volatility)

This module implements the Realised Model DLV, a pricing and implied volatility model
that captures the realized volatility dynamics of an underlying asset. The model combines
historical time series data with option market data to estimate a volatility surface and
price options.

The main components of the Realised Model DLV are:

1. Time Series Processing:
   - The model processes a time series DataFrame representing the historical prices of the
     underlying asset.
   - It creates lagged windows of the time series based on specified time deltas to capture
     the realized volatility dynamics over different time horizons.

2. Distribution Fitting:
   - For each time delta, the model fits a distribution to the lagged returns using the
     Gaussianizing transformation and the Lambert W distribution.
   - The fitted distribution is used to compute option prices for a given set of strike prices.

3. Option Market Data:
   - The model takes option market data as input, including option prices, strike prices, and
     time to maturity.
   - It organizes the market data into a structured format using the OptionMarketData class.

4. Discrete Local Volatility Model:
   - The model uses the Discrete Local Volatility (DLV) approach to estimate a volatility surface.
   - It fits the DLV model to the option market data and the fitted distribution from the time series.
   - The DLV model estimates the local volatility dynamics based on the realized volatility and
     market prices.

5. Pricing and Implied Volatility:
   - Once the DLV model is fitted, it can be used for pricing options and computing implied volatilities.
   - The model provides methods to compute option prices and implied volatilities for given strike
     prices and time to maturity values.

The Realised Model DLV combines historical volatility information from the time series with current
market data to provide a comprehensive view of the volatility dynamics. It allows for the pricing and
implied volatility calculation of options across different strike prices and maturities.

The model is flexible and can be fitted to different time series and market data, making it adaptable
to various underlying assets and market conditions. It provides a framework for incorporating realized
volatility into option pricing and risk management.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torchlambertw
from pylambertw.preprocessing import gaussianizing

from gleam.models.base import OptionMarketData, Estimator, \
    ImpliedVolatilityModel, PricingModel
from gleam.models.base import Slice
from gleam.models.dlv import DiscreteLocalVolatilityModel
import matplotlib.pyplot as plt


def get_windows(df: pd.DataFrame, delta: pd.Timedelta, num_lags: int):
    """
    Constructs the admissible time windows to create the lagged time series.

    Args:
        df (pd.DataFrame): Input time series DataFrame.
        delta (pd.Timedelta): Time delta for creating lagged windows.
        num_lags (int): Number of lags to create.

    Returns:
        np.ndarray: Array of time windows.
    """
    ts = df.index.values.reshape(-1, 1)
    windows = np.concatenate([ts + i * delta for i in range(num_lags)], axis=1)
    windows = windows[windows[:, -1] <= ts[-1]]
    windows = windows[np.isin(windows, ts).all(1)]
    return windows


def windowed(df: pd.DataFrame, delta: pd.Timedelta, num_lags: int):
    """
    Creates a windowed representation of the input time series.

    Args:
        df (pd.DataFrame): Input time series DataFrame.
        delta (pd.Timedelta): Time delta for creating lagged windows.
        num_lags (int): Number of lags to create.

    Returns:
        np.ndarray: Windowed representation of the time series.
    """
    windows = get_windows(df, delta, num_lags)
    return np.concatenate(
        [df.loc[windows[:, i]].values[..., None] for i in
            range(num_lags)
        ],
        axis=1
    )


def fit_distribution(
    df: pd.DataFrame,
    delta: pd.Timedelta,
    strikes: List[float],
    n_iters: int = 500000
):
    """
    Fits a distribution to the input time series and computes option prices.

    Args:
        df (pd.DataFrame): Input time series DataFrame.
        delta (pd.Timedelta): Time delta for creating lagged windows.
        strikes (List[float]): List of strike prices.
        n_iters (int): Number of iterations for computing option prices.

    Returns:
        Slice: Slice object containing prices, strikes, and time to maturity.
    """
    data = windowed(df, delta, num_lags=2)
    rtn = data[:, 1:2] / data[:, 0:1]
    x = pd.DataFrame(data=rtn)
    clf = gaussianizing.Gaussianizer(lambertw_type="h", method="igmm")
    clf.fit(x)

    dist = torchlambertw.distributions.TailLambertWNormal(
        loc=1.0,
        scale=clf.estimators[0].tau.scale,
        tailweight=clf.estimators[0].tau.lambertw_params.delta,
    )

    x = dist.icdf(torch.linspace(0, 1, n_iters + 2))[1:-1]
    pdf = dist.log_prob(x).double().exp()
    strikes = torch.Tensor(strikes)

    b = pdf.unsqueeze(1) * \
        (x.unsqueeze(1) - strikes.unsqueeze(0).double()).relu()
    a = 0.5 * (b[1:] + b[:-1]) * (x[1:] - x[:-1]).unsqueeze(-1)

    prices = a.sum(0)

    ttm = delta.total_seconds() / (365 * 24 * 60 * 60)

    return Slice(prices.tolist(), strikes.tolist(), ttm, spot=1.0)


class RealisedModelDLV(Estimator, PricingModel, ImpliedVolatilityModel):
    """
    Realised Model using Discrete Local Volatility (DLV) for pricing and implied volatility.
    """

    def __init__(self):
        self.data_prices = list()
        self.data_strikes = list()
        self.data_ttm = list()
        self.dlv = DiscreteLocalVolatilityModel()

    def fit(
        self,
        time_series: pd.DataFrame,
        deltas: List[pd.Timedelta],
        strikes: List[List[float]],
        bounds: List[Tuple[float, float]]
    ):
        """
        Fits the Realised Model DLV to the input time series and market data.

        Args:
            time_series (pd.DataFrame): Input time series DataFrame.
            deltas (List[pd.Timedelta]): List of time deltas for creating lagged windows.
            strikes (List[List[float]]): List of strike prices for each time delta.
            bounds (List[Tuple[float, float]]): Bounds for the DLV model.
        """
        prices = list()
        ttm = list()
        self.data_strikes = strikes

        for delta, strike_slice in zip(deltas, strikes):
            slice = fit_distribution(
                time_series, delta, strike_slice
            )
            self.data_prices.append(slice.prices)
            self.data_ttm.append(slice.ttm)

            # compute call prices

        market_data = OptionMarketData(
            prices=prices,
            strikes=strikes,
            ttm=ttm,
            spot=1.0,
        )

        self.dlv.fit(
            option_market_data=market_data,
            bounds=bounds
        )

    def get_prices(
        self, k: List[np.array], tau: List[float]
    ) -> List[np.array]:
        """
        Computes option prices using the fitted DLV model.

        Args:
            k (List[np.array]): List of strike prices.
            tau (List[float]): List of time to maturity values.

        Returns:
            List[np.array]: List of computed option prices.
        """
        return self.dlv.get_prices(k, tau)

    def get_ivs(
        self, k: List[np.array], tau: List[float]
    ) -> List[np.array]:
        """
        Computes implied volatilities using the fitted DLV model.

        Args:
            k (List[np.array]): List of strike prices.
            tau (List[float]): List of time to maturity values.

        Returns:
            List[np.array]: List of computed implied volatilities.
        """
        return self.dlv.get_ivs(k, tau)


