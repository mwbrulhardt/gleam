import numpy as np

import gleam.black_scholes as bs
import gleam.framework as fw
from gleam.models.base import (
    Estimator,
    ImpliedVolatilityModel,
    OptionMarketData,
    PricingModel,
)


class QuadraticLog(Estimator, PricingModel, ImpliedVolatilityModel):
    name = "quad-log"
    feature_names = ["intercept", "M", "M_2", "tau", "M_tau"]

    def __init__(self, alpha: float = -0.2):
        self.alpha = alpha
        self.coef = 5 * [
            None,
        ]

    @property
    def parameters(self):
        return {
            "intercept": self.coef[0],
            "M": self.coef[1],
            "M_2": self.coef[2],
            "tau": self.coef[3],
            "M_tau": self.coef[4],
        }

    def make_features(self, F: fw.TensorType, K: fw.TensorType, tau: fw.TensorType):
        F = F.reshape(-1, 1)
        K = K.reshape(-1, 1)
        tau = tau.reshape(-1, 1)

        M = fw.log(F / K) / tau**0.5

        features = [fw.ones_like(M), M, M**2, tau, M * tau]
        X = fw.concat(features, dim=1)
        return X

    def fit(self, market_data: OptionMarketData):
        X = self.make_features(
            np.concatenate(market_data.forwards),
            np.concatenate(market_data.strikes),
            np.concatenate(market_data.ttm),
        )
        iv = np.concatenate(market_data.implied_vols)
        self.coef = fw.linalg.inv(X.T @ X) @ X.T @ iv

    def get_ivs(self, forwards: np.array, strikes: np.array, tau: np.array):
        X = self.make_features(forwards, strikes, tau)
        return X @ self.coef

    def get_prices(self, forwards: np.array, strikes: np.array, tau: np.array, w: int):
        iv = self.get_ivs(forwards, strikes, tau)
        prices = bs.price(fw.ones_like(iv), strikes / forwards, tau, iv, r=0, q=0, w=w)
        return forwards * prices

    @classmethod
    def from_params(cls, params: dict):
        model = cls()
        model.coef = np.array([params[name] for name in model.feature_names])
        return model
