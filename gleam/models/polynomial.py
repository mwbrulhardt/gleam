import numpy as np

import gleam.black_scholes as bs
import gleam.framework as fw
from gleam.models.base import Estimator, ImpliedVolatilityModel, PricingModel


class QuadraticLog(Estimator, PricingModel, ImpliedVolatilityModel):
    name = "quad-log"
    feature_names = ["intercept", "M", "M_2", "tau", "M_tau"]

    def __init__(self, alpha: float = -0.2):
        self.alpha = alpha
        self.coef = None

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

    def fit(self, F, K, tau, iv):
        X = self.make_features(F, K, tau)
        self.coef = fw.linalg.inv(X.T @ X) @ X.T @ iv

    def get_iv(self, F, K, tau):
        X = self.make_features(F, K, tau)
        return X @ self.coef

    def get_price(self, F, K, tau, w):
        iv = self.get_ivol(F, K, tau)
        return F * bs.price(fw.ones_like(iv), K / F, tau, iv, r=0, q=0, w=w)

    @classmethod
    def from_params(cls, params: dict):
        model = cls()
        model.coef = np.array([params[name] for name in model.feature_names])
        return model
