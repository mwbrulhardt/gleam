from abc import ABCMeta, abstractmethod


class Estimator(ABCMeta):
    @abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError()


class ImpliedVolatilityModel(ABCMeta):
    @abstractmethod
    def get_iv(self, *args, **kwargs):
        raise NotImplementedError()


class PricingModel(ABCMeta):
    @abstractmethod
    def get_price(self, *args, **kwargs):
        raise NotImplementedError()
