import numpy as np
import pytest

from gleam.models.base import OptionMarketData
from gleam.models.polynomial import QuadraticLog


@pytest.fixture
def market_data():
    prices = [[0.11, 0.01, 0.001], [0.12, 0.015, 0.002]]
    strikes = [[0.9, 1.0, 1.1], [0.9, 1.0, 1.1]]
    ttm = [0.1, 0.5]
    spot = 0.9
    w = 1
    framework = 'numpy'

    return OptionMarketData(
        prices=prices,
        strikes=strikes,
        ttm=ttm,
        spot=spot,
        w=w,
        framework=framework
    )


def test_initialization():
    model = QuadraticLog(alpha=-0.2)
    assert model.alpha == -0.2
    assert model.coef == (5 * [None, ])


def test_parameters():
    model = QuadraticLog(alpha=-0.2)
    assert model.parameters == {
        "intercept": None,
        "M": None,
        "M_2": None,
        "tau": None,
        "M_tau": None
    }


def test_make_features():
    model = QuadraticLog(alpha=-0.2)
    F = np.array([100, 100, 100])
    K = np.array([90, 100, 110])
    tau = np.array([0.1, 0.2, 0.1])
    features = model.make_features(F, K, tau)
    assert features.shape == (3, 5)


def test_fit(market_data):
    model = QuadraticLog(alpha=-0.2)
    model.fit(market_data)
    assert model.coef is not None


def test_get_iv(market_data):
    model = QuadraticLog(alpha=-0.2)
    model.fit(market_data)
    forwards = np.concatenate(market_data.forwards)
    strikes = np.concatenate(market_data.strikes)
    tau = np.concatenate(market_data.ttm)
    iv = model.get_ivs(forwards, strikes, tau)
    assert iv.shape == (sum(len(x) for x in market_data.strikes),)


def test_get_price(market_data):
    model = QuadraticLog(alpha=-0.2)
    model.fit(market_data)
    forwards = np.concatenate(market_data.forwards)
    strikes = np.concatenate(market_data.strikes)
    tau = np.concatenate(market_data.ttm)
    prices = model.get_prices(forwards, strikes, tau, w=1)
    assert prices.shape == (sum(len(x) for x in market_data.strikes),)
