import numpy as np
import pytest

import gleam.black_scholes as bs
from gleam.models.base import OptionMarketData
from gleam.models.dlv import DiscreteLocalVolatilityModel


@pytest.fixture
def option_market_data():
    prices = [[0.11, 0.01, 0.001], [0.12, 0.015, 0.002]]
    strikes = [[0.9, 1.0, 1.1], [0.9, 1.0, 1.1]]
    ttm = [0.1, 0.5]
    spot = 0.9
    w = 1
    framework = 'numpy'
    return OptionMarketData(prices=prices, strikes=strikes, ttm=ttm,
                            spot=spot, w=w, framework=framework)


def test_initialization():
    model = DiscreteLocalVolatilityModel()
    assert model.maturities is None
    assert model.parameters is None
    assert model.slices == []


def test_fit(option_market_data):
    model = DiscreteLocalVolatilityModel()
    bounds = [(0.5, 1.5), (0.5, 1.5)]
    model.fit(option_market_data, bounds)
    assert model.maturities is not None
    assert model.parameters is not None
    assert len(model.slices) == len(option_market_data.ttm)


def test_get_prices(option_market_data):
    model = DiscreteLocalVolatilityModel()
    bounds = [(0.5, 1.5), (0.5, 1.5)]
    model.fit(option_market_data, bounds)
    k = option_market_data.strikes
    tau = option_market_data._ttm
    print(tau)
    prices = model.get_prices(k, tau)
    assert len(prices) == len(tau)
    assert all(len(p) == len(k[i]) for i, p in enumerate(prices))


def test_get_ivs(option_market_data):
    model = DiscreteLocalVolatilityModel()
    bounds = [(0.5, 1.5), (0.5, 1.5)]
    model.fit(option_market_data, bounds)
    k = option_market_data.strikes
    tau = option_market_data._ttm
    ivs = model.get_ivs(k, tau)
    assert len(ivs) == len(tau)
    assert all(len(iv) == len(k[i]) for i, iv in enumerate(ivs))


def test_get_dlvs(option_market_data: OptionMarketData):
    model = DiscreteLocalVolatilityModel()
    bounds = [(0.5, 1.5), (0.5, 1.5)]
    model.fit(option_market_data, bounds)
    k = option_market_data.strikes
    tau = option_market_data._ttm
    dlvs = model.get_dlvs(k, tau)
    assert len(dlvs) == len(tau)
    assert all(len(dlv) == len(k[i]) for i, dlv in enumerate(dlvs))


def test_dlv():
    spot = np.array([1])
    strikes = [
        [0.8, 0.9, 1.0, 1.1, 1.2],
        [0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.5]
    ]
    ttm = [0.01, 0.02]
    sigma = np.array([1.0])

    prices = [
        bs.price(spot, np.array(strike_slice), np.array(t), sigma)
        for strike_slice, t in zip(strikes, ttm)
    ]

    market_data = OptionMarketData(
        prices=prices,
        strikes=strikes,
        ttm=ttm,
        spot=1.0,
        w=1
    )

    bounds = [(0.2, 5), (0.2, 5)]

    dlv = DiscreteLocalVolatilityModel()
    dlv.fit(market_data, bounds=bounds)
    strikes_np = [np.array(strike_slice) for strike_slice in strikes]

    prices_interp = dlv.get_prices(strikes_np, ttm)

    for p, p_expected in zip(prices_interp, prices):
        np.testing.assert_allclose(p[1:-1], p_expected[1:-1], atol=1e-6)


if __name__ == '__main__':
    test_dlv()
