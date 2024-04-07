import numpy as np
import pytest
import torch

from gleam.models.base import OptionMarketData


@pytest.fixture
def option_market_data():
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


def test_initialization(option_market_data):
    assert option_market_data is not None


def test_properties(option_market_data):
    assert len(option_market_data.prices) == len(
        option_market_data.strikes) == len(option_market_data.ttm)
    assert len(option_market_data.xstrikes) == len(
        option_market_data.xprices) == len(option_market_data.rates) == len(
        option_market_data.discount_factors) == len(
        option_market_data.forwards) == len(option_market_data.implied_vols)
    assert len(option_market_data.sofd) == len(
        option_market_data.prices)


def test_framework_conversion(option_market_data):
    option_market_data.torch()
    assert option_market_data._framework == 'torch'
    option_market_data.numpy()
    assert option_market_data._framework == 'numpy'


def test_implied_vols(option_market_data):
    implied_vols = option_market_data.implied_vols
    assert all(isinstance(vol, np.ndarray) for vol in implied_vols)


def test_sofd(option_market_data):
    sofd = option_market_data.sofd
    assert all(
        bfs is None or isinstance(bfs, np.ndarray) for bfs in sofd)


def test_property_dtype(option_market_data):
    # test prices
    prices = option_market_data.prices
    prices_list = option_market_data._prices
    for p, p_expected in zip(prices, prices_list):
        np.testing.assert_allclose(p, np.array(p_expected), atol=1e-6)

    # test prices
    strikes = option_market_data.strikes
    strikes_list = option_market_data._strikes
    for s, s_expected in zip(strikes, strikes_list):
        np.testing.assert_allclose(s, np.array(s_expected), atol=1e-6)


def test_to_conversion_numpy(option_market_data):
    assert option_market_data.to([1, 2, 3]).dtype == np.array([1, 2, 3]).dtype


def test_to_conversion_torch(option_market_data):
    option_market_data.torch()
    assert option_market_data.to([1, 2, 3]).dtype == torch.tensor(
        [1, 2, 3]).dtype


def test_invalid_input_raises_error():
    with pytest.raises(AssertionError):
        OptionMarketData(prices=[[1, 2], [3, 4]], strikes=[[1, 2], [3]],
                         ttm=[0.1, 0.2], spot=100.0)

