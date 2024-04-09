import pytest
import pandas as pd
import numpy as np
from gleam.models.realised import get_windows, windowed, fit_distribution, \
    Slice, RealisedModelDLV


@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    prices = np.random.randint(50, 100, size=len(dates))
    return pd.DataFrame({'price': prices}, index=dates)


def test_get_windows(sample_data):
    delta = pd.Timedelta(days=1)
    num_lags = 3
    windows = get_windows(sample_data, delta, num_lags)
    assert windows.shape[1] == num_lags
    assert np.all(np.isin(windows, sample_data.index.values))


def test_windowed(sample_data):
    delta = pd.Timedelta(days=1)
    num_lags = 3
    windowed_data = windowed(sample_data, delta, num_lags)
    assert windowed_data.shape[1] == num_lags
    assert windowed_data.shape[0] == len(sample_data) - num_lags + 1


def test_fit_distribution(sample_data):
    delta = pd.Timedelta(days=1)
    strikes = np.linspace(0.9, 1.1, 5)
    slice = fit_distribution(sample_data, delta, strikes)
    assert isinstance(slice, Slice), "Returned value should be of type Slice."
    assert len(slice.prices) == len(strikes)
    assert len(slice.strikes) == len(strikes)


def test_realised_model_dlv(sample_data):
    model = RealisedModelDLV()
    deltas = [pd.Timedelta(days=1), pd.Timedelta(days=7)]
    strikes = [np.linspace(0.9, 1.1, 5), np.linspace(0.8, 1.2, 5)]
    bounds = [(0.5, 1.5), (0.5, 1.5)]
    model.fit(sample_data, deltas, strikes, bounds)

    k = [np.linspace(0.9, 1.1, 5), np.linspace(0.8, 1.2, 5)]
    tau = [0.01, 0.015]
    prices = model.get_prices(k, tau)
    assert len(prices) == len(k), \
        "Length of prices and strikes need to coincide."

    ivs = model.get_ivs(k, tau)
    assert len(ivs) == len(k), \
        "Lenght of IVs and strikes need to coincide."
