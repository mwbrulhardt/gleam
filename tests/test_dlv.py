import pytest

from gleam.models.dlv import *


@pytest.fixture
def option_market_data():
    prices = [[0.11, 0.01, 0.001], [0.12, 0.015, 0.002]]
    strikes = [[0.9, 1.0, 1.1], [0.9, 1.0, 1.1]]
    ttm = [0.1, 0.5]
    spot = 1.0
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


@pytest.fixture
def option_market_data():
    prices = [
        [0.122, 0.11, 0.1, 0.001, 0.0001],
        [0.21, 0.12, 0.112, 0.0012, 0.0001]
    ]
    strikes = [[0.88, 0.9, 1.0, 1.1, 1.2], [0.8, 0.9, 1.0, 1.1, 1.2]]
    maturities = [0.5, 1.0]
    data = OptionMarketData(
        strikes=strikes,
        prices=prices,
        ttm=maturities,
        spot=1.0
    )
    return data


def test_delta():
    C = np.array([1.0, 0.9, 0.8])
    dk_plus = np.array([0.1, 0.1])
    expected_delta = np.array([-1.0, -1.0])
    assert np.allclose(delta(C, dk_plus), expected_delta)


def test_augment_data(option_market_data):
    weights = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    bounds = [(0.75, 1.25), (0.7, 1.3)]
    prices = option_market_data._prices
    strikes = option_market_data._strikes
    ttm = option_market_data._ttm
    C, k, t, w = augment_data(prices, strikes, ttm, weights, bounds)
    assert len(C) == len(k) == len(t) == len(w)

    assert np.allclose(C[0], [0.20, 0.1, 0., 0., 0.])
    assert np.allclose(C[1], [0.35, 0.25, *prices[0], 0., 0.])
    assert np.allclose(C[2], [0.40, 0.30, *prices[1], 0., 0.])
    assert np.allclose(k[0], [0.8, 0.9, 1.0, 1.1, 1.2])
    assert np.allclose(k[1], [0.65, 0.75, *strikes[0], 1.25, 1.35])
    assert np.allclose(k[2], [0.60, 0.70, *strikes[1], 1.30, 1.40])
    assert np.allclose(t, [0.0, *ttm])


def test_correct_numerical_errors():
    C = [
        np.array([0.2, 0.1, 0.0, 0.0, 0.0]),
        np.array([0.32, 0.22, 0.11, 0.05, 0.01, 0.001, 0.0001]),
    ]
    C_model = [None, None]
    C_model[0] = cp.Variable(5)
    C_model[1] = cp.Variable(7)
    C_model[0].value = np.array([0.2, 0.1, 0.0, 0.0, 0.0])
    C_model[1].value = np.array([0.3, 0.2, 0.11, 0.05, 0.01, 0.0001, 0.0001])
    k = [
        np.array([0.8, 0.9, 1.0, 1.1, 1.2]),
        np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    ]
    t = np.array([0.0, 1.0])
    outputs = correct_numerical_errors(C, C_model, k, t)
    assert "C" in outputs
    assert "C_model" in outputs
    assert "p" in outputs
    assert "backward_theta" in outputs
    assert "gamma" in outputs
    assert "lv" in outputs


def test_calibrate_dlvs(option_market_data):
    weights = [[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]]
    bounds = [(0.7, 1.3), (0.7, 1.3)]
    variables = calibrate_dlvs(
        option_market_data._prices,
        option_market_data._strikes,
        option_market_data._ttm,
        weights,
        bounds
    )
    assert "t" in variables
    assert "k" in variables
    assert "C" in variables
    assert "C_model" in variables
    assert "p" in variables
    assert "backward_theta" in variables
    assert "gamma" in variables
    assert "lv" in variables


def test_compute_psi():
    k = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    psi = compute_psi(k)
    assert psi.shape == (3, 5)


def test_compute_forward_op():
    k = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    gamma = np.array([0.1, 0.2, 0.1,])
    backward_theta = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    forward_op = compute_forward_op(k, gamma, backward_theta)
    assert forward_op.shape == (3, 3)


def test_slice_transition_op():
    slice_data = {
        "num_strikes": 5,
        "k": np.array([0.8, 0.9, 1.0, 1.1, 1.2]),
        "k_next": np.array([0.8, 0.9, 1.0, 1.1, 1.2]),
        "maturity_bounds": (0.0, 1.0),
        "density": np.array([0.1, 0.2, 0.4, 0.2, 0.1]),
        "omega": np.zeros((5, 5)),
        "psi": np.zeros((5, 5)),
        "forward_op_decomp": (np.eye(5), np.eye(5), np.eye(5)),
    }
    slice_obj = Slice(**slice_data)
    transition_op = slice_obj.transition_op(0.5)
    assert transition_op.shape == (5, 5)


def test_slice_interpolate(option_market_data):
    model = DiscreteLocalVolatilityModel()
    bounds = [(0.7, 1.3), (0.7, 1.3)]
    model.fit(option_market_data, bounds)
    k_interp = np.array([0.85, 0.95, 1.05, 1.15])
    C_hat = model.slices[0].interpolate(k_interp, 0.5)
    assert C_hat.shape == (4,)


def test_dlv_model_fit(option_market_data):
    model = DiscreteLocalVolatilityModel()
    bounds = [(0.7, 1.3), (0.7, 1.3)]
    model.fit(option_market_data, bounds)
    assert model.maturities is not None
    assert model.parameters is not None
    assert len(model.slices) == len(option_market_data.ttm)


def test_dlv_model_get_prices(option_market_data):
    model = DiscreteLocalVolatilityModel()
    bounds = [(0.7, 1.3), (0.7, 1.3)]
    model.fit(option_market_data, bounds)
    k_test = [np.array([0.9, 1.0, 1.1]), np.array([0.9, 1.0, 1.1])]
    tau_test = [0.5, 1.0]
    prices_test = model.get_prices(k_test, tau_test)
    assert len(prices_test) == len(tau_test)
    assert all(len(p) == len(k) for p, k in zip(prices_test, k_test))


def test_dlv_model_get_ivs(option_market_data):
    model = DiscreteLocalVolatilityModel()
    bounds = [(0.7, 1.3), (0.7, 1.3)]
    model.fit(option_market_data, bounds)
    k_test = [np.array([0.9, 1.0, 1.1]), np.array([0.9, 1.0, 1.1])]
    tau_test = [0.5, 1.0]
    ivs_test = model.get_ivs(k_test, tau_test)
    assert len(ivs_test) == len(tau_test)
    assert all(len(iv) == len(k) for iv, k in zip(ivs_test, k_test))


def test_dlv_model_get_dlvs(option_market_data):
    model = DiscreteLocalVolatilityModel()
    bounds = [(0.7, 1.3), (0.7, 1.3)]
    model.fit(option_market_data, bounds)
    k_test = [np.array([0.9, 1.0, 1.1]), np.array([0.9, 1.0, 1.1])]
    tau_test = [0.5, 1.0]
    dlvs_test = model.get_dlvs(k_test, tau_test)
    assert len(dlvs_test) == len(tau_test)
    assert all(len(dlv) == len(k) for dlv, k in zip(dlvs_test, k_test))


if __name__ == '__main__':
    test_dlv()
