import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from py_vollib.black_scholes_merton import black_scholes_merton as bsm
from py_vollib.black_scholes_merton.greeks.analytical import delta as bs_delta
from py_vollib.black_scholes_merton.greeks.analytical import vega as bs_vega
from py_vollib.black_scholes_merton.implied_volatility import \
    implied_volatility as bs_iv

from gleam.black_scholes import price, delta, vega, iv, delta_strike, \
    d_plus, d_minus


@pytest.mark.parametrize("tensor_type", [np.array, torch.Tensor])
def test_price(tensor_type):
    S = tensor_type([100, 100])
    K = tensor_type([90, 110])
    tau = tensor_type([1, 1])
    sigma = tensor_type([0.2, 0.2])
    r = tensor_type([0.05, 0.05])
    q = tensor_type([0.0, 0.0])

    expected_call = [bsm('c', S[i], K[i], tau[i], r[i], sigma[i], q[i]) for i
                     in
                     range(len(S))]
    expected_put = [bsm('p', S[i], K[i], tau[i], r[i], sigma[i], q[i]) for i in
                    range(len(S))]

    call_prices = price(S, K, tau, sigma, r, q)
    put_prices = price(S, K, tau, sigma, r, q, w=-1)

    return_type = np.ndarray if tensor_type == np.array else torch.Tensor
    assert type(call_prices) == return_type
    assert type(put_prices) == return_type

    assert_allclose(call_prices, expected_call, atol=1e-6, rtol=1e-5)
    assert_allclose(put_prices, expected_put, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("tensor_type", [np.array, torch.Tensor])
def test_delta(tensor_type):
    S = tensor_type([100, 100])
    K = tensor_type([90, 110])
    tau = tensor_type([1, 1])
    sigma = tensor_type([0.2, 0.2])
    r = tensor_type([0.05, 0.05])
    q = tensor_type([0.0, 0.0])

    expected_call_delta = [
        bs_delta('c', S[i], K[i], tau[i], r[i], sigma[i], q[i])
        for i in range(len(S))]
    expected_put_delta = [
        bs_delta('p', S[i], K[i], tau[i], r[i], sigma[i], q[i]) for
        i in range(len(S))]

    call_delta = delta(S, K, tau, sigma, r, q)
    put_delta = delta(S, K, tau, sigma, r, q, w=-1)

    return_type = np.ndarray if tensor_type == np.array else torch.Tensor

    assert type(call_delta) == return_type
    assert type(put_delta) == return_type

    assert_allclose(call_delta, expected_call_delta, atol=1e-6, rtol=1e-5)
    assert_allclose(put_delta, expected_put_delta, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("tensor_type", [np.array, torch.Tensor])
def test_vega(tensor_type):
    S = tensor_type([100, 100])
    K = tensor_type([90, 110])
    tau = tensor_type([1, 1])
    sigma = tensor_type([0.2, 0.2])
    r = tensor_type([0.05, 0.05])
    q = tensor_type([0.0, 0.0])

    expected_vega = [bs_vega(S[i], K[i], tau[i], r[i], sigma[i], q[i]) for i in
                     range(len(S))]

    output_vega = vega(S, K, tau, sigma, r, q)

    return_type = np.ndarray if tensor_type == np.array else torch.Tensor
    assert type(output_vega) == return_type

    assert_allclose(output_vega, expected_vega, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("tensor_type", [np.array, torch.Tensor])
def test_iv(tensor_type):
    V = tensor_type([22.63, 9.37])
    S = tensor_type([100, 100])
    K = tensor_type([90, 110])
    tau = tensor_type([1, 1])
    r = tensor_type([0.05, 0.05])
    q = tensor_type([0.0, 0.0])

    expected_call_iv = [bs_iv(V[i], S[i], K[i], tau[i], r[i], q[i], 'c') for i
                        in range(len(V))]
    expected_put_iv = [bs_iv(V[i], S[i], K[i], tau[i], r[i], q[i], 'p') for i
                       in range(len(V))]

    call_iv = iv(V, S, K, tau, r, q)
    put_iv = iv(V, S, K, tau, r, q, w=-1)

    return_type = np.ndarray if tensor_type == np.array else torch.Tensor
    assert type(call_iv) == return_type
    assert type(put_iv) == return_type
    assert_allclose(call_iv, expected_call_iv, atol=1e-6, rtol=1e-5)
    assert_allclose(put_iv, expected_put_iv, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("tensor_type", [np.array, torch.Tensor])
def test_delta_strike(tensor_type):
    S = tensor_type([100, 100])
    delta_val = tensor_type([0.7, 0.3])
    tau = tensor_type([1, 1])
    sigma = tensor_type([0.2, 0.2])
    r = tensor_type([0.05, 0.05])
    q = tensor_type([0.0, 0.0])

    call_strikes = delta_strike(S, delta_val, tau, sigma, r, q)
    put_strikes = delta_strike(S, delta_val, tau, sigma, r, q, w=-1)

    call_deltas = [
        bs_delta('c', S[i], call_strikes[i], tau[i], r[i], sigma[i], q[i])
        for i in range(len(S))]
    put_deltas = [
        bs_delta('p', S[i], put_strikes[i], tau[i], r[i], sigma[i], q[i])
        for i in range(len(S))]

    return_type = np.ndarray if tensor_type == np.array else torch.Tensor
    assert type(call_strikes) == return_type
    assert type(put_strikes) == return_type
    assert_allclose(call_deltas, delta_val, atol=1e-3)
    assert_allclose(put_deltas, delta_val, atol=1e-3)


@pytest.mark.parametrize("tensor_type", [np.array, torch.Tensor])
def test_d_plus(tensor_type):
    F = tensor_type([100, 100])
    K = tensor_type([90, 110])
    tau = tensor_type([1, 1])
    sigma = tensor_type([0.2, 0.2])

    expected_d_plus = [(np.log(F[i] / K[i]) + 0.5 * sigma[i] ** 2 * tau[i]) / (
        sigma[i] * np.sqrt(tau[i])) for i in range(len(F))]

    output_d_plus = d_plus(F, K, tau, sigma)

    return_type = np.ndarray if tensor_type == np.array else torch.Tensor
    assert type(output_d_plus) == return_type
    np.testing.assert_allclose(output_d_plus, expected_d_plus)


@pytest.mark.parametrize("tensor_type", [np.array, torch.Tensor])
def test_d_minus(tensor_type):
    F = tensor_type([100, 100])
    K = tensor_type([90, 110])
    tau = tensor_type([1, 1])
    sigma = tensor_type([0.2, 0.2])

    expected_d_minus = [
        (np.log(F[i] / K[i]) + 0.5 * sigma[i] ** 2 * tau[i]) / (
            sigma[i] * np.sqrt(tau[i])) - sigma[i] * np.sqrt(tau[i]) for i
        in range(len(F))]

    output_d_minus = d_minus(F, K, tau, sigma)

    return_type = np.ndarray if tensor_type == np.array else torch.Tensor
    assert type(output_d_minus) == return_type
    np.testing.assert_allclose(output_d_minus, expected_d_minus)
