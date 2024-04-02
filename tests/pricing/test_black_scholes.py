import numpy as np
import torch

from gleam.pricing import black_scholes as bs


def test_d1():
    S = [1600, 1200, 4320]
    K = [1000, 1000, 1000]
    tau = [5 / 365, 10 / 365, 15 / 365]
    sigma = [0.5, 0.6, 0.7]
    r = 0
    q = 0

    expected = np.array([8.06068582, 1.88548948, 10.38247742])

    # should work for numpy
    d1 = bs.d1(np.array(S), np.array(K), np.array(tau), np.array(sigma), r, q)
    np.testing.assert_allclose(d1, expected)

    # should work for torch
    d1 = bs.d1(
        torch.tensor(S), torch.tensor(K), torch.tensor(tau), torch.tensor(sigma), r, q
    ).numpy()
    np.testing.assert_allclose(d1, expected, atol=1e-6)
