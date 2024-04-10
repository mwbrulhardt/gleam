"""
Discrete Local Volatility Model
===============================

This module provides an implementation of the Discrete Local Volatility (DLV) model
for option pricing and implied volatility calculation. The DLV model is a discrete-
space, discrete-time model that calibrates to a set of given option prices and
allows for efficient pricing and risk management.

The main components of the module are:

- `delta`: Function to compute the delta of call prices.
- `augment_data`: Function to augment the input data with ghost and boundary strikes
  and call prices.
- `correct_numerical_errors`: Function to correct numerical errors in the model
  calibration.
- `calibrate_dlvs`: Function to calibrate the discrete local volatility surface.
- `compute_psi`: Function to compute the psi matrix for interpolation.
- `compute_forward_op`: Function to compute the forward operator for the DLV model.
- `Slice`: Class representing a slice of the DLV model, containing transition operators
  and interpolation matrices.
- `DiscreteLocalVolatilityModel`: Class implementing the DLV model, providing methods
  for calibration, pricing, and implied volatility calculation.

The `DiscreteLocalVolatilityModel` class inherits from `Estimator`, `PricingModel`,
and `ImpliedVolatilityModel` base classes (not shown in this module) and provides
the following methods:

- `fit`: Calibrate the DLV model to market data.
- `get_prices`: Get call prices for a given set of strike prices and maturities.
- `get_ivs`: Get implied volatilities for a given set of strike prices and maturities.
- `get_dlvs`: Get discrete local volatilities for a given set of strike prices and
  maturities.

The module also includes several helper functions for data augmentation, error
correction, and matrix computations.
"""

from dataclasses import dataclass
from typing import List, Tuple

import cvxpy as cp
import numpy as np

import gleam.black_scholes as bs
from gleam.models.base import (
    Estimator,
    ImpliedVolatilityModel,
    OptionMarketData,
    PricingModel,
)


def delta(C: np.array, dk_plus: np.array):
    """
    Compute the delta of the call prices.

    Parameters
    ----------
    C : np.array
        Call prices.
    dk_plus : np.array
        Differences between consecutive strike prices.

    Returns
    -------
    np.array
        Delta of the call prices.
    """
    return (C[1:] - C[:-1]) / dk_plus


def augment_data(
    prices: List[List[float]],
    strikes: List[List[float]],
    maturities: List[float],
    weights: List[List[float]],
    bounds: List[Tuple[float, float]],
) -> (List[np.ndarray], List[np.ndarray], np.ndarray, List[np.ndarray]):
    """
    Augment the input data with ghost and boundary strikes and call prices.

    Parameters
    ----------
    prices : list,
        List of call prices for each maturity.
    strikes : list,
        List of strike prices for each maturity.
    maturities : list,
        List of maturities.
    weights : list,
        List of weights for each call price.
    bounds : list,
        List of lower and upper bounds for each maturity.

    Returns
    -------
    tuple
        Tuple containing augmented call prices, strike prices, maturities, and weights.

    Notes
    -----
    This function adds ghost and boundary strikes and call prices to the input data.
    It ensures that the strike range is increasing and that there are no repeated strikes
    on each maturity.
    """
    # Run validation check
    assert len(prices) == len(strikes) == len(maturities)
    assert len(bounds) == len(strikes)
    assert all([len(b) == 2 for b in bounds])

    t = np.array([0] + maturities)
    m = t.shape[0]

    k = [1 + 0.1 * np.arange(-2, 3)]
    C = [(1 - k[0]).clip(min=0)]
    w = [None]

    for j in range(1, m):
        b1, b2 = bounds[j - 1]
        g1, g2 = b1 - 0.1, b2 + 0.1

        # Add ghost and boundary strikes
        k += [np.array([g1, b1] + strikes[j - 1] + [b2, g2])]

        # Add ghost and boundary call prices (instrinsic value)
        c_left = [max(1 - g1, 0), max(1 - b1, 0)]
        c_right = [max(1 - b2, 0), max(1 - g2, 0)]
        C += [np.array(c_left + prices[j - 1] + c_right)]

        # Check conditions for expanding strike range
        assert min(k[-1]) < 1, "Missing strike on left side."
        assert max(k[-1]) > 1, "Missing strike on right side."
        assert min(k[-1]) <= min(k[-2]), "Grid not expanding. Lower bound violation."
        assert max(k[-1]) >= max(k[-2]), "Grid not expanding. Upper bound violation."
        assert len(np.unique(strikes[j - 1])) == len(
            strikes[j - 1]
        ), f"Repeated strikes on maturity {t[j]}"

        w += [np.array(weights[j - 1])]

    return C, k, t, w


def correct_numerical_errors(
    C: List[np.array], C_model: List[cp.Variable], k: List[np.array], t: np.array
):
    """
    Correct numerical errors in the model calibration.

    Parameters
    ----------
    C : list
        List of call prices for each maturity.
    C_model : list,
        List of modeled call prices for each maturity.
    k : list,
        List of strike prices for each maturity.
    t : np.array,
        Array of maturities.

    Returns
    -------
    dict
        Dictionary containing corrected model outputs.

    Notes
    -----
    This function corrects numerical errors that may arise during the model calibration.
    It ensures that the call prices at the boundary strikes match the intrinsic value
    and computes the gamma, backward theta, and local volatility.
    """
    m = t.shape[0]

    gamma = m * [None]
    backward_theta = m * [None]
    p = m * [None]
    C_mod = m * [None]
    lv = m * [None]

    C_mod[0] = (1 - k[0]).clip(min=0)

    p[0] = np.array([0, 1, 0])
    dk_plus = k[0][2:] - k[0][1:-1]
    dk_minus = k[0][1:-1] - k[0][:-2]
    gamma[0] = 2 * p[0] / (dk_plus + dk_minus)

    for j in range(1, m):
        C_mod[j] = C_model[j].value
        C_mod[j][:2] = (1 - k[j][:2]).clip(min=0)
        C_mod[j][-2:] = (1 - k[j][-2:]).clip(min=0)

        # i = -1 : n_j
        dk_plus = k[j][2:] - k[j][1:-1]
        dk_minus = k[j][1:-1] - k[j][:-2]
        delta0 = delta(C_mod[j][:-1], dk_minus)
        delta1 = delta(C_mod[j][1:], dk_plus)

        # i = -1 : n_j
        p[j] = delta1 - delta0
        gamma[j] = 2 * (delta1 - delta0) / (dk_plus + dk_minus)

        # i = -2 : n_j + 1
        omega = (k[j - 1][1:-1].reshape(-1, 1).T - k[j].reshape(-1, 1)).clip(min=0)
        backward_theta[j] = C_mod[j] - omega @ p[j - 1]

        # i = -1 : n_j
        dt = t[j] - t[j - 1]

        lv[j] = (
            np.where(
                (gamma[j][1:-1] > 0) & (backward_theta[j][2:-2] >= 0),
                2 * (backward_theta[j][2:-2] / (gamma[j][1:-1] * k[j][2:-2] ** 2 * dt)),
                0,
            )
            ** 0.5
        )

    outputs = {
        "t": t,
        "k": k,
        "C": C,
        "C_model": [C_mod[j] for j in range(m)],
        "p": [p[j] for j in range(m)],
        "backward_theta": [None] + [backward_theta[j] for j in range(1, m)],
        "gamma": [gamma[j] for j in range(m)],
        "lv": [None] + [lv[j] for j in range(1, m)],
    }

    return outputs


def calibrate_dlvs(
    prices: List[List[float]],
    strikes: List[List[float]],
    maturities: List[float],
    weights: List[List[float]],
    bounds: List[Tuple[float, float]],
    local_vol_min: float = 0.01,
    local_vol_max: float = 4,
    implied_vol_min: float = 0.05,
    implied_vol_max: float = 3,
):
    """
    Calibrate the discrete local volatility surface.

    Parameters
    ----------
    prices : List[List[float]]
        List of call prices for each maturity.
    strikes : List[List[float]]
        List of strike prices for each maturity.
    maturities : List[float]
        List of maturities.
    weights : List[List[float]]
        List of weights for each call price.
    bounds : List[Tuple[float, float]]
        List of lower and upper bounds for each maturity.
    local_vol_min : float, optional
        Minimum local volatility, by default 0.01.
    local_vol_max : float, optional
        Maximum local volatility, by default 4.
    implied_vol_min : float, optional
        Minimum implied volatility, by default 0.05.
    implied_vol_max : float, optional
        Maximum implied volatility, by default 3.

    Returns
    -------
    dict
        Dictionary containing the calibrated model outputs.

    Notes
    -----
    This function calibrates the discrete local volatility surface by finding the
    closest arbitrage-free call prices that satisfy the constraints on local and
    implied volatility. It uses linear programming to solve the optimization problem.
    """
    assert 0 <= local_vol_min <= local_vol_max, "Local volatility boundaries violated."

    lv_min, lv_max = local_vol_min, local_vol_max
    iv_min, iv_max = implied_vol_min, implied_vol_max

    # Augment data and validate
    C, k, t, w = augment_data(prices, strikes, maturities, weights, bounds)

    m = t.shape[0]

    # Variables
    C_model = [cp.Variable(len(C[j]), nonneg=True) for j in range(m)]
    C_model[0].value = (1 - k[0]).clip(min=0)

    # Constraints
    constraints = [C_model[0] == (1 - k[0]).clip(min=0)]

    gamma = m * [None]
    backward_theta = m * [None]
    p = m * [None]

    p[0] = np.array([0, 1, 0])
    dk_plus = k[0][2:] - k[0][1:-1]
    dk_minus = k[0][1:-1] - k[0][:-2]
    gamma[0] = 2 * p[0] / (dk_plus + dk_minus)

    loss = 0

    for j in range(1, m):
        # i = -1 : n_j
        dk_plus = k[j][2:] - k[j][1:-1]
        dk_minus = k[j][1:-1] - k[j][:-2]
        delta0 = delta(C_model[j][:-1], dk_minus)
        delta1 = delta(C_model[j][1:], dk_plus)

        # i = -1 : n_j
        p[j] = delta1 - delta0
        gamma[j] = 2 * (delta1 - delta0) / (dk_plus + dk_minus)

        # i = -2 : n_j + 1
        omega = (k[j - 1][1:-1].reshape(-1, 1).T - k[j].reshape(-1, 1)).clip(min=0)
        backward_theta[j] = C_model[j] - omega @ p[j - 1]

        # i = -1 : n_j
        dt = t[j] - t[j - 1]
        lb = 0.5 * cp.multiply(gamma[j], k[j][1:-1] ** 2) * dt * lv_min**2
        ub = 0.5 * cp.multiply(gamma[j], k[j][1:-1] ** 2) * dt * lv_max**2

        constraints += [
            # i = 0 : n_j - 1
            gamma[j][1:-1] >= 0,
            # i = 0 : n_j - 1
            backward_theta[j][2:-2] >= lb[1:-1],
            # i = 0 : n_j - 1
            backward_theta[j][2:-2] <= ub[1:-1],
            # Call prices of boundary and ghost strikes must equal instrinsic value
            # which is the same as having them equal the call prices at the boundary
            # strikes
            C_model[j][:2] == C[j][:2],
            C_model[j][-2:] == C[j][-2:],
            C_model[j] >= bs.price(1, k[j], t[j], iv_min),
            C_model[j] <= bs.price(1, k[j], t[j], iv_max),
        ]

        # i = 0 : n_j - 1
        loss += w[j].T @ cp.abs(C[j][2:-2] - C_model[j][2:-2])

    objective = cp.Minimize(loss)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS)

    if not np.isfinite(problem.value):
        raise ValueError(f"Optimization infeasible. Value: {problem.value}")

    return correct_numerical_errors(C, C_model, k, t)


def compute_psi(k: np.ndarray):
    """
    Compute the psi matrix for interpolation.

    Parameters
    ----------
    k : np.ndarray
        Strike prices.

    Returns
    -------
    np.ndarray
        Psi matrix.

    Notes
    -----
    The psi matrix is used for interpolating the call prices and densities between
    different strike grids. It is computed using finite differences.
    """
    n = len(k) - 4

    dk_plus = k[1:] - k[:-1]

    a = np.zeros([n + 2, n + 4])
    a[:, : n + 2] = np.diag(1 / dk_plus[:-1])

    b = np.zeros([n + 2, n + 4])
    b[:, 1 : n + 3] = np.diag(
        -(dk_plus[:-1] + dk_plus[1:]) / (dk_plus[:-1] * dk_plus[1:])
    )

    c = np.zeros([n + 2, n + 4])
    c[:, 2 : n + 4] = np.diag(1 / dk_plus[1:])

    psi = a + b + c

    return psi


def compute_forward_op(k: np.ndarray, gamma: np.ndarray, backward_theta: np.ndarray):
    """
    Compute the forward operator for the discrete local volatility model.

    Parameters
    ----------
    k : np.ndarray
        Strike prices.
    gamma : np.ndarray
        Gamma values.
    backward_theta : np.ndarray
        Backward theta values.

    Returns
    -------
    np.ndarray
        Forward operator.

    Notes
    -----
    The forward operator represents the transition matrix between consecutive time steps
    in the discrete local volatility model. It is computed using an implicit finite
    difference scheme.
    """
    dk_plus = k[2:] - k[1:-1]
    dk_minus = k[1:-1] - k[:-2]
    bt = backward_theta[1:-1]

    gamma_plus = 2 / ((dk_plus + dk_minus) * dk_plus)
    gamma_center = 1 / (dk_plus * dk_minus)
    gamma_minus = 2 / ((dk_plus + dk_minus) * dk_minus)

    u = np.where((gamma > 0) & (bt > 0), bt / gamma, 0)
    w_minus = u * gamma_minus
    w_center = u * gamma_center
    w_plus = u * gamma_plus

    a = np.diag(1 + 2 * w_center)
    b = np.diag(-w_minus[1:], 1)
    c = np.diag(-w_plus, -1)[:-1, :-1]

    I_inv = a + b + c

    return np.linalg.inv(I_inv)


@dataclass
class Slice:
    """
    Represents a slice of the discrete local volatility model.

    Attributes
    ----------
    num_strikes : int
        Number of strikes in the slice.
    k : np.ndarray
        Strike prices for the current maturity.
    k_next : np.ndarray
        Strike prices for the next maturity.
    maturity_bounds : Tuple[float, float]
        Lower and upper bounds of the maturity range.
    density : np.ndarray
        Density values for the current maturity.
    omega : np.ndarray
        Omega matrix for interpolation.
    psi : np.ndarray
        Psi matrix for interpolation.
    forward_op_decomp : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Decomposition of the forward operator.
    """

    num_strikes: int
    k: np.ndarray
    k_next: np.ndarray
    maturity_bounds: Tuple[float, float]
    density: np.ndarray
    omega: np.ndarray
    psi: np.ndarray
    forward_op_decomp: Tuple[np.ndarray, np.ndarray, np.ndarray]

    def transition_op(self, tau: float) -> np.ndarray:
        """
        Compute the transition operator for a given maturity.

        Parameters
        ----------
        tau : float
            Maturity for which to compute the transition operator.

        Returns
        -------
        np.ndarray
            Transition operator.

        Raises
        ------
        AssertionError
            If the maturity is outside the bounds of the slice.

        Notes
        -----
        The transition operator represents the transition probabilities between
        strike prices at the given maturity. It is computed using the decomposition
        of the forward operator and interpolation matrices.
        """
        lower, upper = self.maturity_bounds
        assert lower < tau <= upper, "Maturity outside of range."
        U, D, U_inv = self.forward_op_decomp

        lower, upper = self.maturity_bounds

        beta = (tau - lower) / (upper - lower)
        f_op = U @ D**beta @ U_inv

        xi = self.psi @ self.omega

        return f_op @ xi

    def interpolate(self, k: np.ndarray, tau: float) -> np.ndarray:
        """
        Interpolate call prices for a given set of strike prices and maturity.

        Parameters
        ----------
        k : np.ndarray
            Strike prices for which to interpolate the call prices.
        tau : float
            Maturity for which to interpolate the call prices.

        Returns
        -------
        np.ndarray
            Interpolated call prices.

        Raises
        ------
        AssertionError
            If the maturity is outside the bounds of the slice or if the strike prices
            are not increasing.

        Notes
        -----
        The call prices are interpolated using the transition operator and the density
        values of the slice. The interpolation is performed using the omega matrix.
        """
        lower, upper = self.maturity_bounds
        assert lower < tau <= upper, "Maturity outside of range."
        assert np.all(k[:-1] < k[1:]), "Strikes are not increasing."

        p_hat = self.transition_op(tau) @ self.density

        omega = (self.k_next[1:-1].reshape(1, -1) - k.reshape(-1, 1)).clip(min=0)
        C_hat = omega @ p_hat

        return C_hat


class DiscreteLocalVolatilityModel(Estimator, PricingModel, ImpliedVolatilityModel):
    """
    Discrete Local Volatility model for option pricing and implied volatility.

    Attributes
    ----------
    maturities : np.ndarray
        Maturities of the model.
    parameters : dict
        Parameters of the calibrated model.
    slices : List[Slice]
        Slices of the model, representing the transition operators and densities.
    """

    def __init__(self):
        self.maturities = None
        self.parameters = None
        self.slices = []

    def fit(
        self,
        option_market_data: OptionMarketData,
        bounds: List[Tuple[float, float]],
        weights: np.ndarray = None,
        local_vol_min: float = 0.01,
        local_vol_max: float = 4,
    ):
        """
        Calibrate the discrete local volatility model to market data.

        Parameters
        ----------
        option_market_data : OptionMarketData
            Market data for options, including prices, strikes, and maturities.
        bounds : List[Tuple[float, float]]
            Bounds for each maturity, representing the range of admissible strikes.
        weights : np.ndarray, optional
            Weights for each option in the calibration, by default None.
        local_vol_min : float, optional
            Minimum local volatility, by default 0.01.
        local_vol_max : float, optional
            Maximum local volatility, by default 4.

        Notes
        -----
        The model is calibrated using the `calibrate_dlvs` function, which
        finds the closest arbitrage-free call prices that satisfy the
        constraints on local volatility. The calibrated parameters are
        stored in the `parameters` attribute, and the slices of the model
        are computed and stored in the `slices` attribute.
        """
        prices = [p.tolist() for p in option_market_data.xprices]
        strikes = [k.tolist() for k in option_market_data.xstrikes]
        maturities = option_market_data._ttm
        if weights is None:
            weights = [len(p) * [1.0] for p in option_market_data.xstrikes]

        # Calibrate
        variables = calibrate_dlvs(
            prices, strikes, maturities, weights, bounds, local_vol_min, local_vol_max
        )
        self.parameters = variables

        # Compute operators
        self.maturities = variables["t"]
        m = len(self.maturities)

        self.slices = []

        for j in range(m - 1):
            k0 = variables["k"][j]
            k1 = variables["k"][j + 1]
            gamma1 = variables["gamma"][j + 1]
            backward_theta1 = variables["backward_theta"][j + 1]

            # Compute the I (forward operator)
            f_op = compute_forward_op(k1, gamma1, backward_theta1)
            lam, U = np.linalg.eig(f_op)

            self.slices += [
                Slice(
                    num_strikes=k0.shape[0] - 4,
                    k=k0,
                    k_next=k1,
                    maturity_bounds=(self.maturities[j], self.maturities[j + 1]),
                    density=variables["p"][j],
                    omega=(k0[1:-1].reshape(1, -1) - k1.reshape(-1, 1)).clip(min=0),
                    psi=compute_psi(k1),
                    forward_op_decomp=(U, np.diag(lam), np.linalg.inv(U)),
                )
            ]

    def _interpolate_price(self, k: np.ndarray, tau: float) -> np.ndarray:
        """
        Interpolate call prices for a given set of strike prices and maturity.

        Parameters
        ----------
        k : np.ndarray
            Strike prices for which to interpolate the call prices.
        tau : float
            Maturity for which to interpolate the call prices.

        Returns
        -------
        np.ndarray
            Interpolated call prices.

        Raises
        ------
        AssertionError
            If the maturity is outside the bounds of the model.
        """
        assert tau <= self.maturities[-1], "Out of bounds: ttm"

        if tau <= 0:
            return (1 - k).clip(min=0)

        j = np.searchsorted(self.maturities, tau, side="left") - 1

        idx = np.argsort(k)

        C_hat = self.slices[j].interpolate(k[idx], tau)

        return C_hat[np.argsort(idx)]

    def _get_prices(self, k: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """
        Get call prices for a given set of strike prices and maturities.

        Parameters
        ----------
        k : np.ndarray
            Strike prices for which to get the call prices.
        tau : np.ndarray
            Maturities for which to get the call prices.

        Returns
        -------
        np.ndarray
            Call prices.

        Notes
        -----
        The call prices are computed by interpolating the prices for each unique maturity
        in `tau` and then reshaping the result to match the input shape of `k`.
        """
        shape = k.shape

        if len(k.shape) == 2 and len(tau.shape) == 1 and k.shape[0] == tau.shape[0]:
            k = k.flatten()
            tau = tau.repeat(shape[1])

        C_hat = np.zeros_like(k)

        T = np.unique(tau)

        for i in range(len(T)):
            c = tau == T[i]
            C_hat[c] = self._interpolate_price(k[c], T[i])

        return C_hat.reshape(*shape)

    def get_prices(self, k: List[np.array], tau: List[float]) -> List[np.array]:
        """
        Get call prices for a given set of strike prices and maturities.

        Parameters
        ----------
        k : List[np.array]
            Strike prices for each maturity.
        tau : List[float]
            Maturities for which to get the call prices.

        Returns
        -------
        List[np.array]
            Call prices for each maturity.

        Raises
        ------
        AssertionError
            If the maturities are not monotonically increasing.
        """
        C_hat = [
            len(k_slice)
            * [
                0,
            ]
            for k_slice in k
        ]

        assert all(
            [t1 > t0 for t0, t1 in zip(tau[:-1], tau[1:])]
        ), "The time to maturities have to be monotonically increasing."

        T = np.unique(tau)

        for i in range(len(T)):
            C_hat[i] = self._interpolate_price(k[i], T[i])

        return C_hat

    def get_ivs(self, k: List[np.array], tau: List[float]) -> List[np.array]:
        """
        Get implied volatilities for a given set of strike prices and maturities.

        Parameters
        ----------
        k : List[np.array]
            Strike prices for each maturity.
        tau : List[float]
            Maturities for which to get the implied volatilities.

        Returns
        -------
        List[np.array]
            Implied volatilities for each maturity.
        """
        prices = self.get_prices(k, tau)
        ivs = [
            bs.iv(V=V, S=1.0, tau=t, K=strikes) for V, strikes, t in zip(prices, k, tau)
        ]
        return ivs

    def _interpolate_dlv(self, k: np.ndarray, tau: float) -> np.ndarray:
        """
        Interpolate discrete local volatilities for a given set of strike prices and maturity.

        Parameters
        ----------
        k : np.ndarray
            Strike prices for which to interpolate the discrete local volatilities.
        tau : float
            Maturity for which to interpolate the discrete local volatilities.

        Returns
        -------
        np.ndarray
            Interpolated discrete local volatilities.

        Raises
        ------
        AssertionError
            If the maturity is outside the bounds of the model.
        """
        assert tau <= self.maturities[-1], "Out of bounds: ttm"

        j = np.searchsorted(self.maturities, tau, side="left") - 1
        s = self.slices[j]

        idx = np.argsort(k)

        k_aug = np.concatenate([s.k_next[:2], k[idx], s.k_next[-2:]])

        omega = (s.k[1:-1].reshape(1, -1) - k_aug.reshape(-1, 1)).clip(min=0)

        C_tilda = omega @ s.density
        C_hat = self.slices[j].interpolate(k_aug, tau)

        dt = tau - s.maturity_bounds[0]

        bt = C_hat - C_tilda

        dk_plus = k_aug[2:] - k_aug[1:-1]
        dk_minus = k_aug[1:-1] - k_aug[:-2]
        delta0 = (C_hat[1:-1] - C_hat[:-2]) / dk_minus
        delta1 = (C_hat[2:] - C_hat[1:-1]) / dk_plus

        p = delta1 - delta0

        # Clip negligible amounts very close/below zero
        gamma = (2 * p / (dk_plus + dk_minus)).clip(min=0)

        lv = (
            np.where(
                (gamma[1:-1] > 0) & (bt[2:-2] >= 0),
                2 * (bt[2:-2] / (gamma[1:-1] * k_aug[2:-2] ** 2 * dt)),
                0,
            )
            ** 0.5
        )

        return lv[np.argsort(idx)]

    def get_dlvs(self, k: List[np.ndarray], tau: List[float]) -> List[np.ndarray]:
        """
        Get discrete local volatilities for a given set of strike prices and maturities.

        Parameters
        ----------
        k : List[np.ndarray]
            Strike prices for each maturity.
        tau : List[float]
            Maturities for which to get the discrete local volatilities.

        Returns
        -------
        List[np.ndarray]
            Discrete local volatilities for each maturity.
        """
        lv_hat = list()

        T = np.unique(tau)

        for i in range(len(T)):
            lv_hat.append(np.sqrt(self._interpolate_dlv(k[i], T[i])))

        return lv_hat
