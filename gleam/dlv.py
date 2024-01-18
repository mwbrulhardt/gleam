from dataclasses import dataclass
from typing import Tuple, List

import cvxpy as cp
import numpy as np

import gleam.black_scholes as bs


def delta(C, dk_plus):
    return (C[1:] - C[:-1]) / dk_plus


def augment_data(prices, strikes, maturities, weights, bounds):
    # Run validation check
    assert len(prices) == len(strikes) == len(maturities)
    assert bounds.shape == (len(strikes), 2)

    t = np.array([0] + maturities)
    m = t.shape[0]

    k = [1 + 0.1*np.arange(-2, 3)]
    C = [(1 - k[0]).clip(min=0)]
    w = [None]

    for j in range(1, m):
        b1, b2 = bounds[j - 1, :]
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
        assert len(np.unique(strikes[j - 1])) == len(strikes[j - 1]), f"Repeated strikes on maturity {t[j]}"

        w += [np.array(weights[j - 1])]
    
    return C, k, t, w


def correct_numerical_errors(C, C_model, k, t):
    m = t.shape[0]
    dt_plus = t[1:] - t[:-1]

    gamma = m * [None]
    backward_theta = m * [None]
    p = m * [None]
    C_mod = m * [None]
    lv = m * [None]

    C_mod[0] = (1 - k[0]).clip(min=0)

    p[0] = np.array([0, 1, 0])
    dk_plus = k[0][2:] - k[0][1:-1]
    dk_minus =  k[0][1:-1] - k[0][:-2]
    gamma[0] = 2*p[0] / (dk_plus + dk_minus)

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
        gamma[j] = 2*(delta1 - delta0) / (dk_plus + dk_minus)

        # i = -2 : n_j + 1
        omega = (k[j - 1][1:-1].reshape(-1, 1).T - k[j].reshape(-1, 1)).clip(min=0)
        backward_theta[j] = C_mod[j] - omega@p[j - 1]

        # i = -1 : n_j
        theta = backward_theta[j][2:-2] / dt_plus[j - 1]
        gamma_ = gamma[j][1:-1]
        k_ = k[j][2:-2]

        lv[j] = np.where(
            (gamma_ < 0) | (theta < 0) | ((gamma_ == 0) & (theta > 0)),
            np.inf,
            2*(theta / (gamma_*k_**2))
        )**0.5


    outputs = {
        "t": t,
        "k": k,
        "C": C,
        "C_model": [C_mod[j] for j in range(m)],
        "p": [p[j] for j in range(m)],
        "backward_theta": [None] + [backward_theta[j] for j in range(1, m)],
        "gamma": [gamma[j] for j in range(m)],
        "lv": [None] + [lv[j] for j in range(1, m)]
    }

    return outputs


def calibrate_dlvs(
        prices: List[List[float]],
        strikes: List[List[float]],
        maturities: List[float],
        weights: List[List[float]],
        bounds: np.ndarray,
        local_vol_min: float = 0.01,
        local_vol_max: float = 4,
        implied_vol_min: float = 0.05,
        implied_vol_max: float = 3
    ):
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
    dk_minus =  k[0][1:-1] - k[0][:-2]
    gamma[0] = 2*p[0] / (dk_plus + dk_minus)

    loss = 0

    for j in range(1, m):

        # i = -1 : n_j
        dk_plus = k[j][2:] - k[j][1:-1]
        dk_minus = k[j][1:-1] - k[j][:-2]
        delta0 = delta(C_model[j][:-1], dk_minus)
        delta1 = delta(C_model[j][1:], dk_plus)

        # i = -1 : n_j
        p[j] = delta1 - delta0
        gamma[j] = 2*(delta1 - delta0) / (dk_plus + dk_minus)

        # i = -2 : n_j + 1
        omega = (k[j - 1][1:-1].reshape(-1, 1).T - k[j].reshape(-1, 1)).clip(min=0)
        backward_theta[j] = C_model[j] - omega@p[j - 1]

        # i = -1 : n_j
        dt = t[j] - t[j - 1]
        lb = 0.5*cp.multiply(gamma[j], k[j][1:-1]**2)*dt*lv_min**2
        ub = 0.5*cp.multiply(gamma[j], k[j][1:-1]**2)*dt*lv_max**2

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
            C_model[j] <= bs.price(1, k[j], t[j], iv_max)
        ]

        # i = 0 : n_j - 1
        loss += w[j].T@cp.abs(C[j][2:-2] - C_model[j][2:-2])

    objective = cp.Minimize(loss)
    
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if not np.isfinite(problem.value):
        raise ValueError(f"Optimization infeasible. Value: {problem.value}")

    return correct_numerical_errors(C, C_model, k, t)


def compute_psi(k: np.ndarray):
    n = len(k) - 4

    dk_plus = k[1:] - k[:-1]

    a = np.zeros([n + 2, n + 4])
    a[:, :n + 2] = np.diag(1 / dk_plus[:-1])

    b = np.zeros([n + 2, n + 4])
    b[:, 1:n + 3] = np.diag(-(dk_plus[:-1] + dk_plus[1:]) / (dk_plus[:-1]*dk_plus[1:]))

    c = np.zeros([n + 2, n + 4])
    c[:, 2:n + 4] = np.diag(1 / dk_plus[1:])

    psi = a + b + c

    return psi


def compute_forward_op(k: np.ndarray, gamma: np.ndarray, backward_theta: np.ndarray):
    dk_plus = k[2:] - k[1:-1]
    dk_minus = k[1:-1] - k[:-2]
    bt = backward_theta[1:-1] 

    gamma_plus = 2 / ((dk_plus + dk_minus) * dk_plus)
    gamma_center = 1 / (dk_plus * dk_minus)
    gamma_minus = 2 / ((dk_plus + dk_minus) * dk_minus)

    u = np.where((gamma > 0) & (bt > 0), bt / gamma, 0)
    w_minus = u*gamma_minus
    w_center = u*gamma_center
    w_plus = u*gamma_plus

    a = np.diag(1 + 2 * w_center)
    b = np.diag(-w_minus[1:], 1)
    c = np.diag(-w_plus, -1)[:-1, :-1]

    I_inv = a + b + c

    return np.linalg.inv(I_inv)


def reformat(C: np.ndarray, k: np.ndarray, tau: np.ndarray, w: np.ndarray = None):
    maturities = list(np.sort(np.unique(tau)))

    if not isinstance(w, np.ndarray):
        w = (1 / k.shape[0]) * np.ones_like(k)

    prices = []
    strikes = []
    weights = []

    for j in range(len(maturities)):

        c = (tau == maturities[j])

        idx = np.argsort(k[c])

        strikes += [list(k[c][idx])]
        prices += [list(C[c][idx])]
        weights += [list(w[c][idx])]

    return prices, strikes, maturities, weights


@dataclass
class Slice:
    num_strikes: int
    k: np.ndarray
    k_next: np.ndarray
    maturity_bounds: Tuple[float, float]
    density: np.ndarray
    omega: np.ndarray
    psi: np.ndarray
    forward_op_decomp: Tuple[np.ndarray,np.ndarray, np.ndarray]

    def transition_op(self, tau: float) -> np.ndarray:
        lower, upper = self.maturity_bounds
        assert lower < tau <= upper, "Maturity outside of range."
        U, D, U_inv = self.forward_op_decomp

        lower, upper = self.maturity_bounds

        beta = (tau - lower) / (upper - lower)
        I = U@D**beta@U_inv

        xi = self.psi@self.omega

        return I@xi

    def interpolate(self, k: np.ndarray, tau: float) -> np.ndarray:
        lower, upper = self.maturity_bounds
        assert lower < tau <= upper, "Maturity outside of range."
        assert np.all(k[:-1] < k[1:]), "Strikes are not increasing."

        p_hat = self.transition_op(tau)@self.density

        omega = (self.k_next[1:-1].reshape(1, -1) - k.reshape(-1, 1)).clip(min=0)
        C_hat = omega@p_hat

        return C_hat


class DiscreteLocalVolatilityModel:

    def __init__(self):
        self.maturities = None
        self.parameters = None
        self.slices = []

    def calibrate(
            self,
            prices: np.ndarray,
            strikes: np.ndarray,
            maturities: np.ndarray,
            bounds: np.ndarray,
            weights: np.ndarray = None,
            local_vol_min: float = 0.01,
            local_vol_max: float = 4
        ):
        # Reformat data
        prices, strikes, maturities, weights = reformat(
            prices, 
            strikes, 
            maturities, 
            weights
        )

        # Calibrate
        variables = calibrate_dlvs(
            prices,
            strikes,
            maturities,
            weights,
            bounds,
            local_vol_min,
            local_vol_max
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
            I = compute_forward_op(k1, gamma1, backward_theta1)
            lam, U = np.linalg.eig(I)

            self.slices += [Slice(
                num_strikes=k0.shape[0] - 4,
                k=k0,
                k_next=k1,
                maturity_bounds=(self.maturities[j], self.maturities[j + 1]),
                density=variables["p"][j],
                omega=(k0[1:-1].reshape(1, -1) - k1.reshape(-1, 1)).clip(min=0),
                psi=compute_psi(k1),
                forward_op_decomp=(U, np.diag(lam), np.linalg.inv(U))
            )] 

    def _interpolate_price(self, k: np.ndarray, tau: float) -> np.ndarray:
        assert tau <= self.maturities[-1], "Out of bounds: ttm"

        if tau <= 0:
            return (1 - k).clip(min=0)

        j = np.searchsorted(self.maturities, tau, side="left") - 1

        idx = np.argsort(k)

        C_hat = self.slices[j].interpolate(k[idx], tau)

        return C_hat[np.argsort(idx)]

    def get_prices(self, k: np.ndarray, tau: np.ndarray) -> np.ndarray:
        shape = k.shape

        if len(k.shape) == 2 and len(tau.shape) == 1 and k.shape[0] == tau.shape[0]:
            k = k.flatten()
            tau = tau.repeat(shape[1])

        C_hat = np.zeros_like(k)

        T = np.unique(tau)
        
        for i in range(len(T)):
            c = (tau == T[i])
            C_hat[c] = self._interpolate_price(k[c], T[i])

        return C_hat.reshape(*shape)

    def _interpolate_dlv(self, k: np.ndarray, tau: float) -> np.ndarray:
        assert tau <= self.maturities[-1], "Out of bounds: ttm"

        j = np.searchsorted(self.maturities, tau, side="left") - 1
        s = self.slices[j]

        idx = np.argsort(k)

        k_aug = np.concatenate([s.k_next[:2], k[idx], s.k_next[-2:]])

        omega = (s.k[1:-1].reshape(1, -1) - k_aug.reshape(-1, 1)).clip(min=0)

        C_tilda = omega@s.density
        C_hat = self.slices[j].interpolate(k_aug, tau)

        dt = tau - s.maturity_bounds[0]

        bt = C_hat - C_tilda

        dk_plus = k_aug[2:] - k_aug[1:-1]
        dk_minus = k_aug[1:-1] - k_aug[:-2]
        delta0 = (C_hat[1:-1] - C_hat[:-2]) / dk_minus
        delta1 = (C_hat[2:] - C_hat[1:-1]) / dk_plus

        p = delta1 - delta0

        # Clip negligible amounts very close/below zero
        gamma = (2*p / (dk_plus + dk_minus)).clip(min=0)

        lv = np.where(
            gamma[1:-1] > 0,
            2*(bt[2:-2] / (gamma[1:-1]*k_aug[2:-2]**2*dt)),
            0
        )**0.5

        return lv[np.argsort(idx)]
    
    def get_dlvs(self, k: np.ndarray, tau: np.ndarray) -> np.ndarray:
        shape = k.shape

        if len(k.shape) == 2 and len(tau.shape) == 1 and k.shape[0] == tau.shape[0]:
            k = k.flatten()
            tau = tau.repeat(shape[1])

        lv_hat = np.zeros_like(k)

        T = np.unique(tau)
        
        for i in range(len(T)):
            c = (tau == T[i])
            lv_hat[c] = self._interpolate_dlv(k[c], T[i])

        return lv_hat.reshape(*shape)**0.5