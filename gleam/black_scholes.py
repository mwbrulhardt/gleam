"""Module containing all functions for the black-scholes-merton model.

This file include functions for computing call prices as well as implied volatilities.
"""


import numpy as np
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility

import gleam.framework as fw

jackel_iv = np.vectorize(implied_volatility)


def d1(
    S: fw.TensorTypeOrScalar,
    K: fw.TensorTypeOrScalar,
    tau: fw.TensorTypeOrScalar,
    sigma: fw.TensorTypeOrScalar,
    r: fw.TensorTypeOrScalar,
    q: fw.TensorTypeOrScalar,
) -> fw.TensorTypeOrScalar:
    """Helper function for computing the d1 component in the Black-Scholes-Merton model.

    Parameters
    ----------
    S : fw.TensorTypeOrScalar
        The spot price for the underlying asset.
    K : fw.TensorTypeOrScalar
        The strike price of the option.
    tau: fw.TensorTypeOrScalar
        The time until maturity (in years) for the option (T - t).
    sigma: fw.TensorTypeOrScalar
        The annulaized volatility of the underlying price process.
    r : fw.TensorTypeOrScalar
        The annualized risk-free interest rate, continuously compounded.
    q : fw.TensorTypeOrScalar
        The annualized dividend yield rate, continuously compounded.

    Returns
    -------
    fw.TensorTypeOrScalar
        The value of the d1 component.
    """
    return (fw.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * tau**0.5)


def d2(
    S: fw.TensorTypeOrScalar,
    K: fw.TensorTypeOrScalar,
    tau: fw.TensorTypeOrScalar,
    sigma: fw.TensorTypeOrScalar,
    r: fw.TensorTypeOrScalar,
    q: fw.TensorTypeOrScalar,
) -> fw.TensorTypeOrScalar:
    """Helper function for computing the d2 component in the Black-Scholes-Merton model.

    Parameters
    ----------
    S : fw.TensorTypeOrScalar
        The spot price for the underlying asset.
    K : fw.TensorTypeOrScalar
        The strike price of the option.
    tau: fw.TensorTypeOrScalar
        The time until maturity (in years) for the option (T - t).
    sigma: fw.TensorTypeOrScalar
        The annulaized volatility of the underlying price process.
    r : fw.TensorTypeOrScalar
        The annualized risk-free interest rate, continuously compounded.
    q : fw.TensorTypeOrScalar
        The annualized dividend yield rate, continuously compounded.

    Returns
    -------
    fw.TensorTypeOrScalar
        The value of the d2 component.
    """
    return d1(S, K, tau, sigma, r, q) - sigma * tau**0.5


def price(
    S: fw.TensorTypeOrScalar,
    K: fw.TensorTypeOrScalar,
    tau: fw.TensorTypeOrScalar,
    sigma: fw.TensorTypeOrScalar,
    r: fw.TensorTypeOrScalar = 0,
    q: fw.TensorTypeOrScalar = 0,
    w: fw.TensorTypeOrScalar = 1,
) -> fw.TensorTypeOrScalar:
    """Computes the valuation of an option under the Black-Scholes-Merton model.

    Parameters
    ----------
    S : fw.TensorTypeOrScalar
        The spot price for the underlying asset.
    K : fw.TensorTypeOrScalar
        The strike price of the option.
    tau: fw.TensorTypeOrScalar
        The time until maturity (in years) for the option (T - t).
    sigma: fw.TensorTypeOrScalar
        The annulaized volatility of the underlying price process.
    r : fw.TensorTypeOrScalar, default 0
        The annualized risk-free interest rate, continuously compounded.
    q : fw.TensorTypeOrScalar, default 0
        The annualized dividend yield rate, continuously compounded.
    w : fw.TensorTypeOrScalar, default 1
        The option type, where 1 and -1 denote a call and put, respectively.

    Returns
    -------
    V : fw.TensorTypeOrScalar
        The valuation of an option under the Black-Scholes-Merton model.
    """
    v1 = S * fw.exp(-q * tau) * fw.dist.normal.cdf(w * d1(S, K, tau, sigma, r, q))
    v2 = K * fw.exp(-r * tau) * fw.dist.normal.cdf(w * d2(S, K, tau, sigma, r, q))
    V = w * (v1 - v2)
    return V


def iv(
    V: fw.TensorTypeOrScalar,
    S: fw.TensorTypeOrScalar,
    K: fw.TensorTypeOrScalar,
    tau: fw.TensorTypeOrScalar,
    r: fw.TensorTypeOrScalar = 0,
    q: fw.TensorTypeOrScalar = 0,
    w: fw.TensorTypeOrScalar = 1,
    method: str = "jackel",
) -> fw.TensorTypeOrScalar:
    """Computes the implied volatility of an option under the Black-Scholes-Merton.

    Parameters
    ----------
    V : fw.TensorTypeOrScalar
        The valuation of the option.
    S : fw.TensorTypeOrScalar
        The spot price for the underlying asset.
    K : fw.TensorTypeOrScalar
        The strike price of the option.
    tau: fw.TensorTypeOrScalar
        The time until maturity (in years) for the option (T - t).
    sigma: fw.TensorTypeOrScalar
        The annulaized volatility of the underlying price process.
    r : fw.TensorTypeOrScalar, default 0
        The annualized risk-free interest rate, continuously compounded.
    q : fw.TensorTypeOrScalar, default 0
        The annualized dividend yield rate, continuously compounded.
    w : fw.TensorTypeOrScalar, default 1
        The option type, where 1 and -1 denote a call and put, respectively.
    method : str, default "jackel"
        The method to use for deriving the implied volatility from the option price.

    Returns
    -------
    fw.TensorTypeOrScalar
        The implied volatility of an option under the Black-Scholes-Merton.
    """

    is_numpy = isinstance(V, np.ndarray)
    V, S, K, tau, r, q, w = fw.to_numpy(V, S, K, tau, r, q, w)
    option_type = np.where(w == 1, "c", "p")

    sigma: np.ndarray = jackel_iv(V, S, K, tau, r, q, option_type)

    if is_numpy:
        return sigma
    return fw.to_torch(sigma)


def delta(
    S: fw.TensorTypeOrScalar,
    K: fw.TensorTypeOrScalar,
    tau: fw.TensorTypeOrScalar,
    sigma: fw.TensorTypeOrScalar,
    r: fw.TensorTypeOrScalar = 0,
    q: fw.TensorTypeOrScalar = 0,
    w: fw.TensorTypeOrScalar = 1,
) -> fw.TensorTypeOrScalar:
    """Computes the delta of an option under the Black-Scholes-Merton model.

    Parameters
    ----------
    S : fw.TensorTypeOrScalar
        The spot price for the underlying asset.
    K : fw.TensorTypeOrScalar
        The strike price of the option.
    tau: fw.TensorTypeOrScalar
        The time until maturity (in years) for the option (T - t).
    sigma: fw.TensorTypeOrScalar
        The annulaized volatility of the underlying price process.
    r : fw.TensorTypeOrScalar, default 0
        The annualized risk-free interest rate, continuously compounded.
    q : fw.TensorTypeOrScalar, default 0
        The annualized dividend yield rate, continuously compounded.
    w : fw.TensorTypeOrScalar, default 1
        The option type, where 1 and -1 denote a call and put, respectively.

    Returns
    -------
    fw.TensorTypeOrScalar
        The delta of an option under the Black-Scholes-Merton model.
    """
    return w * fw.exp(-q * tau) * fw.dist.normal.cdf(w * d1(S, K, tau, sigma, r, q))


def vega(
    S: fw.TensorTypeOrScalar,
    K: fw.TensorTypeOrScalar,
    tau: fw.TensorTypeOrScalar,
    sigma: fw.TensorTypeOrScalar,
    r: fw.TensorTypeOrScalar = 0,
    q: fw.TensorTypeOrScalar = 0,
    w: fw.TensorTypeOrScalar = 1,
) -> fw.TensorTypeOrScalar:
    """Computes the vega of an option under the Black-Scholes-Merton model.

    Parameters
    ----------
    S : fw.TensorTypeOrScalar
        The spot price for the underlying asset.
    K : fw.TensorTypeOrScalar
        The strike price of the option.
    tau: fw.TensorTypeOrScalar
        The time until maturity (in years) for the option (T - t).
    sigma: fw.TensorTypeOrScalar
        The annulaized volatility of the underlying price process.
    r : fw.TensorTypeOrScalar, default 0
        The annualized risk-free interest rate, continuously compounded.
    q : fw.TensorTypeOrScalar, default 0
        The annualized dividend yield rate, continuously compounded.
    w : fw.TensorTypeOrScalar, default 1
        The option type, where 1 and -1 denote a call and put, respectively.

    Returns
    -------
    fw.TensorTypeOrScalar
        The vega of an option under the Black-Scholes-Merton model.
    """
    c: fw.TensorType | bool = w == 1
    U = fw.where(c, S, K)
    d = fw.where(c, d1(S, K, tau, sigma, r, q), d2(S, K, tau, sigma, r, q))
    rate = fw.where(c, q, r)
    return U * fw.exp(-rate * tau) * fw.dist.normal.pdf(d) * tau**0.5


def delta_strike(
    S: fw.TensorTypeOrScalar,
    delta: fw.TensorTypeOrScalar,
    tau: fw.TensorTypeOrScalar,
    sigma: fw.TensorTypeOrScalar,
    r: fw.TensorTypeOrScalar = 0,
    q: fw.TensorTypeOrScalar = 0,
    w: fw.TensorTypeOrScalar = 1,
) -> fw.TensorTypeOrScalar:
    """Computes the strike of an option given a delta under the Black-Scholes-Merton model.

    Parameters
    ----------
    S : fw.TensorTypeOrScalar
        The spot price for the underlying asset.
    delta : fw.TensorTypeOrScalar
        The strike price of the option.
    tau: fw.TensorTypeOrScalar
        The time until maturity (in years) for the option (T - t).
    sigma: fw.TensorTypeOrScalar
        The annulaized volatility of the underlying price process.
    r : fw.TensorTypeOrScalar, default 0
        The annualized risk-free interest rate, continuously compounded.
    q : fw.TensorTypeOrScalar, default 0
        The annualized dividend yield rate, continuously compounded.
    w : fw.TensorTypeOrScalar, default 1
        The option type, where 1 and -1 denote a call and put, respectively.

    Returns
    -------
    fw.TensorTypeOrScalar
        The strike of an option given a delta under the Black-Scholes-Merton model.
    """
    delta = fw.where(w == 1, delta, 1 - delta)
    u = fw.dist.normal.icdf(delta * fw.exp(q * tau))
    K = S * fw.exp(-(u * sigma * tau**0.5 - (r - q + 0.5 * sigma**2) * tau))
    return K
