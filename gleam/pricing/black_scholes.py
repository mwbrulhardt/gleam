import numpy as np
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
from scipy.stats import norm

from gleam.framework import resolve

jackel_iv = np.vectorize(implied_volatility)


def d1(S, K, tau, sigma, r, q):
    fw = resolve(S)
    return (fw.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * tau**0.5)


def d2(S, K, tau, sigma, r, q):
    return d1(S, K, tau, sigma, r, q) - sigma * tau**0.5


def price(S, K, tau, sigma, r: float = 0, q: float = 0, w: float = 1):
    fw = resolve(S)
    v1 = S * fw.exp(-q * tau) * fw.norm_cdf(w * d1(S, K, tau, sigma, r, q))
    v2 = K * fw.exp(-r * tau) * fw.norm_cdf(w * d2(S, K, tau, sigma, r, q))
    P = w * (v1 - v2)
    return P


def iv(P, S, K, tau, r: float = 0, q: float = 0, w: float = 1):
    option_type = np.where(w == 1, "c", "p")
    return jackel_iv(P, S, K, tau, r, q, option_type)


def delta(S, K, tau, sigma, r: float = 0, q: float = 0, w: float = 1):
    return w * np.exp(-q * tau) * norm.cdf(w * d1(S, K, tau, sigma, r, q))


def delta_strike(S, delta, tau, sigma, r: float = 0, q: float = 0, w: float = 1):
    fw = resolve(S)
    delta = fw.where(w == 1, delta, 1 - delta)
    u = fw.norm_icdf(delta * np.exp(q * tau))
    K = S * fw.exp(-(u * sigma * tau**0.5 - (r - q + 0.5 * sigma**2) * tau))
    return K


def vega(S, K, tau, sigma, r: float = 0, q: float = 0, w: float = 1):
    fw = resolve(S)
    V = fw.where(w == 1, S, K)
    d = fw.where(w == 1, d1(S, K, tau, sigma, r, q), d2(S, K, tau, sigma, r, q))
    rate = fw.where(w == 1, q, r)
    return V * fw.exp(-rate * tau) * fw.norm_pdf(d) * tau**0.5


def parity(P, S, K, tau, r: float = 0, q: float = 0, w: float = 1):
    fw = resolve(S)
    return P + w * (S * fw.exp(-q * tau) - K * fw.exp(-r * tau))
