
import numpy as np
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
from scipy.stats import norm


jackel_iv = np.vectorize(implied_volatility)


def d1(S, K, tau, sigma, r, q):
    return (np.log(S / K) + (r - q + 0.5*sigma**2)*tau) / (sigma*tau**0.5)


def d2(S, K, tau, sigma, r, q):
    return d1(S, K, tau, sigma, r, q) - sigma*tau**0.5


def price(S, K, tau, sigma, r: float = 0, q: float = 0, w: float = 1):
    v1 = S*np.exp(-q*tau)*norm.cdf(w*d1(S, K, tau, sigma, r, q))
    v2 = K*np.exp(-r*tau)*norm.cdf(w*d2(S, K, tau, sigma, r, q))
    P = w*(v1 - v2)
    return P


def iv(P, S, K, tau, r: float = 0, q: float = 0, w: float = 1):
    option_type = np.where(w == 1, "c", "p")
    return jackel_iv(P, S, K, tau, r, q, option_type)


def delta(S, K, tau, sigma, r: float = 0, q: float = 0, w: float = 1):
    return w*np.exp(-q*tau)*norm.cdf(w*d1(S, K, tau, sigma, r, q))


def delta_strike(S, delta, tau, sigma, r: float = 0, q: float = 0, w: float = 1):
    delta = np.where(w == 1, delta, 1 - delta)
    u = norm.ppf(delta*np.exp(q*tau))
    K = S*np.exp(-(u*sigma*tau**0.5 - (r - q + 0.5*sigma**2)*tau))
    return K


def vega(S, K, tau, sigma, r: float = 0, q: float = 0, w: float = 1):
    V = np.where(w == 1, S, K)
    d = np.where(
        w == 1, 
        d1(S, K, tau, sigma, r, q), 
        d2(S, K, tau, sigma, r, q)
    )
    rate = np.where(w == 1, q, r)
    return V*np.exp(-rate*tau)*norm.pdf(d)*tau**0.5


def parity(P, S, K, tau, r: float = 0, q: float = 0, w: float = 1):
    return P + w*(S*np.exp(-q*tau) - K*np.exp(-r*tau))