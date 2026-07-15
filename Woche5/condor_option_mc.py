"""
Exercise 4: Monte Carlo pricing of the Condor option

    V(S,T) = 2(S_T - 8)   for 8  < S_T <= 9
           = 2             for 9  < S_T <= 10
           = 2(11 - S_T)   for 10 < S_T <= 11
           = 0             otherwise

in the Black-Scholes model, S0 = 9, r = 0.06, sigma = 0.1, T = 1.

Under the risk-neutral measure S_T = G(X) with X ~ N(0,1) and
    G(x) = S0 * exp( (r - sigma^2/2) T + sigma*sqrt(T)*x ).

Three estimators, each for N = 10^2, 10^3, 10^4:
  (i)   standard Monte Carlo estimator
  (ii)  antithetic estimator
  (iii) stratified estimator with proportional allocation, strata
            A1 = G^-1((8,9]),  A2 = G^-1((9,10]),  A3 = G^-1((10,11])

For each estimator we report the price estimate, the approximate
(1-alpha) confidence interval (alpha = 0.05), and the running time
(measured with time.perf_counter, the Python analogue of tic/toc).
"""

import time
from math import log, sqrt, exp, erf
from statistics import NormalDist
import numpy as np

# ---------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------
S0 = 9.0
r = 0.06
sigma = 0.1
T = 1.0

ALPHA = 0.05
Z_CRIT = 1.959963984540054  # standard normal 1-alpha/2 quantile for alpha=0.05
SEED = 12345

_normal = NormalDist()  # standard normal, mean 0, sd 1


def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def norm_ppf(p):
    return _normal.inv_cdf(p)


# ---------------------------------------------------------------
# G and its inverse
# ---------------------------------------------------------------
def G(x):
    return S0 * np.exp((r - sigma ** 2 / 2) * T + sigma * sqrt(T) * x)


def G_inv(level):
    return (log(level / S0) - (r - sigma ** 2 / 2) * T) / (sigma * sqrt(T))


# strata boundaries in x-space, x1 < x2 < x3 < x4
x1, x2, x3, x4 = G_inv(8), G_inv(9), G_inv(10), G_inv(11)

# probability weights of the three strata under N(0,1)
p1 = norm_cdf(x2) - norm_cdf(x1)
p2 = norm_cdf(x3) - norm_cdf(x2)
p3 = norm_cdf(x4) - norm_cdf(x3)


# ---------------------------------------------------------------
# Condor payoff
# ---------------------------------------------------------------
def payoff(ST):
    ST = np.asarray(ST, dtype=float)
    out = np.zeros_like(ST)
    m1 = (ST > 8) & (ST <= 9)
    m2 = (ST > 9) & (ST <= 10)
    m3 = (ST > 10) & (ST <= 11)
    out[m1] = 2 * (ST[m1] - 8)
    out[m2] = 2.0
    out[m3] = 2 * (11 - ST[m3])
    return out


def confidence_interval(mean, se):
    return mean - Z_CRIT * se, mean + Z_CRIT * se


# ---------------------------------------------------------------
# (i) Standard Monte Carlo estimator
# ---------------------------------------------------------------
def standard_mc(N, rng):
    t0 = time.perf_counter()
    Z = rng.standard_normal(N)
    ST = G(Z)
    disc_payoff = exp(-r * T) * payoff(ST)
    price = np.mean(disc_payoff)
    se = np.std(disc_payoff, ddof=1) / sqrt(N)
    t1 = time.perf_counter()
    return price, se, t1 - t0


# ---------------------------------------------------------------
# (ii) Antithetic estimator
# ---------------------------------------------------------------
def antithetic_mc(N, rng):
    t0 = time.perf_counter()
    M = N // 2  # number of antithetic pairs (2*M paths in total)
    Z = rng.standard_normal(M)

    ST_plus = G(Z)
    ST_minus = G(-Z)

    payoff_plus = exp(-r * T) * payoff(ST_plus)
    payoff_minus = exp(-r * T) * payoff(ST_minus)
    pair_avg = 0.5 * (payoff_plus + payoff_minus)

    price = np.mean(pair_avg)
    se = np.std(pair_avg, ddof=1) / sqrt(M)
    t1 = time.perf_counter()
    return price, se, t1 - t0


# ---------------------------------------------------------------
# (iii) Stratified estimator, proportional allocation
# ---------------------------------------------------------------
def sample_stratum(n, a, b, rng):
    """Sample n draws of X ~ N(0,1) restricted to (a, b] via inverse cdf."""
    Fa, Fb = norm_cdf(a), norm_cdf(b)
    U = rng.uniform(0.0, 1.0, size=n)
    P = Fa + U * (Fb - Fa)
    P = np.clip(P, 1e-15, 1 - 1e-15)
    return np.array([norm_ppf(p) for p in P])


def stratified_mc(N, rng):
    t0 = time.perf_counter()

    weights = [p1, p2, p3]
    bounds = [(x1, x2), (x2, x3), (x3, x4)]

    # proportional allocation: N_k = round(N * p_k), remainder to largest stratum
    N_k = [int(round(N * w)) for w in weights]
    N_k[int(np.argmax(weights))] += N - sum(N_k)
    N_k = [max(n, 2) for n in N_k]  # need >=2 for sample variance

    means = []
    vars_ = []
    for n_k, (a, b) in zip(N_k, bounds):
        X = sample_stratum(n_k, a, b, rng)
        ST = G(X)
        disc_payoff = exp(-r * T) * payoff(ST)
        means.append(np.mean(disc_payoff))
        vars_.append(np.var(disc_payoff, ddof=1))

    price = sum(w * m for w, m in zip(weights, means))
    var_price = sum((w ** 2) * v / n for w, v, n in zip(weights, vars_, N_k))
    se = sqrt(var_price)

    t1 = time.perf_counter()
    return price, se, t1 - t0


if __name__ == "__main__":
    print(f"Strata boundaries (x-space): x1={x1:.4f}, x2={x2:.4f}, "
          f"x3={x3:.4f}, x4={x4:.4f}")
    print(f"Strata weights: p1={p1:.4f}, p2={p2:.4f}, p3={p3:.4f}, "
          f"p(outside)={1 - p1 - p2 - p3:.4f}\n")

    N_values = [100, 1_000, 10_000]

    for N in N_values:
        print(f"===== N = {N} =====")

        rng = np.random.default_rng(SEED)
        price, se, rt = standard_mc(N, rng)
        lo, hi = confidence_interval(price, se)
        print(f"(i)   standard MC   : price = {price:.6f}, "
              f"{100*(1-ALPHA):.0f}% CI = [{lo:.6f}, {hi:.6f}], "
              f"runtime = {rt:.6f} s")

        rng = np.random.default_rng(SEED)
        price, se, rt = antithetic_mc(N, rng)
        lo, hi = confidence_interval(price, se)
        print(f"(ii)  antithetic    : price = {price:.6f}, "
              f"{100*(1-ALPHA):.0f}% CI = [{lo:.6f}, {hi:.6f}], "
              f"runtime = {rt:.6f} s")

        rng = np.random.default_rng(SEED)
        price, se, rt = stratified_mc(N, rng)
        lo, hi = confidence_interval(price, se)
        print(f"(iii) stratified    : price = {price:.6f}, "
              f"{100*(1-ALPHA):.0f}% CI = [{lo:.6f}, {hi:.6f}], "
              f"runtime = {rt:.6f} s")

        print()
