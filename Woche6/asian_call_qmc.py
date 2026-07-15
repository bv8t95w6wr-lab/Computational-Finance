"""
Exercise 2 (QMC version): Quasi-Monte Carlo pricing of the arithmetic
Asian call

    V(S,T) = ( (1/n) * sum_{i=1}^n S_{t_i} - K )^+

for S0=5, K=6, r=0.05, sigma=0.3, T=1, t_i = i/10, i=1,...,10.

The n=10 driving normal shocks per path are obtained from a
10-dimensional Halton sequence, transformed to N(0,1) via the inverse
standard-normal cdf (norm.ppf), instead of pseudo-random numbers.

Three estimators, each for N = 10^2, 10^3, 10^4:
  (i)   QMC-standard estimator
  (ii)  QMC-antithetic estimator (Halton point u paired with 1-u,
        which maps to Z and -Z after the inverse-cdf transform)
  (iii) QMC-control variate estimator (control variable: geometric
        Asian call, priced in closed form)

We report the price estimate and the running time (time.perf_counter,
the Python analogue of tic/toc) for each estimator and each N.
No confidence intervals are computed here: Halton points are
deterministic (not i.i.d.), so the classical CLT-based CI does not
apply directly to plain QMC estimators.
"""

import time
from math import log, sqrt, exp
from statistics import NormalDist
import numpy as np

# ---------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------
S0 = 5.0
K = 6.0
r = 0.05
sigma = 0.3
T = 1.0
n_dates = 10
dt = T / n_dates
t_grid = np.array([i / 10 for i in range(1, n_dates + 1)])

_normal = NormalDist()


def norm_cdf(x):
    return _normal.cdf(x)


def norm_ppf(p):
    return _normal.inv_cdf(p)


norm_ppf_vec = np.vectorize(norm_ppf)

# first 10 primes -> Halton bases for the 10-dimensional sequence
PRIMES_10 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


# ---------------------------------------------------------------
# Halton / van der Corput sequence generator
# ---------------------------------------------------------------
def van_der_corput(n_points, base, start=1):
    """First n_points terms of the van der Corput sequence in given base,
    starting at index `start` (start=1 avoids the point 0)."""
    seq = np.zeros(n_points)
    for i in range(n_points):
        n = start + i
        f = 1.0
        r = 0.0
        while n > 0:
            f /= base
            r += f * (n % base)
            n //= base
        seq[i] = r
    return seq


def halton_sequence(n_points, dim=n_dates, start=1):
    """n_points points of the dim-dimensional Halton sequence in (0,1)^dim."""
    bases = PRIMES_10[:dim]
    pts = np.zeros((n_points, dim))
    for d, base in enumerate(bases):
        pts[:, d] = van_der_corput(n_points, base, start=start)
    return pts


# ---------------------------------------------------------------
# Closed-form price of the discretely monitored geometric Asian call
# (used as the control variate)
# ---------------------------------------------------------------
def geometric_asian_call_exact(S0, K, r, sigma, T, t):
    t = np.asarray(t, dtype=float)
    n = len(t)
    tbar = np.mean(t)
    Tmin = np.minimum.outer(t, t)
    Sigma2 = sigma ** 2 * np.sum(Tmin) / n ** 2
    Sigma = sqrt(Sigma2)
    mu = log(S0) + (r - 0.5 * sigma ** 2) * tbar
    d1 = (mu - log(K) + Sigma2) / Sigma
    d2 = d1 - Sigma
    price = exp(-r * T) * (exp(mu + 0.5 * Sigma2) * norm_cdf(d1) - K * norm_cdf(d2))
    return price


# ---------------------------------------------------------------
# Path construction from standard-normal shocks Z (n_points x n_dates)
# ---------------------------------------------------------------
def paths_from_Z(Z):
    increments = (r - 0.5 * sigma ** 2) * dt + sigma * sqrt(dt) * Z
    logS = log(S0) + np.cumsum(increments, axis=1)
    return np.exp(logS)


def arithmetic_payoff(S):
    return np.maximum(np.mean(S, axis=1) - K, 0.0)


def geometric_payoff(S):
    g = np.exp(np.mean(np.log(S), axis=1))
    return np.maximum(g - K, 0.0)


# ---------------------------------------------------------------
# (i) QMC-standard estimator
# ---------------------------------------------------------------
def qmc_standard(N):
    t0 = time.perf_counter()
    U = halton_sequence(N, dim=n_dates, start=1)
    Z = norm_ppf_vec(U)
    S = paths_from_Z(Z)
    disc_payoff = exp(-r * T) * arithmetic_payoff(S)
    price = np.mean(disc_payoff)
    t1 = time.perf_counter()
    return price, t1 - t0


# ---------------------------------------------------------------
# (ii) QMC-antithetic estimator
# ---------------------------------------------------------------
def qmc_antithetic(N):
    t0 = time.perf_counter()
    M = N // 2
    U = halton_sequence(M, dim=n_dates, start=1)
    Z_plus = norm_ppf_vec(U)
    Z_minus = norm_ppf_vec(1.0 - U)  # = -Z_plus (symmetry of the normal cdf)

    S_plus = paths_from_Z(Z_plus)
    S_minus = paths_from_Z(Z_minus)

    payoff_plus = exp(-r * T) * arithmetic_payoff(S_plus)
    payoff_minus = exp(-r * T) * arithmetic_payoff(S_minus)
    pair_avg = 0.5 * (payoff_plus + payoff_minus)

    price = np.mean(pair_avg)
    t1 = time.perf_counter()
    return price, t1 - t0


# ---------------------------------------------------------------
# (iii) QMC-control variate estimator (geometric Asian call control)
# ---------------------------------------------------------------
def qmc_control_variate(N):
    t0 = time.perf_counter()
    C_exact = geometric_asian_call_exact(S0, K, r, sigma, T, t_grid)

    U = halton_sequence(N, dim=n_dates, start=1)
    Z = norm_ppf_vec(U)
    S = paths_from_Z(Z)

    Y = exp(-r * T) * arithmetic_payoff(S)   # target payoff
    X = exp(-r * T) * geometric_payoff(S)    # control payoff

    cov_XY = np.cov(X, Y, ddof=1)[0, 1]
    var_X = np.var(X, ddof=1)
    c_hat = cov_XY / var_X

    adjusted = Y - c_hat * (X - C_exact)
    price = np.mean(adjusted)
    t1 = time.perf_counter()
    return price, t1 - t0, c_hat


if __name__ == "__main__":
    C_exact = geometric_asian_call_exact(S0, K, r, sigma, T, t_grid)
    print(f"Closed-form geometric Asian call price (control variate mean): "
          f"{C_exact:.6f}\n")

    N_values = [100, 1_000, 10_000]

    header = f"{'N':>7} | {'estimator':<18} | {'price':>10} | {'runtime [s]':>12}"
    print(header)
    print("-" * len(header))

    for N in N_values:
        price, rt = qmc_standard(N)
        print(f"{N:7d} | {'QMC-standard':<18} | {price:10.6f} | {rt:12.6f}")

        price, rt = qmc_antithetic(N)
        print(f"{N:7d} | {'QMC-antithetic':<18} | {price:10.6f} | {rt:12.6f}")

        price, rt, c_hat = qmc_control_variate(N)
        print(f"{N:7d} | {'QMC-control var.':<18} | {price:10.6f} | {rt:12.6f}"
              f"   (c_hat = {c_hat:.4f})")
        print("-" * len(header))
