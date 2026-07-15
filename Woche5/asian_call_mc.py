"""
Exercise 2: Monte Carlo pricing of the arithmetic Asian call

    V(S,T) = ( (1/n) * sum_{i=1}^n S_{t_i} - K )^+

for S0=5, K=6, r=0.05, sigma=0.3, T=1, t_i = i/10, i=1,...,10.

Three estimators, each for N = 10^2, 10^3, 10^4:
  (i)   standard Monte Carlo estimator
  (ii)  antithetic estimator
  (iii) control variate estimator (control variable: geometric Asian call,
        priced in closed form)

For each estimator we report the price estimate, the approximate
(1-alpha) confidence interval (alpha=0.05), and the running time
(measured with time.perf_counter, the Python analogue of tic/toc).
"""

import time
import numpy as np
from math import log, sqrt, exp, erf

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
t_grid = np.array([i / 10 for i in range(1, n_dates + 1)])  # t_i = i/10

ALPHA = 0.05
Z_CRIT = 1.959963984540054  # standard normal 1-alpha/2 quantile for alpha=0.05

SEED = 12345  # for reproducibility


# ---------------------------------------------------------------
# Standard normal cdf (no scipy dependency)
# ---------------------------------------------------------------
def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


# ---------------------------------------------------------------
# Closed-form price of the discretely monitored geometric Asian call
# (used as the control variate)
# ---------------------------------------------------------------
def geometric_asian_call_exact(S0, K, r, sigma, T, t):
    t = np.asarray(t, dtype=float)
    n = len(t)
    tbar = np.mean(t)

    # Var[(1/n) sum_i W_{t_i}] = (1/n^2) * sum_{i,j} min(t_i, t_j)
    Tmin = np.minimum.outer(t, t)
    Sigma2 = sigma ** 2 * np.sum(Tmin) / n ** 2
    Sigma = sqrt(Sigma2)

    mu = log(S0) + (r - 0.5 * sigma ** 2) * tbar

    d1 = (mu - log(K) + Sigma2) / Sigma
    d2 = d1 - Sigma

    price = exp(-r * T) * (exp(mu + 0.5 * Sigma2) * norm_cdf(d1) - K * norm_cdf(d2))
    return price


# ---------------------------------------------------------------
# Path simulation: S_{t_1}, ..., S_{t_n} under the risk-neutral GBM
# ---------------------------------------------------------------
def simulate_paths(n_paths, rng, Z=None):
    """Return array of shape (n_paths, n_dates) with S_{t_i} values.
    If Z is given (shape (n_paths, n_dates)) it is used directly
    (needed for the antithetic estimator)."""
    if Z is None:
        Z = rng.standard_normal((n_paths, n_dates))
    increments = (r - 0.5 * sigma ** 2) * dt + sigma * sqrt(dt) * Z
    logS = log(S0) + np.cumsum(increments, axis=1)
    return np.exp(logS)


def arithmetic_payoff(S):
    return np.maximum(np.mean(S, axis=1) - K, 0.0)


def geometric_payoff(S):
    g = np.exp(np.mean(np.log(S), axis=1))
    return np.maximum(g - K, 0.0)


def confidence_interval(mean, se):
    return mean - Z_CRIT * se, mean + Z_CRIT * se


# ---------------------------------------------------------------
# (i) Standard Monte Carlo estimator
# ---------------------------------------------------------------
def standard_mc(N, rng):
    t0 = time.perf_counter()
    S = simulate_paths(N, rng)
    disc_payoff = exp(-r * T) * arithmetic_payoff(S)
    price = np.mean(disc_payoff)
    se = np.std(disc_payoff, ddof=1) / sqrt(N)
    t1 = time.perf_counter()
    return price, se, t1 - t0


# ---------------------------------------------------------------
# (ii) Antithetic estimator
# ---------------------------------------------------------------
def antithetic_mc(N, rng):
    t0 = time.perf_counter()
    M = N // 2  # number of antithetic pairs (2*M paths simulated in total)
    Z = rng.standard_normal((M, n_dates))

    S_plus = simulate_paths(M, rng, Z=Z)
    S_minus = simulate_paths(M, rng, Z=-Z)

    payoff_plus = exp(-r * T) * arithmetic_payoff(S_plus)
    payoff_minus = exp(-r * T) * arithmetic_payoff(S_minus)
    pair_avg = 0.5 * (payoff_plus + payoff_minus)

    price = np.mean(pair_avg)
    se = np.std(pair_avg, ddof=1) / sqrt(M)
    t1 = time.perf_counter()
    return price, se, t1 - t0


# ---------------------------------------------------------------
# (iii) Control variate estimator (geometric Asian call as control)
# ---------------------------------------------------------------
def control_variate_mc(N, rng):
    t0 = time.perf_counter()
    C_exact = geometric_asian_call_exact(S0, K, r, sigma, T, t_grid)

    S = simulate_paths(N, rng)
    Y = exp(-r * T) * arithmetic_payoff(S)   # arithmetic (target) payoff
    X = exp(-r * T) * geometric_payoff(S)    # geometric (control) payoff

    cov_XY = np.cov(X, Y, ddof=1)[0, 1]
    var_X = np.var(X, ddof=1)
    c_hat = cov_XY / var_X

    adjusted = Y - c_hat * (X - C_exact)
    price = np.mean(adjusted)
    se = np.std(adjusted, ddof=1) / sqrt(N)
    t1 = time.perf_counter()
    return price, se, t1 - t0, c_hat


if __name__ == "__main__":
    C_exact = geometric_asian_call_exact(S0, K, r, sigma, T, t_grid)
    print(f"Closed-form geometric Asian call price (control variate mean): "
          f"{C_exact:.6f}\n")

    N_values = [100, 1_000, 10_000]

    for N in N_values:
        print(f"===== N = {N} =====")

        rng = np.random.default_rng(SEED)
        price, se, rt = standard_mc(N, rng)
        lo, hi = confidence_interval(price, se)
        print(f"(i)   standard MC       : price = {price:.6f}, "
              f"{100*(1-ALPHA):.0f}% CI = [{lo:.6f}, {hi:.6f}], "
              f"runtime = {rt:.6f} s")

        rng = np.random.default_rng(SEED)
        price, se, rt = antithetic_mc(N, rng)
        lo, hi = confidence_interval(price, se)
        print(f"(ii)  antithetic        : price = {price:.6f}, "
              f"{100*(1-ALPHA):.0f}% CI = [{lo:.6f}, {hi:.6f}], "
              f"runtime = {rt:.6f} s")

        rng = np.random.default_rng(SEED)
        price, se, rt, c_hat = control_variate_mc(N, rng)
        lo, hi = confidence_interval(price, se)
        print(f"(iii) control variate    : price = {price:.6f}, "
              f"{100*(1-ALPHA):.0f}% CI = [{lo:.6f}, {hi:.6f}], "
              f"runtime = {rt:.6f} s, c_hat = {c_hat:.4f}")

        print()
