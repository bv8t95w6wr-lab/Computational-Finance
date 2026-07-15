"""
Exercise 1: Crank-Nicolson finite difference method for a European put option.

PDE (Black-Scholes):
    V_t + 0.5*sigma^2*S^2*V_SS + r*S*V_S - r*V = 0,   t in [0,T]
    V(T,S) = max(K-S, 0)                              (terminal condition)
    V(t,0) = K*exp(-r*(T-t)),   V(t,Smax) = 0          (boundary conditions)

We substitute tau = T - t (time to maturity) so the terminal-value problem
becomes an initial-value problem in tau in [0,T], and discretize with
Crank-Nicolson on S in [Smin, Smax].
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from math import log, sqrt, exp
from math import erf

# ---------------------------------------------------------------
# Option / market parameters
# ---------------------------------------------------------------
K = 110.0
T = 1.0
r = 0.02
sigma = 0.2

S_MIN = 0.0
S_MAX = 200.0


# ---------------------------------------------------------------
# Thomas algorithm for tridiagonal systems
# ---------------------------------------------------------------
def thomas_solve(a, b, c, d):
    """Solve tridiagonal system: sub-diag a, diag b, super-diag c, rhs d."""
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


# ---------------------------------------------------------------
# Crank-Nicolson solver for the European put
# ---------------------------------------------------------------
def crank_nicolson_put(Nt, Ns, K=K, T=T, r=r, sigma=sigma,
                        S_min=S_MIN, S_max=S_MAX):
    dS = (S_max - S_min) / Ns
    dtau = T / Nt

    S = S_min + dS * np.arange(Ns + 1)
    j = np.arange(Ns + 1)

    # coefficients of du_j/dtau = a_j u_{j-1} + b_j u_j + c_j u_{j+1}
    a = 0.5 * sigma ** 2 * j ** 2 - 0.5 * r * j
    b = -sigma ** 2 * j ** 2 - r
    c = 0.5 * sigma ** 2 * j ** 2 + 0.5 * r * j

    A = 0.5 * dtau * a
    B = 0.5 * dtau * b
    C = 0.5 * dtau * c

    # interior indices j = 1, ..., Ns-1
    sub = -A[1:-1]          # sub-diagonal of implicit matrix
    diag = 1 - B[1:-1]      # diagonal
    sup = -C[1:-1]          # super-diagonal

    # initial condition at tau = 0 (i.e. t = T): payoff
    u = np.maximum(K - S, 0.0)

    for n in range(Nt):
        tau_n = n * dtau
        tau_np1 = (n + 1) * dtau

        # explicit (known) part of the RHS, interior points
        rhs = A[1:-1] * u[:-2] + (1 + B[1:-1]) * u[1:-1] + C[1:-1] * u[2:]

        # boundary values at old and new time level
        u0_n, uNs_n = K * exp(-r * tau_n), 0.0
        u0_np1, uNs_np1 = K * exp(-r * tau_np1), 0.0

        # move known boundary contributions to the RHS
        rhs[0] += A[1] * u0_np1
        rhs[-1] += C[-2] * uNs_np1

        u_int = thomas_solve(sub, diag, sup, rhs)

        u = np.empty(Ns + 1)
        u[0] = u0_np1
        u[-1] = uNs_np1
        u[1:-1] = u_int

    return S, u  # u = V(0, S) since tau = T at the end


# ---------------------------------------------------------------
# Exact Black-Scholes put price
# ---------------------------------------------------------------
def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_put_price(S0, K=K, T=T, r=r, sigma=sigma):
    if S0 <= 0:
        return K * exp(-r * T)
    d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return K * exp(-r * T) * norm_cdf(-d2) - S0 * norm_cdf(-d1)


if __name__ == "__main__":
    # -------------------------------------------------------
    # (i) Run Crank-Nicolson for Nt = 100, Ns = 100, time it
    # -------------------------------------------------------
    Nt, Ns = 100, 100

    t0 = time.perf_counter()
    S_grid, V_cn = crank_nicolson_put(Nt, Ns)
    t1 = time.perf_counter()
    print(f"(i) Crank-Nicolson (Nt={Nt}, Ns={Ns}) runtime: {t1 - t0:.6f} s")

    # -------------------------------------------------------
    # (ii) Exact Black-Scholes prices for S0 = 0, 2, ..., 200
    # -------------------------------------------------------
    S0_values = np.arange(0, 202, 2, dtype=float)

    t0 = time.perf_counter()
    V_exact = np.array([bs_put_price(S0) for S0 in S0_values])
    t1 = time.perf_counter()
    print(f"(ii) Exact Black-Scholes pricing runtime: {t1 - t0:.6f} s")

    # -------------------------------------------------------
    # (iii) Plot V_cn(S0) and V(S0) for S0 in {2,4,...,200}; compute error
    # -------------------------------------------------------
    mask = S0_values >= 2  # exclude S0 = 0
    S_plot = S0_values[mask]
    V_exact_plot = V_exact[mask]

    # match S_grid to S_plot by index (S_grid coincides with S0_values since dS = 2)
    idx = np.searchsorted(S_grid, S_plot)
    V_cn_plot = V_cn[idx]

    err = np.max(np.abs(V_cn_plot - V_exact_plot))
    print(f"(iii) max error over S in {{2,4,...,200}}: {err:.6e}")

    plt.figure(figsize=(7, 5))
    plt.plot(S_plot, V_cn_plot, 'o-', ms=3, label="Crank-Nicolson $V^{cn}(S_0)$")
    plt.plot(S_plot, V_exact_plot, '--', label="Exact Black-Scholes $V(S_0)$")
    plt.xlabel("$S_0$")
    plt.ylabel("Put price")
    plt.title(f"European put: Crank-Nicolson vs. exact (max err = {err:.3e})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("crank_nicolson_put.png", dpi=150)
    print("Plot saved to crank_nicolson_put.png")
