"""
Exercise 2: Finite difference methods for the heat equation

    u_t(t,x) = u_xx(t,x),   t in [0, 0.1],   x in [0, pi]

    u(0,x) = 2x            for 0 < x <= pi/2
    u(0,x) = 2(pi - x)      for pi/2 < x < pi
    u(t,0) = u(t,pi) = 0

(i)  Explicit finite difference scheme (FTCS)
(ii) Implicit finite difference scheme (Backward Euler, solved with
     the Thomas algorithm for tridiagonal systems)

Each solver takes only Nx and Nt as input parameters.
"""

import numpy as np
import matplotlib.pyplot as plt

T_END = 0.1
X_END = np.pi


# ---------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------
def initial_condition(x):
    return np.where(x <= np.pi / 2, 2 * x, 2 * (np.pi - x))


# ---------------------------------------------------------------
# Exact (series) solution, given without proof in the exercise
# ---------------------------------------------------------------
def exact_solution(t, x, n_terms=2000):
    x = np.atleast_1d(x).astype(float)
    u = np.zeros_like(x)
    for i in range(1, n_terms + 1):
        k = 2 * i - 1
        u += (-1) ** (i - 1) * np.exp(-(k ** 2) * t) * np.sin(k * x) / k ** 2
    return 8.0 / np.pi * u


# ---------------------------------------------------------------
# Thomas algorithm for tridiagonal systems (used by implicit scheme)
# ---------------------------------------------------------------
def thomas_solve(a, b, c, d):
    """
    Solve a tridiagonal system with sub-diagonal a, diagonal b,
    super-diagonal c and right-hand side d. a[0] and c[-1] are unused.
    """
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
# (i) Explicit finite difference method (FTCS)
# ---------------------------------------------------------------
def explicit_fd(Nx, Nt):
    dx = X_END / Nx
    dt = T_END / Nt
    r = dt / dx ** 2

    x = np.linspace(0, X_END, Nx + 1)
    u = initial_condition(x)
    u[0] = 0.0
    u[-1] = 0.0

    for _ in range(Nt):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
        u_new[0] = 0.0
        u_new[-1] = 0.0
        u = u_new

    return x, u, r


# ---------------------------------------------------------------
# (ii) Implicit finite difference method (Backward Euler)
# ---------------------------------------------------------------
def implicit_fd(Nx, Nt):
    dx = X_END / Nx
    dt = T_END / Nt
    r = dt / dx ** 2

    x = np.linspace(0, X_END, Nx + 1)
    u = initial_condition(x)
    u[0] = 0.0
    u[-1] = 0.0

    # interior unknowns j = 1, ..., Nx-1
    n_int = Nx - 1
    a = np.full(n_int, -r)   # sub-diagonal
    b = np.full(n_int, 1 + 2 * r)  # diagonal
    c = np.full(n_int, -r)   # super-diagonal

    for _ in range(Nt):
        d = u[1:-1].copy()
        u_int = thomas_solve(a, b, c, d)
        u[1:-1] = u_int
        u[0] = 0.0
        u[-1] = 0.0

    return x, u, r


# ---------------------------------------------------------------
# Error at final time t = 0.1
# ---------------------------------------------------------------
def max_error(x, u_num):
    u_ex = exact_solution(T_END, x)
    err = np.abs(u_ex - u_num)
    return np.max(err), err, u_ex


# ---------------------------------------------------------------
# Run + plot for a given method
# ---------------------------------------------------------------
def run_and_plot(method, Nx, Nt, label):
    x, u_num, r = method(Nx, Nt)
    err_max, err, u_ex = max_error(x, u_num)

    print(f"{label}: Nx={Nx}, Nt={Nt}, r=dt/dx^2={r:.4f}, "
          f"max error at t=0.1: {err_max:.6e}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(x, u_num, 'o-', ms=3, label=f"numerical ({label})")
    axes[0].plot(x, u_ex, '--', label="exact (series)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(0.1, x)")
    axes[0].set_title(f"{label}: solution at t=0.1\nNx={Nx}, Nt={Nt}")
    axes[0].legend()

    axes[1].plot(x, err, color="tab:red")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("|u_exact - u_numerical|")
    axes[1].set_title(f"Error at t=0.1 (max = {err_max:.3e})")

    fig.tight_layout()
    fname = f"{label.lower()}_Nx{Nx}_Nt{Nt}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  -> plot saved to {fname}")
    return err_max


if __name__ == "__main__":
    test_cases = [(75, 108), (75, 150)]

    print("Explicit method:")
    for Nx, Nt in test_cases:
        run_and_plot(explicit_fd, Nx, Nt, "Explicit")

    print("\nImplicit method:")
    for Nx, Nt in test_cases:
        run_and_plot(implicit_fd, Nx, Nt, "Implicit")
