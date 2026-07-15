"""
Exercise 3: Crank-Nicolson method for the heat equation with discontinuous
initial data.

    u_t = u_xx,   x in [-1, 1],   t in [0, 1]

    u0(x) = 1     if |x| < 1/2
    u0(x) = 1/2   if |x| = 1/2
    u0(x) = 0     otherwise

    u(-1,t) = u(1,t) = 0

(i)  Pure Crank-Nicolson, dx = dt = 0.01.
(ii) Two steps of the (fully) implicit method, then Crank-Nicolson,
     dx = dt = 0.1.

Written observations (what happens near |x_j| ~ 0.5) are in the separate
file `observations.md`.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d projection)

X_LEFT, X_RIGHT = -1.0, 1.0
T_END = 1.0


# ---------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------
def initial_condition(x, tol=1e-9):
    ax = np.abs(x)
    u = np.zeros_like(x)
    u[ax < 0.5 - tol] = 1.0
    u[np.abs(ax - 0.5) <= tol] = 0.5
    return u


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
# One Crank-Nicolson step (theta = 1/2), homogeneous Dirichlet BC
# ---------------------------------------------------------------
def cn_step(u, r):
    n_int = len(u) - 2
    a = np.full(n_int, -r / 2)
    b = np.full(n_int, 1 + r)
    c = np.full(n_int, -r / 2)
    d = (r / 2) * u[:-2] + (1 - r) * u[1:-1] + (r / 2) * u[2:]
    u_int = thomas_solve(a, b, c, d)
    u_new = np.zeros_like(u)
    u_new[1:-1] = u_int
    return u_new  # u_new[0] = u_new[-1] = 0


# ---------------------------------------------------------------
# One fully implicit (backward Euler, theta = 1) step
# ---------------------------------------------------------------
def implicit_step(u, r):
    n_int = len(u) - 2
    a = np.full(n_int, -r)
    b = np.full(n_int, 1 + 2 * r)
    c = np.full(n_int, -r)
    d = u[1:-1]
    u_int = thomas_solve(a, b, c, d)
    u_new = np.zeros_like(u)
    u_new[1:-1] = u_int
    return u_new


# ---------------------------------------------------------------
# (i) Pure Crank-Nicolson solver, returns full space-time grid
# ---------------------------------------------------------------
def solve_cn(dx, dt, T=T_END, X=(X_LEFT, X_RIGHT)):
    Nx = int(round((X[1] - X[0]) / dx))
    Nt = int(round(T / dt))
    r = dt / dx ** 2

    x = np.linspace(X[0], X[1], Nx + 1)
    t = np.linspace(0, T, Nt + 1)

    U = np.zeros((Nt + 1, Nx + 1))
    U[0] = initial_condition(x)

    u = U[0].copy()
    for n in range(Nt):
        u = cn_step(u, r)
        U[n + 1] = u

    return x, t, U


# ---------------------------------------------------------------
# (ii) Two implicit steps followed by Crank-Nicolson
# ---------------------------------------------------------------
def solve_implicit_then_cn(dx, dt, n_implicit=2, T=T_END, X=(X_LEFT, X_RIGHT)):
    Nx = int(round((X[1] - X[0]) / dx))
    Nt = int(round(T / dt))
    r = dt / dx ** 2

    x = np.linspace(X[0], X[1], Nx + 1)
    t = np.linspace(0, T, Nt + 1)

    U = np.zeros((Nt + 1, Nx + 1))
    U[0] = initial_condition(x)

    u = U[0].copy()
    for n in range(Nt):
        if n < n_implicit:
            u = implicit_step(u, r)
        else:
            u = cn_step(u, r)
        U[n + 1] = u

    return x, t, U


# ---------------------------------------------------------------
# 3D surface plot helper
# ---------------------------------------------------------------
def plot_surface(x, t, U, title, fname):
    Xg, Tg = np.meshgrid(x, t)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Xg, Tg, U, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u(t,x)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  -> plot saved to {fname}")


if __name__ == "__main__":
    # -------------------------------------------------------
    # (i) Pure Crank-Nicolson, dx = dt = 0.01
    # -------------------------------------------------------
    print("(i) Pure Crank-Nicolson, dx = dt = 0.01")
    x1, t1, U1 = solve_cn(dx=0.01, dt=0.01)
    plot_surface(x1, t1, U1, "Crank-Nicolson (dx=dt=0.01)", "cn_dx001_dt001.png")

    # inspect behaviour near |x| = 0.5
    j_left = np.argmin(np.abs(x1 - (-0.5)))
    j_right = np.argmin(np.abs(x1 - 0.5))
    print(f"  u(t, x~-0.5) range: [{U1[:, j_left].min():.4f}, {U1[:, j_left].max():.4f}]")
    print(f"  u(t, x~ 0.5) range: [{U1[:, j_right].min():.4f}, {U1[:, j_right].max():.4f}]")
    print(f"  global min/max of U: [{U1.min():.4f}, {U1.max():.4f}] "
          f"(physically u should stay within [0, 1])")

    # -------------------------------------------------------
    # (ii) Two implicit steps + Crank-Nicolson, dx = dt = 0.1
    # -------------------------------------------------------
    print("\n(ii) Two implicit steps then Crank-Nicolson, dx = dt = 0.1")
    x2, t2, U2 = solve_implicit_then_cn(dx=0.1, dt=0.1, n_implicit=2)
    plot_surface(x2, t2, U2, "2x implicit + Crank-Nicolson (dx=dt=0.1)",
                 "implicit2_cn_dx01_dt01.png")

    print(f"  global min/max of U: [{U2.min():.4f}, {U2.max():.4f}] "
          f"(physically u should stay within [0, 1])")

    # for comparison: pure CN with the same coarse dx=dt=0.1
    print("\n(for comparison) pure Crank-Nicolson, dx = dt = 0.1")
    x3, t3, U3 = solve_cn(dx=0.1, dt=0.1)
    plot_surface(x3, t3, U3, "Pure Crank-Nicolson (dx=dt=0.1)", "cn_dx01_dt01.png")
    print(f"  global min/max of U: [{U3.min():.4f}, {U3.max():.4f}]")
