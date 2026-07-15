"""
Exercise 4: Implied volatility (Black-Scholes) + volatility smile regression.

(i)  Compute sigma_imp(K) for European call prices via Newton's method
     (with bisection fallback), i.e. root-finding for
         BS(sigma; S, r, T, K) - Price(S, r, T, K) = 0
(ii) Fit sigma_imp(K) with a degree-2 and a degree-4 polynomial.
"""

import math
import numpy as np

# ---------------------------------------------------------------
# Market parameters
# ---------------------------------------------------------------
T = 0.21
S = 5290.4
r = 0.033

K = np.array([5400, 5500, 5600, 5700, 5800, 5900, 6000, 6100, 6150,
              6200, 6250, 6300, 6350, 6400, 6600, 6800, 7000, 7200], dtype=float)

Price = np.array([286.3, 239.4, 198.7, 160.7, 129.1, 102.4, 80.2, 61.7, 53.8,
                   47.1, 41.0, 35.9, 31.1, 27.7, 16.5, 11.4, 7.4, 5.7], dtype=float)


# ---------------------------------------------------------------
# Black-Scholes call price and vega
# ---------------------------------------------------------------
def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_call_price(sigma, S, r, T, K):
    if sigma <= 0:
        return max(S - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def bs_vega(sigma, S, r, T, K):
    if sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return S * norm_pdf(d1) * math.sqrt(T)


# ---------------------------------------------------------------
# Root finding: Newton-Raphson with bisection fallback
# ---------------------------------------------------------------
def implied_vol(price, S, r, T, K, sigma0=0.2, tol=1e-8, max_iter=100):
    sigma = sigma0

    # Newton-Raphson
    for _ in range(max_iter):
        f = bs_call_price(sigma, S, r, T, K) - price
        if abs(f) < tol:
            return sigma
        vega = bs_vega(sigma, S, r, T, K)
        if vega < 1e-12:
            break
        sigma_new = sigma - f / vega
        if sigma_new <= 0:
            break
        if abs(sigma_new - sigma) < tol:
            return sigma_new
        sigma = sigma_new

    # Bisection fallback (robust, guaranteed to converge)
    lo, hi = 1e-6, 5.0
    f_lo = bs_call_price(lo, S, r, T, K) - price
    f_hi = bs_call_price(hi, S, r, T, K) - price
    if f_lo * f_hi > 0:
        raise ValueError(f"No sign change for K={K}: check price/arbitrage bounds.")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = bs_call_price(mid, S, r, T, K) - price
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------
# (i) implied volatilities
# ---------------------------------------------------------------
sigma_imp = np.array([implied_vol(p, S, r, T, k) for p, k in zip(Price, K)])

print("K       Price     sigma_imp")
for k, p, s in zip(K, Price, sigma_imp):
    print(f"{k:6.0f}  {p:8.2f}   {s:.6f}")

# ---------------------------------------------------------------
# (ii) polynomial regression of sigma_imp(K)
# ---------------------------------------------------------------
# Degree 2: sigma_imp(K) = a2*K^2 + a1*K + a0
coeffs2 = np.polyfit(K, sigma_imp, 2)   # returns [a2, a1, a0]
a2, a1, a0 = coeffs2

# Degree 4: sigma_imp(K) = a4*K^4 + a3*K^3 + a2*K^2 + a1*K + a0
coeffs4 = np.polyfit(K, sigma_imp, 4)   # returns [a4, a3, a2, a1, a0]
b4, b3, b2, b1, b0 = coeffs4

print("\nDegree-2 fit:  a2={:.6e}, a1={:.6e}, a0={:.6e}".format(a2, a1, a0))
print("Degree-4 fit:  a4={:.6e}, a3={:.6e}, a2={:.6e}, a1={:.6e}, a0={:.6e}"
      .format(b4, b3, b2, b1, b0))

# ---------------------------------------------------------------
# Optional: plot the smile with both fits
# ---------------------------------------------------------------
if __name__ == "__main__":
    try:
        import matplotlib.pyplot as plt

        Kfine = np.linspace(K.min(), K.max(), 400)
        fit2 = np.polyval(coeffs2, Kfine)
        fit4 = np.polyval(coeffs4, Kfine)

        plt.figure(figsize=(8, 5))
        plt.scatter(K, sigma_imp, color="black", label="implied vol (data)")
        plt.plot(Kfine, fit2, label="degree-2 fit", color="tab:blue")
        plt.plot(Kfine, fit4, label="degree-4 fit", color="tab:red")
        plt.xlabel("Strike K")
        plt.ylabel(r"$\sigma_{imp}$")
        plt.title("Volatility smile")
        plt.legend()
        plt.tight_layout()
        plt.savefig("volatility_smile.png", dpi=150)
        print("\nPlot saved to volatility_smile.png")
    except ImportError:
        print("\n(matplotlib not installed - skipping plot)")
