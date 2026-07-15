# Exercise 3 — Observations

## (i) Pure Crank-Nicolson, dx = dt = 0.01

Here r = dt/dx² = 100. Away from the jump the solution behaves as expected
(smooth decay toward the steady state u ≡ 0). But for |x_j| ≈ 0.5, i.e. at
the grid points immediately next to the discontinuity of u0, the numerical
solution shows strong non-physical oscillations for the first several time
steps: values alternate between roughly 0.13 and 0.87 from one time step to
the next (see printed output), even though the true solution decays
monotonically there. Exactly at x = 0.5 itself the solution stays smooth
(it starts at the "correct" jump-midpoint value 0.5 and decays slowly).

This is the well-known oscillation (Gibbs-type) defect of the
Crank-Nicolson / theta = 1/2 scheme for discontinuous initial data: CN is
A-stable but not L-stable, so high-frequency error modes are not damped
(amplification factor tends to -1 as the mode number k → π), only
alternated in sign. With a large mesh ratio r = dt/dx² = 100 these
high-frequency components introduced by the jump in u0 are excited
strongly and decay only very slowly in an oscillatory manner, producing
visible "ringing" near x = ±0.5 in the surface plot
(cn_dx001_dt001.png).

## (ii) Two implicit steps + Crank-Nicolson, dx = dt = 0.1

For comparison, pure Crank-Nicolson at the coarser grid dx = dt = 0.1
(r = 10) shows the same qualitative oscillatory behaviour near the jump
(see cn_dx01_dt01.png, and the printed values at x = 0.4 / x = 0.6, which
alternate rather than decay monotonically).

Once the first two time steps are computed with the fully implicit
(backward Euler, theta = 1) scheme instead, and only the remaining steps
use Crank-Nicolson, the oscillations disappear
(implicit2_cn_dx01_dt01.png): the solution decays smoothly and
monotonically at every grid point, including next to the discontinuity.

This is because the implicit Euler scheme is L-stable: its amplification
factor tends to 0 for high-frequency modes, so it strongly damps the
high-frequency error introduced by the jump discontinuity within the
first couple of steps. Once those problematic modes are removed, the
subsequent Crank-Nicolson steps (which are second-order accurate in time
but not L-stable) no longer have anything left to oscillate and the
scheme behaves well. This "start with a few implicit/L-stable steps,
then switch to Crank-Nicolson" strategy is a standard fix for
non-smooth initial/terminal data and is known as Rannacher time-stepping.
