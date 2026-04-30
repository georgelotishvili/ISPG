"""Test π/4 geometric correction for E_1D mass formula.

User's insight: the 23% discrepancy might be exactly π/4 (circle-in-square ratio).
E_1D treats the oscillon as "square" (1D) when it's actually "round" (3D).

Test: what exponent p in E(Ω) = (1-Ω²)^p × (4Ω²+1) gives the best match?
Does it relate to π?
"""
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

import numpy as np
from scipy.integrate import solve_bvp
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.interpolate import interp1d

ALPHA = 0.5

m_e, m_mu, m_tau = 0.51099895, 105.6583755, 1776.86


def koide(m1, m2, m3):
    return (m1+m2+m3) / (np.sqrt(m1)+np.sqrt(m2)+np.sqrt(m3))**2


def dnl_func(Phi):
    return (1.0 - np.exp(-ALPHA*Phi)) + ALPHA*Phi*np.exp(-ALPHA*Phi)


def nl_func(Phi):
    return Phi * (1.0 - np.exp(-ALPHA * Phi))


def solve_osc(Phi0, r_max=60.0, r_prev=None, y_prev=None, p_prev=None):
    Om_guess = np.sqrt(max(0.01, 1.0 - min(Phi0/4.2, 0.95)))

    def ode(r, y, p):
        Om = p[0]
        Phi, dPhi = y
        r_safe = np.maximum(r, 1e-8)
        NL = nl_func(Phi)
        d2 = -(2.0/r_safe)*dPhi - (Om**2 - 1)*Phi - NL
        d2_0 = -(Om**2 - 1)*Phi/3.0 - NL/3.0
        d2 = np.where(r < 1e-8, d2_0, d2)
        return np.vstack([dPhi, d2])

    def bc(ya, yb, p):
        return np.array([ya[0] - Phi0, ya[1], yb[0]])

    if r_prev is not None:
        N = max(500, len(r_prev))
        r = np.linspace(1e-6, r_max, N)
        f0 = interp1d(r_prev, y_prev[0], fill_value=0.0, bounds_error=False)
        f1 = interp1d(r_prev, y_prev[1], fill_value=0.0, bounds_error=False)
        sc = Phi0 / max(abs(y_prev[0][0]), 1e-30)
        y_init = np.vstack([f0(r)*sc, f1(r)*sc])
        Om_guess = p_prev[0]
    else:
        r = np.linspace(1e-6, r_max, 500)
        kg = np.sqrt(max(0.01, 1.0 - Om_guess**2))
        Phi_init = Phi0 / np.cosh(r * kg)**2
        y_init = np.vstack([Phi_init, np.gradient(Phi_init, r)])

    sol = solve_bvp(ode, bc, r, y_init, p=[Om_guess],
                    tol=1e-5, max_nodes=30000, verbose=0)
    if sol.success and 0.01 < sol.p[0] < 0.999:
        return sol.p[0], sol
    return None, None


def cavity_eigs(r_bg, Phi_bg, l_val, N=2000):
    r_max = r_bg[-1] * 0.9
    dr = r_max / (N + 1)
    r = np.linspace(dr, r_max - dr, N)
    f = interp1d(r_bg, Phi_bg, fill_value=0.0, bounds_error=False)
    Phi = f(r)
    c_lin = dnl_func(Phi)
    V = 1.0 - c_lin + l_val*(l_val+1)/r**2
    H = diags([-np.ones(N-1)/dr**2, 2.0/dr**2 + V, -np.ones(N-1)/dr**2],
              [-1, 0, 1], format='csc')
    evals, _ = eigsh(H, k=min(20, N-2), which='SM')
    bound = evals[evals < 1.0]
    return np.sqrt(np.maximum(np.sort(bound), 0))


def E_p(Om, p):
    """Generalized energy: (1-Om^2)^p * (4Om^2+1)"""
    k2 = 1.0 - Om**2
    if k2 <= 0: return 0.0
    return k2**p * (4*Om**2 + 1)


print("=" * 70)
print("  π/4 Correction Test")
print("=" * 70)

print("\n--- Getting cavity eigenvalues at Phi0=2.35 ---\n")

prev = None
for Phi0 in np.arange(0.05, 2.40, 0.05):
    if prev:
        Om, sol = solve_osc(Phi0, r_prev=prev.x, y_prev=prev.y, p_prev=prev.p)
    else:
        Om, sol = solve_osc(Phi0)
    if Om:
        prev = sol

Om_bg, sol_bg = solve_osc(2.35, r_prev=prev.x, y_prev=prev.y, p_prev=prev.p)
print(f"  Background: Phi0=2.35, Om={Om_bg:.6f}")

modes = {}
for l_val in range(4):
    eigs = cavity_eigs(sol_bg.x, sol_bg.y[0], l_val, N=2000)
    for n, om in enumerate(eigs):
        if om > 0.001:
            modes[(n, l_val)] = om

om_tau = modes[(0, 0)]
om_mu = modes[(1, 0)]
om_e = modes[(0, 2)]

print(f"  tau (0,0): omega = {om_tau:.6f}")
print(f"  muon (1,0): omega = {om_mu:.6f}")
print(f"  electron (0,2): omega = {om_e:.6f}")

print(f"\n--- Scan exponent p in (1-Om^2)^p * (4Om^2+1) ---\n")
print(f"  {'p':>8} {'Q':>10} {'|Q-2/3|':>10} {'mu/e':>8} {'tau/e':>8}"
      f" {'tau/mu':>8} {'notes':>20}")
print(f"  " + "-" * 80)

pi = np.pi
special_p = {
    1.0: "p=1",
    1.5: "E_1D (3/2)",
    3*pi/8: f"3pi/8={3*pi/8:.4f}",
    pi/2: f"pi/2={pi/2:.4f}",
    2.0: "p=2",
    6/pi: f"6/pi={6/pi:.4f}",
    3/(2*pi/4): f"3/(2*pi/4)={3/(2*pi/4):.4f}",
}

best_Q = (999, None)
best_ratio = (999, None)
best_both = (999, None)

for p in np.arange(1.0, 2.01, 0.01):
    E_t = E_p(om_tau, p)
    E_m = E_p(om_mu, p)
    E_el = E_p(om_e, p)

    if E_el <= 0:
        continue

    Q = koide(E_t, E_m, E_el)
    ratio_mu = E_m / E_el
    ratio_tau = E_t / E_el
    ratio_tm = E_t / E_m

    note = ""
    for sp, sn in special_p.items():
        if abs(p - sp) < 0.005:
            note = sn

    dQ = abs(Q - 2/3)
    d_ratio = abs(ratio_tau/m_tau*m_e - 1) + abs(ratio_mu/m_mu*m_e - 1)

    if dQ < best_Q[0]:
        best_Q = (dQ, p)
    if d_ratio < best_ratio[0]:
        best_ratio = (d_ratio, p)
    score = dQ + 0.01*d_ratio
    if score < best_both[0]:
        best_both = (score, p)

    if note or abs(p - round(p*10)/10) < 0.005:
        print(f"  {p:8.4f} {Q:10.6f} {dQ:10.2e} {ratio_mu:8.1f}"
              f" {ratio_tau:8.0f} {ratio_tm:8.2f} {note:>20}")

print(f"\n  Target:    Q={koide(m_e,m_mu,m_tau):.6f}"
      f"        mu/e={m_mu/m_e:.1f} tau/e={m_tau/m_e:.0f}"
      f" tau/mu={m_tau/m_mu:.2f}")

print(f"\n--- Best results ---\n")
print(f"  Best Q:     p = {best_Q[1]:.4f}  |Q-2/3| = {best_Q[0]:.2e}")
print(f"  Best ratio: p = {best_ratio[1]:.4f}")
print(f"  Best both:  p = {best_both[1]:.4f}")

print(f"\n--- π-related exponents ---\n")
print(f"  3/2         = {1.5:.6f}  (E_1D)")
print(f"  3π/8        = {3*pi/8:.6f}")
print(f"  π/2         = {pi/2:.6f}")
print(f"  6/π         = {6/pi:.6f}")
print(f"  3/(π/2)     = {3/(pi/2):.6f}")
print(f"  3·(4/π)/8   = {3*4/(pi*8):.6f}")

print(f"\n--- Direct test: multiply ratios by 4/π ---\n")

E_t = E_p(om_tau, 1.5)
E_m = E_p(om_mu, 1.5)
E_el = E_p(om_e, 1.5)
print(f"  E_1D (p=3/2):")
print(f"    mu/e = {E_m/E_el:.1f}  tau/e = {E_t/E_el:.0f}  Q = {koide(E_t,E_m,E_el):.8f}")

scale = 4/pi
print(f"\n  Multiply ratios by 4/π = {scale:.6f}:")
ratio_mu_corr = (E_m/E_el) * scale
ratio_tau_corr = (E_t/E_el) * scale
m_e_c = 1.0
m_mu_c = ratio_mu_corr
m_tau_c = ratio_tau_corr
Q_corr = koide(m_tau_c, m_mu_c, m_e_c)
print(f"    mu/e = {ratio_mu_corr:.1f}  tau/e = {ratio_tau_corr:.0f}"
      f"  Q = {Q_corr:.8f}")
print(f"    Target: mu/e = {m_mu/m_e:.1f}  tau/e = {m_tau/m_e:.0f}"
      f"  Q = {koide(m_e,m_mu,m_tau):.8f}")
print(f"    mu/e error: {abs(ratio_mu_corr - m_mu/m_e)/(m_mu/m_e)*100:.2f}%")
print(f"    tau/e error: {abs(ratio_tau_corr - m_tau/m_e)/(m_tau/m_e)*100:.2f}%")
