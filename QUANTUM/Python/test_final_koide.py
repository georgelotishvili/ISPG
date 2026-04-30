"""Final comprehensive Koide calculation.

1. Fine scan Phi0 near the threshold where electron mode (l=2) appears
2. Apply 4/pi geometric correction
3. Find optimal Phi0 for best Q + mass ratios
4. Report definitive numbers
"""
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

import numpy as np
from scipy.integrate import solve_bvp
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.interpolate import interp1d

ALPHA = 0.5
PI = np.pi

m_e, m_mu, m_tau = 0.51099895, 105.6583755, 1776.86
Q_target = (m_e+m_mu+m_tau)/(np.sqrt(m_e)+np.sqrt(m_mu)+np.sqrt(m_tau))**2


def koide(m1, m2, m3):
    return (m1+m2+m3) / (np.sqrt(m1)+np.sqrt(m2)+np.sqrt(m3))**2


def nl_func(Phi):
    return Phi * (1.0 - np.exp(-ALPHA * Phi))


def dnl_func(Phi):
    return (1.0 - np.exp(-ALPHA*Phi)) + ALPHA*Phi*np.exp(-ALPHA*Phi)


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
                    tol=1e-6, max_nodes=30000, verbose=0)
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


def E_1d(Om):
    k2 = 1.0 - Om**2
    if k2 <= 0: return 0.0
    return k2**1.5 * (4*Om**2 + 1)


print("=" * 70)
print("  FINAL KOIDE CALCULATION")
print("=" * 70)
print(f"  alpha = {ALPHA}")
print(f"  Target Q = {Q_target:.8f}")
print(f"  Target ratios: 1 : {m_mu/m_e:.1f} : {m_tau/m_e:.1f}")
print(f"  4/pi = {4/PI:.6f}")

# ===== Phase 1: Build continuation chain =====
print(f"\n--- Phase 1: Building oscillon solutions ---\n")

prev = None
solutions = {}
for Phi0 in np.arange(0.05, 5.01, 0.01):
    if prev:
        Om, sol = solve_osc(Phi0, r_prev=prev.x, y_prev=prev.y, p_prev=prev.p)
    else:
        Om, sol = solve_osc(Phi0)
    if Om is None:
        if prev is not None:
            print(f"  Continuation broke at Phi0={Phi0:.2f}")
            break
        continue
    prev = sol
    solutions[round(Phi0*100)/100] = (Om, sol)

print(f"  Built {len(solutions)} solutions")

# ===== Phase 2: Fine scan near electron threshold =====
print(f"\n--- Phase 2: Finding electron threshold (l=2 appearance) ---\n")

threshold_Phi0 = None
for Phi0_key in sorted(solutions.keys()):
    if Phi0_key < 1.5:
        continue
    Om, sol = solutions[Phi0_key]
    eigs_l2 = cavity_eigs(sol.x, sol.y[0], 2, N=2000)
    if len(eigs_l2) > 0 and eigs_l2[0] > 0.001:
        if threshold_Phi0 is None:
            threshold_Phi0 = Phi0_key
            print(f"  Electron mode APPEARS at Phi0 = {Phi0_key:.2f}"
                  f"  (Om_bg={Om:.4f}, om_e={eigs_l2[0]:.6f})")
        if Phi0_key < threshold_Phi0 + 0.05:
            print(f"    Phi0={Phi0_key:.2f}: om_e={eigs_l2[0]:.6f}")

print(f"\n  Threshold: Phi0 = {threshold_Phi0:.2f}")

# ===== Phase 3: Comprehensive scan with corrections =====
print(f"\n--- Phase 3: Koide scan (E_1D and 4/pi corrected) ---\n")
print(f"  {'Phi0':>5} {'Om_bg':>7} {'Q_raw':>8} {'Q_corr':>8}"
      f" {'mu/e_r':>7} {'tau/e_r':>8} {'mu/e_c':>7} {'tau/e_c':>8}"
      f" {'tm_raw':>7}")

best_raw_Q = (999, None)
best_corr_ratio = (999, None)
best_combined = (999, None)

results = []

for Phi0_key in sorted(solutions.keys()):
    if Phi0_key < 2.0 or Phi0_key > 4.0:
        continue
    Om, sol = solutions[Phi0_key]

    modes = {}
    for l_val in [0, 1, 2, 3]:
        eigs = cavity_eigs(sol.x, sol.y[0], l_val, N=2000)
        for n, om in enumerate(eigs):
            if om > 0.001:
                modes[(n, l_val)] = om

    om_tau = modes.get((0, 0))
    om_mu = modes.get((1, 0))
    om_e = modes.get((0, 2))

    if om_tau is None or om_mu is None or om_e is None:
        continue

    E_t = E_1d(om_tau)
    E_m = E_1d(om_mu)
    E_el = E_1d(om_e)

    if E_el < 1e-20:
        continue

    Q_raw = koide(E_t, E_m, E_el)
    mu_e_raw = E_m / E_el
    tau_e_raw = E_t / E_el
    tm_raw = E_t / E_m

    corr = 4.0 / PI
    mu_e_corr = mu_e_raw * corr
    tau_e_corr = tau_e_raw * corr
    m_e_c, m_mu_c, m_tau_c = 1.0, mu_e_corr, tau_e_corr
    Q_corr = koide(m_tau_c, m_mu_c, m_e_c)

    dQ_raw = abs(Q_raw - 2/3)
    err_mu = abs(mu_e_corr - m_mu/m_e) / (m_mu/m_e)
    err_tau = abs(tau_e_corr - m_tau/m_e) / (m_tau/m_e)
    err_ratio = err_mu + err_tau

    results.append({
        'Phi0': Phi0_key, 'Om': Om,
        'Q_raw': Q_raw, 'Q_corr': Q_corr,
        'mu_e_raw': mu_e_raw, 'tau_e_raw': tau_e_raw,
        'mu_e_corr': mu_e_corr, 'tau_e_corr': tau_e_corr,
        'tm_raw': tm_raw,
        'om_tau': om_tau, 'om_mu': om_mu, 'om_e': om_e,
        'dQ_raw': dQ_raw, 'err_ratio': err_ratio,
    })

    if dQ_raw < best_raw_Q[0]:
        best_raw_Q = (dQ_raw, results[-1])
    if err_ratio < best_corr_ratio[0]:
        best_corr_ratio = (err_ratio, results[-1])
    score = dQ_raw * 1000 + err_ratio
    if score < best_combined[0]:
        best_combined = (score, results[-1])

    if abs(Phi0_key - round(Phi0_key*5)/5) < 0.005 or Phi0_key == threshold_Phi0:
        print(f"  {Phi0_key:5.2f} {Om:7.4f} {Q_raw:8.5f} {Q_corr:8.5f}"
              f" {mu_e_raw:7.1f} {tau_e_raw:8.0f}"
              f" {mu_e_corr:7.1f} {tau_e_corr:8.0f} {tm_raw:7.2f}")

# ===== Phase 4: Best results =====
print(f"\n{'='*70}")
print(f"  DEFINITIVE RESULTS")
print(f"{'='*70}")

print(f"\n  --- Best raw Q (no correction) ---")
r = best_raw_Q[1]
print(f"  Phi0 = {r['Phi0']:.2f}, Om_bg = {r['Om']:.4f}")
print(f"  Modes: tau(0,0)={r['om_tau']:.4f}  mu(1,0)={r['om_mu']:.4f}"
      f"  e(0,2)={r['om_e']:.6f}")
print(f"  Q_raw = {r['Q_raw']:.8f}  |Q-2/3| = {r['dQ_raw']:.2e}")
print(f"  Ratios: 1 : {r['mu_e_raw']:.1f} : {r['tau_e_raw']:.0f}"
      f"  (tau/mu = {r['tm_raw']:.2f})")

print(f"\n  --- Best corrected mass ratios (x 4/pi) ---")
r = best_corr_ratio[1]
print(f"  Phi0 = {r['Phi0']:.2f}, Om_bg = {r['Om']:.4f}")
print(f"  Modes: tau(0,0)={r['om_tau']:.4f}  mu(1,0)={r['om_mu']:.4f}"
      f"  e(0,2)={r['om_e']:.6f}")
print(f"  Q_raw = {r['Q_raw']:.8f}")
print(f"  Corrected ratios: 1 : {r['mu_e_corr']:.1f} : {r['tau_e_corr']:.0f}")
print(f"  Target:           1 : {m_mu/m_e:.1f} : {m_tau/m_e:.0f}")
mu_err = abs(r['mu_e_corr'] - m_mu/m_e)/(m_mu/m_e)*100
tau_err = abs(r['tau_e_corr'] - m_tau/m_e)/(m_tau/m_e)*100
print(f"  mu/e error: {mu_err:.2f}%")
print(f"  tau/e error: {tau_err:.2f}%")

print(f"\n  --- Best combined (Q + ratios) ---")
r = best_combined[1]
print(f"  Phi0 = {r['Phi0']:.2f}, Om_bg = {r['Om']:.4f}")
print(f"  Q_raw = {r['Q_raw']:.8f}  |Q-2/3| = {r['dQ_raw']:.2e}")
print(f"  Corrected ratios: 1 : {r['mu_e_corr']:.1f} : {r['tau_e_corr']:.0f}")
print(f"  tau/mu = {r['tm_raw']:.2f}  (target {m_tau/m_mu:.2f})")

print(f"\n  --- Comparison table ---")
print(f"  {'':>20} {'Experiment':>12} {'Cavity':>12} {'Cavity+4/pi':>12}")
print(f"  {'Koide Q':>20} {Q_target:12.7f} {best_raw_Q[1]['Q_raw']:12.7f}"
      f" {best_raw_Q[1]['Q_corr']:12.7f}")
print(f"  {'|Q - 2/3|':>20} {'':>12}"
      f" {best_raw_Q[1]['dQ_raw']:12.2e} {'':>12}")
r_best = best_raw_Q[1]
print(f"  {'e : mu : tau':>20}"
      f" {'1:'+str(round(m_mu/m_e,1))+':'+str(round(m_tau/m_e)):>12}"
      f" {'1:'+str(round(r_best['mu_e_raw'],1))+':'+str(round(r_best['tau_e_raw'])):>12}"
      f" {'1:'+str(round(r_best['mu_e_corr'],1))+':'+str(round(r_best['tau_e_corr'])):>12}")
print(f"  {'tau/mu':>20} {m_tau/m_mu:12.2f} {r_best['tm_raw']:12.2f}")

print(f"\n  --- Physical interpretation ---")
print(f"  4 cavity modes = 3 leptons + 1 gravity:")
print(f"    tau   = (n=0, l=0) ground state       [heaviest]")
print(f"    muon  = (n=1, l=0) radial excitation   [middle]")
print(f"    e     = (n=0, l=2) angular mode         [lightest]")
print(f"    grav  = (n=0, l=1) translational mode   [= Om_bg, not a particle]")
print(f"  alpha=0.5 from ISPG theory (not tuned)")
print(f"  4/pi geometric correction (1D square -> 3D circle)")
