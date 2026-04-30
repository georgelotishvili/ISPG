"""Three cavity modes -> three leptons: Koide analysis.

Key discovery: the Phi^2 oscillon cavity has EXACTLY 3 bound states:
  Mode 1: (n=0, l=0) - ground state, smallest omega
  Mode 2: (n=0, l=1) - angular mode
  Mode 3: (n=1, l=0) - radial excited state, largest omega

This = 3 lepton generations? Check mass ratios and Koide Q.
Test multiple mass formulas:
  (A) m ~ omega           (paper formula)
  (B) m ~ E_1D(omega)     (1D oscillon energy)
  (C) m ~ E_3D(omega)     (3D oscillon energy)
  (D) m ~ (1-omega^2)^p   (power-law relation)
"""
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

import numpy as np
from scipy.integrate import solve_bvp
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar


def solve_bg_cont(Phi0_list, r_max=50.0, g=1.0):
    """Build background oscillons via continuation."""
    results = {}
    seed = Phi0_list[0]
    kappa_est = np.sqrt(min(seed / 4.2, 0.95))
    Om_guess = np.sqrt(max(0.01, 1.0 - kappa_est**2))

    def ode(r, y, p):
        Om = p[0]
        Phi, dPhi = y
        r_safe = np.maximum(r, 1e-8)
        d2 = -(2.0/r_safe)*dPhi - (Om**2 - 1)*Phi - g*Phi**2
        d2_0 = -(Om**2 - 1)*Phi/3.0 - g*Phi**2/3.0
        d2 = np.where(r < 1e-8, d2_0, d2)
        return np.vstack([dPhi, d2])

    def bc(ya, yb, p):
        return np.array([ya[0] - Phi0_list[0], ya[1], yb[0]])

    r = np.linspace(1e-6, r_max, 400)
    kappa_g = np.sqrt(max(0.01, 1.0 - Om_guess**2))
    Phi_init = seed / np.cosh(r * kappa_g)**2
    y_init = np.vstack([Phi_init, np.gradient(Phi_init, r)])

    sol = solve_bvp(ode, bc, r, y_init, p=[Om_guess],
                    tol=1e-6, max_nodes=20000, verbose=0)
    if not sol.success:
        return results

    results[seed] = (sol.p[0], sol)
    prev_r, prev_y, prev_p = sol.x, sol.y, sol.p

    for Phi0 in Phi0_list[1:]:
        def bc2(ya, yb, p):
            return np.array([ya[0] - Phi0, ya[1], yb[0]])

        r_new = np.linspace(1e-6, r_max, max(400, len(prev_r)))
        f0 = interp1d(prev_r, prev_y[0], fill_value=0.0, bounds_error=False)
        f1 = interp1d(prev_r, prev_y[1], fill_value=0.0, bounds_error=False)
        scale = Phi0 / max(abs(prev_y[0][0]), 1e-30)
        y_new = np.vstack([f0(r_new)*scale, f1(r_new)*scale])

        sol = solve_bvp(ode, bc2, r_new, y_new, p=prev_p,
                        tol=1e-6, max_nodes=20000, verbose=0)
        if sol.success and 0.01 < sol.p[0] < 0.999:
            results[Phi0] = (sol.p[0], sol)
            prev_r, prev_y, prev_p = sol.x, sol.y, sol.p
        else:
            break

    return results


def cavity_eigs(r_bg, Phi_bg, l_val, g=1.0, N=2000):
    """Eigenvalues of linearized equation."""
    r_max = r_bg[-1] * 0.9
    dr = r_max / (N + 1)
    r = np.linspace(dr, r_max - dr, N)
    f = interp1d(r_bg, Phi_bg, fill_value=0.0, bounds_error=False)
    Phi = f(r)

    V = 1.0 - 2.0*g*Phi + l_val*(l_val+1)/r**2
    H = diags([-np.ones(N-1)/dr**2, 2.0/dr**2 + V, -np.ones(N-1)/dr**2],
              [-1, 0, 1], format='csc')

    n_eig = min(15, N-2)
    evals, _ = eigsh(H, k=n_eig, which='SM')
    return np.sqrt(np.maximum(np.sort(evals[evals < 1.0]), 0))


def E_1d(Om):
    """1D oscillon energy."""
    k2 = 1.0 - Om**2
    if k2 <= 0:
        return 0.0
    return k2**1.5 * (4*Om**2 + 1)


def koide(m1, m2, m3):
    return (m1+m2+m3) / (np.sqrt(m1)+np.sqrt(m2)+np.sqrt(m3))**2


print("=" * 70)
print("  3 Cavity Modes = 3 Leptons: Koide Check")
print("=" * 70)

m_e, m_mu, m_tau = 0.51099895, 105.6583755, 1776.86
Q_target = 2.0/3.0
print(f"  Target: m_e={m_e}, m_mu={m_mu}, m_tau={m_tau}")
print(f"  Koide Q = {koide(m_e, m_mu, m_tau):.8f}")
print(f"  Ratios: 1 : {m_mu/m_e:.1f} : {m_tau/m_e:.1f}\n")

Phi0_vals = list(np.arange(0.10, 4.01, 0.02))
bg = solve_bg_cont(Phi0_vals)
print(f"  Background solutions: {len(bg)} (Phi0 = {min(bg):.2f}..{max(bg):.2f})\n")

print("=" * 70)
print("  Scan: 3 cavity modes for each Phi0")
print("=" * 70)

best_by_formula = {
    'omega': (999, None),
    'E_1d': (999, None),
    'kappa_pow': (999, None),
}

print(f"\n  {'Phi0':>5} | {'om_00':>7} {'om_01':>7} {'om_10':>7}"
      f" | {'Q(omega)':>10} {'Q(E_1d)':>10}"
      f" | {'ratio(E)':>22}")

for Phi0 in sorted(bg.keys()):
    Om_bg, sol_bg = bg[Phi0]

    omegas_l0 = cavity_eigs(sol_bg.x, sol_bg.y[0], l_val=0, N=2000)
    omegas_l1 = cavity_eigs(sol_bg.x, sol_bg.y[0], l_val=1, N=2000)

    if len(omegas_l0) < 2 or len(omegas_l1) < 1:
        continue

    om_00 = omegas_l0[0]
    om_10 = omegas_l0[1]
    om_01 = omegas_l1[0]

    if om_00 < 0.001 or om_01 < 0.001 or om_10 < 0.001:
        continue

    modes = sorted([om_00, om_01, om_10])

    Q_om = koide(*modes)
    diff_om = abs(Q_om - Q_target)
    if diff_om < best_by_formula['omega'][0]:
        best_by_formula['omega'] = (diff_om, (Phi0, modes, Q_om, 'omega'))

    E_modes = sorted([E_1d(m) for m in modes], reverse=True)
    if E_modes[-1] > 1e-10:
        Q_E = koide(*E_modes)
        diff_E = abs(Q_E - Q_target)
        if diff_E < best_by_formula['E_1d'][0]:
            best_by_formula['E_1d'] = (diff_E, (Phi0, E_modes, Q_E, 'E_1d'))

        kappas = sorted([np.sqrt(1-m**2) for m in modes], reverse=True)
        for p in np.arange(0.5, 4.0, 0.01):
            m_kp = [k**p for k in kappas]
            if m_kp[-1] > 1e-15:
                Q_kp = koide(*m_kp)
                diff_kp = abs(Q_kp - Q_target)
                if diff_kp < best_by_formula['kappa_pow'][0]:
                    best_by_formula['kappa_pow'] = (diff_kp, (Phi0, kappas, Q_kp, f'kappa^p, p={p:.2f}', m_kp))

    if abs(Phi0 - round(Phi0, 1)) < 0.01:
        r_str = f"1:{E_modes[1]/E_modes[2]:.1f}:{E_modes[0]/E_modes[2]:.1f}" if E_modes[-1] > 1e-10 else "---"
        print(f"  {Phi0:5.2f} | {om_00:7.4f} {om_01:7.4f} {om_10:7.4f}"
              f" | {Q_om:10.6f} {Q_E if E_modes[-1] > 1e-10 else 0:10.6f}"
              f" | {r_str:>22}")

print("\n" + "=" * 70)
print("  BEST RESULTS")
print("=" * 70)

for name, (diff, data) in sorted(best_by_formula.items(), key=lambda x: x[1][0]):
    if data is None:
        continue
    print(f"\n  Formula: {name}")
    print(f"  |Q - 2/3| = {diff:.2e}")
    if name == 'omega':
        Phi0, modes, Q, _ = data
        print(f"  Phi0 = {Phi0:.2f}")
        print(f"  Q = {Q:.10f}")
        print(f"  omegas: {modes}")
        print(f"  mass ratios: 1 : {modes[1]/modes[0]:.4f} : {modes[2]/modes[0]:.4f}")
    elif name == 'E_1d':
        Phi0, modes, Q, _ = data
        print(f"  Phi0 = {Phi0:.2f}")
        print(f"  Q = {Q:.10f}")
        print(f"  E values: {modes}")
        print(f"  mass ratios: 1 : {modes[1]/modes[2]:.2f} : {modes[0]/modes[2]:.2f}")
    elif name == 'kappa_pow':
        Phi0, kappas, Q, label, m_kp = data
        print(f"  Phi0 = {Phi0:.2f}, {label}")
        print(f"  Q = {Q:.10f}")
        print(f"  kappas: {kappas}")
        print(f"  masses: {m_kp}")
        r0 = min(m_kp)
        print(f"  mass ratios: 1 : {sorted(m_kp)[1]/r0:.2f} : {sorted(m_kp)[2]/r0:.2f}")

print(f"\n  Target mass ratios: 1 : {m_mu/m_e:.1f} : {m_tau/m_e:.1f}")

print("\n" + "=" * 70)
print("  KEY QUESTION: does the Poschl-Teller structure")
print("  predict Q = 2/3 analytically?")
print("=" * 70)

print("\n  Poschl-Teller potential from sech^2 oscillon:")
print("  V(r) = -2*Phi0 * sech^2(kappa*r), where kappa = sqrt(1-Om^2)/2")
print("  lambda*(lambda-1) = 2*Phi0/kappa^2")
print("  For 1D oscillon: Phi0 = 3(1-Om^2)/2, kappa = sqrt(1-Om^2)/2")
print("  -> lambda*(lambda-1) = 12, lambda = 4")
print("  -> Eigenvalues: E_n = -kappa^2*(3-n)^2 for n=0,1,2")
print("  -> omega_n^2 = 1 - kappa^2*(3-n)^2")
print()

for Om_bg in np.arange(0.75, 0.99, 0.02):
    k2 = 1.0 - Om_bg**2
    kappa = np.sqrt(k2) / 2.0
    E_vals = [-kappa**2 * (3 - n)**2 for n in range(3)]
    om_vals = [np.sqrt(max(1.0 + E, 0)) for E in E_vals]
    if all(o > 0.001 for o in om_vals):
        Q = koide(*om_vals)
        E_masses = sorted([E_1d(o) for o in om_vals], reverse=True)
        if E_masses[-1] > 1e-10:
            Q_E = koide(*E_masses)
        else:
            Q_E = -1
        print(f"  Om_bg={Om_bg:.2f}: omegas={[f'{o:.4f}' for o in om_vals]}"
              f"  Q(omega)={Q:.6f}  Q(E_1d)={Q_E:.6f}"
              f"  ratios(E)=1:{E_masses[1]/E_masses[2]:.1f}:{E_masses[0]/E_masses[2]:.1f}")
