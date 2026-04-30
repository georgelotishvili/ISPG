"""Investigation of the 4th cavity mode (l=1).

At alpha=0.5, Phi0=2.35, the cavity has 4 bound states:
  Mode 1: (n=0, l=0) -> tau      (omega=0.664, E_1d=1.16)
  Mode 2: (n=0, l=1) -> ???      (omega=0.866, E_1d=0.39)
  Mode 3: (n=1, l=0) -> muon     (omega=0.970, E_1d=0.067)
  Mode 4: (n=0, l=2) -> electron (omega=0.999, E_1d=0.00042)

The l=1 mode has intermediate mass between tau and muon.
What particle could this be?

This script investigates:
  1. The l=1 mode properties (mass, width, quantum numbers)
  2. Whether there are known particles at the predicted mass
  3. The full 4-mode spectrum and its Koide-like properties
  4. When does the 4th mode appear/disappear as alpha changes?
"""
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

import numpy as np
from scipy.integrate import solve_bvp
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.interpolate import interp1d
from itertools import combinations

ALPHA = 0.5


def nl_func(Phi):
    return Phi * (1.0 - np.exp(-ALPHA * Phi))


def dnl_func(Phi):
    return (1.0 - np.exp(-ALPHA*Phi)) + ALPHA*Phi*np.exp(-ALPHA*Phi)


def solve_osc(Phi0, r_max=60.0, Om_guess=None,
              r_prev=None, y_prev=None, p_prev=None):
    if Om_guess is None:
        kappa_est = np.sqrt(min(Phi0 / 4.2, 0.95))
        Om_guess = np.sqrt(max(0.01, 1.0 - kappa_est**2))

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
        N_pts = max(500, len(r_prev))
        r = np.linspace(1e-6, r_max, N_pts)
        f0 = interp1d(r_prev, y_prev[0], fill_value=0.0, bounds_error=False)
        f1 = interp1d(r_prev, y_prev[1], fill_value=0.0, bounds_error=False)
        sc = Phi0 / max(abs(y_prev[0][0]), 1e-30)
        y_init = np.vstack([f0(r)*sc, f1(r)*sc])
        Om_guess = p_prev[0] if p_prev is not None else Om_guess
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
    evals, evecs = eigsh(H, k=min(20, N-2), which='SM')
    bound = evals < 1.0
    return np.sqrt(np.maximum(np.sort(evals[bound]), 0))


def E_1d(Om):
    k2 = 1.0 - Om**2
    if k2 <= 0: return 0.0
    return k2**1.5 * (4*Om**2 + 1)


def koide(m1, m2, m3):
    return (m1+m2+m3) / (np.sqrt(m1)+np.sqrt(m2)+np.sqrt(m3))**2


m_e, m_mu, m_tau = 0.51099895, 105.6583755, 1776.86
m_e_MeV = m_e
m_mu_MeV = m_mu
m_tau_MeV = m_tau

print("=" * 70)
print("  The 4th Mode: What particle is l=1?")
print("=" * 70)

print("\n--- Phase 1: Full spectrum at Phi0=2.35, alpha=0.5 ---\n")

Om_bg, sol_bg = solve_osc(2.35)
if Om_bg is None:
    prev = None
    for Phi0 in np.arange(0.05, 2.40, 0.05):
        if prev:
            Om, sol = solve_osc(Phi0, r_prev=prev.x, y_prev=prev.y, p_prev=prev.p)
        else:
            Om, sol = solve_osc(Phi0)
        if Om:
            prev = sol
    Om_bg, sol_bg = solve_osc(2.35, r_prev=prev.x, y_prev=prev.y, p_prev=prev.p)

if Om_bg:
    print(f"  Background: Phi0=2.35, Omega={Om_bg:.6f}")
    print()

    all_modes = {}
    for l_val in range(6):
        eigs = cavity_eigs(sol_bg.x, sol_bg.y[0], l_val, N=2000)
        for n, om in enumerate(eigs):
            if om > 0.001:
                all_modes[(n, l_val)] = om
                E = E_1d(om)
                kappa = np.sqrt(max(0, 1.0 - om**2))
                print(f"  (n={n}, l={l_val}): omega={om:.6f}  kappa={kappa:.4f}"
                      f"  E_1d={E:.6e}")

    print(f"\n  Total modes: {len(all_modes)}")

    print(f"\n--- Phase 2: Mass prediction for l=1 mode ---\n")

    tau_om = all_modes.get((0, 0), None)
    mystery_om = all_modes.get((0, 1), None)
    muon_om = all_modes.get((1, 0), None)
    elec_om = all_modes.get((0, 2), None)

    if tau_om and muon_om and elec_om:
        E_tau = E_1d(tau_om)
        E_muon = E_1d(muon_om)
        E_elec = E_1d(elec_om)

        scale = m_tau_MeV / E_tau

        print(f"  Using tau to set the scale: E_tau={E_tau:.6f} -> {m_tau_MeV} MeV")
        print(f"  Scale factor: {scale:.2f} MeV per unit E_1d")
        print()

        print(f"  Particle       omega     E_1d        Mass (MeV)  Expt (MeV)")
        print(f"  " + "-" * 64)
        print(f"  tau (l=0,n=0)  {tau_om:.4f}  {E_tau:.6e}  {E_tau*scale:10.2f}"
              f"  {m_tau_MeV:.2f}")

        if mystery_om:
            E_mystery = E_1d(mystery_om)
            m_mystery = E_mystery * scale
            print(f"  ??? (l=1,n=0)  {mystery_om:.4f}  {E_mystery:.6e}  {m_mystery:10.2f}"
                  f"  ???")
        else:
            m_mystery = 0

        print(f"  muon (l=0,n=1) {muon_om:.4f}  {E_muon:.6e}  {E_muon*scale:10.2f}"
              f"  {m_mu_MeV:.2f}")
        print(f"  elec (l=2,n=0) {elec_om:.4f}  {E_elec:.6e}  {E_elec*scale:10.2f}"
              f"  {m_e_MeV:.2f}")

        print(f"\n  Predicted mass of l=1 mode: {m_mystery:.1f} MeV")

        known_particles = [
            ("pion (pi+/-)", 139.57),
            ("pion (pi0)", 134.98),
            ("kaon (K+/-)", 493.68),
            ("eta", 547.86),
            ("rho", 775.26),
            ("omega meson", 782.65),
            ("phi meson", 1019.46),
            ("proton", 938.27),
            ("neutron", 939.57),
        ]

        print(f"\n  Known particles near {m_mystery:.0f} MeV:")
        for name, mass in known_particles:
            if abs(mass - m_mystery) / m_mystery < 0.3:
                print(f"    {name}: {mass:.2f} MeV ({abs(mass-m_mystery)/m_mystery*100:.1f}% off)")

    print(f"\n--- Phase 3: How does the 4th mode change with Phi0? ---\n")

    prev_sol = None
    print(f"  {'Phi0':>5} {'Om_bg':>8} {'om_00':>7} {'om_01':>7} {'om_10':>7}"
          f" {'om_02':>7} {'N_tot':>5} {'m_l1(MeV)':>10}")

    for Phi0 in np.arange(0.1, 4.01, 0.1):
        if prev_sol:
            Om, sol = solve_osc(Phi0, r_prev=prev_sol.x, y_prev=prev_sol.y,
                                 p_prev=prev_sol.p)
        else:
            Om, sol = solve_osc(Phi0)
        if Om is None:
            continue
        prev_sol = sol

        modes = {}
        for l_val in range(4):
            eigs = cavity_eigs(sol.x, sol.y[0], l_val, N=1500)
            for n, om in enumerate(eigs):
                if om > 0.001:
                    modes[(n, l_val)] = om

        om_00 = modes.get((0,0), 0)
        om_01 = modes.get((0,1), 0)
        om_10 = modes.get((1,0), 0)
        om_02 = modes.get((0,2), 0)

        if om_00 > 0 and E_1d(om_00) > 0:
            sc = m_tau_MeV / E_1d(om_00)
            m_l1 = E_1d(om_01) * sc if om_01 > 0 else 0
        else:
            m_l1 = 0

        print(f"  {Phi0:5.1f} {Om:8.4f} {om_00:7.4f} {om_01:7.4f}"
              f" {om_10:7.4f} {om_02:7.4f} {len(modes):5d}"
              f" {m_l1:10.1f}")

    print(f"\n--- Phase 4: 4-particle Koide ---\n")
    print(f"  Koide formula for 4 particles (all triples):\n")

    if tau_om and mystery_om and muon_om and elec_om:
        E_vals = {
            'tau': E_1d(tau_om),
            'l1': E_1d(mystery_om),
            'muon': E_1d(muon_om),
            'elec': E_1d(elec_om),
        }

        names = list(E_vals.keys())
        for combo in combinations(names, 3):
            ms = [E_vals[n] for n in combo]
            Q = koide(*ms)
            print(f"    ({combo[0]:>5}, {combo[1]:>5}, {combo[2]:>5}):"
                  f" Q = {Q:.6f}  |Q-2/3| = {abs(Q-2/3):.2e}")
