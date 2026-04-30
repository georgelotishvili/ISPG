"""Cavity spectrum of the ISPG oscillon.

From ISPG_Quantum.tex (Sec. mass_determination):
  m_n = m_tilde * Omega_n
  Mass ratios = eigenfrequency ratios of the self-consistent cavity.

Approach:
  1. Background oscillon Phi0(r) satisfies:
     Phi'' + (2/r)Phi' + (Om^2 - 1)Phi + g*Phi^2 = 0
  2. Linearized perturbations satisfy (u = r * delta_Phi):
     -u'' + V_eff(r)*u = omega^2 * u
     V_eff(r) = 1 - 2g*Phi0(r) + l(l+1)/r^2
  3. Discretize and diagonalize -> eigenvalues omega^2
"""
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

import numpy as np
from scipy.integrate import solve_bvp
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.interpolate import interp1d
from itertools import combinations


def solve_background(Phi0, r_max=50.0, Om_guess=None, g=1.0,
                     r_prev=None, y_prev=None, p_prev=None):
    if Om_guess is None:
        kappa_est = np.sqrt(min(Phi0 / 4.2, 0.95))
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
        return np.array([ya[0] - Phi0, ya[1], yb[0]])

    if r_prev is not None and y_prev is not None:
        N_pts = max(400, len(r_prev))
        r = np.linspace(1e-6, r_max, N_pts)
        f0 = interp1d(r_prev, y_prev[0], fill_value=0.0, bounds_error=False)
        f1 = interp1d(r_prev, y_prev[1], fill_value=0.0, bounds_error=False)
        scale = Phi0 / max(abs(y_prev[0][0]), 1e-30)
        y_init = np.vstack([f0(r) * scale, f1(r) * scale])
        Om_guess = p_prev[0] if p_prev is not None else Om_guess
    else:
        r = np.linspace(1e-6, r_max, 400)
        kappa_g = np.sqrt(max(0.01, 1.0 - Om_guess**2))
        Phi_init = Phi0 / np.cosh(r * kappa_g)**2
        dPhi_init = np.gradient(Phi_init, r)
        y_init = np.vstack([Phi_init, dPhi_init])

    sol = solve_bvp(ode, bc, r, y_init, p=[Om_guess],
                    tol=1e-6, max_nodes=20000, verbose=0)
    if sol.success:
        return sol.p[0], sol
    return None, None


def cavity_eigenvalues(r_bg, Phi_bg, g, l_val, N_grid=2000, r_max_grid=None):
    """Find eigenvalues of linearized equation via matrix diagonalization.

    -u'' + V_eff(r) u = E u,  where E = omega^2
    V_eff(r) = 1 - 2g*Phi0(r) + l(l+1)/r^2
    """
    if r_max_grid is None:
        r_max_grid = r_bg[-1] * 0.9

    dr = r_max_grid / (N_grid + 1)
    r = np.linspace(dr, r_max_grid - dr, N_grid)

    Phi_interp = interp1d(r_bg, Phi_bg, fill_value=0.0, bounds_error=False)
    Phi_vals = Phi_interp(r)

    V = 1.0 - 2.0 * g * Phi_vals + l_val * (l_val + 1) / r**2

    diag_main = 2.0 / dr**2 + V
    diag_off = -np.ones(N_grid - 1) / dr**2

    H = diags([diag_off, diag_main, diag_off], [-1, 0, 1], format='csc')

    n_eigs = min(20, N_grid - 2)
    try:
        eigenvalues, eigenvectors = eigsh(H, k=n_eigs, which='SM')
    except:
        return [], []

    bound_mask = eigenvalues < 1.0
    bound_eigs = eigenvalues[bound_mask]
    bound_vecs = eigenvectors[:, bound_mask]

    bound_eigs = np.sort(bound_eigs)

    omegas = np.sqrt(np.maximum(bound_eigs, 0))

    return omegas, r


def koide(m1, m2, m3):
    return (m1+m2+m3) / (np.sqrt(m1)+np.sqrt(m2)+np.sqrt(m3))**2


print("=" * 70)
print("  ISPG Cavity Spectrum: Matrix Diagonalization")
print("  Background oscillon creates potential well -> discrete modes")
print("=" * 70)

m_e, m_mu, m_tau = 0.51099895, 105.6583755, 1776.86

print("\n--- Step 1: Build background via continuation ---\n")

seed_Phi0 = 0.10
Om_seed, sol_seed = solve_background(seed_Phi0, r_max=60.0, Om_guess=0.99)
if Om_seed is None:
    print("Seed failed, trying Phi0=0.2")
    seed_Phi0 = 0.20
    Om_seed, sol_seed = solve_background(seed_Phi0, r_max=50.0, Om_guess=0.97)

if Om_seed:
    print(f"  Seed: Phi0={seed_Phi0}, Om={Om_seed:.6f},"
          f" kappa={np.sqrt(1-Om_seed**2):.4f}")
    prev_r = sol_seed.x
    prev_y = sol_seed.y
    prev_p = sol_seed.p
else:
    print("SEED FAILED!")
    sys.exit(1)

bg_solutions = {seed_Phi0: (Om_seed, sol_seed)}
for Phi0 in np.arange(seed_Phi0 + 0.05, 4.01, 0.05):
    Om, sol = solve_background(Phi0, r_max=50.0,
                                r_prev=prev_r, y_prev=prev_y, p_prev=prev_p)
    if Om is not None and 0.01 < Om < 0.999:
        bg_solutions[round(Phi0, 2)] = (Om, sol)
        prev_r = sol.x
        prev_y = sol.y
        prev_p = sol.p
    else:
        break

print(f"  Built {len(bg_solutions)} background solutions"
      f" (Phi0 = {min(bg_solutions):.2f} ... {max(bg_solutions):.2f})")

print("\n--- Step 2: Cavity spectrum for each background ---\n")

header = (f"  {'Phi0':>5} {'Om_bg':>8} {'kappa':>6}"
          f" | {'l=0 modes':>20} | {'l=1 modes':>20}"
          f" | {'l=2 modes':>20} | {'l=3 modes':>20}")
print(header)
print("  " + "-" * (len(header) - 2))

all_results = []

for Phi0 in sorted(bg_solutions.keys()):
    Om_bg, sol_bg = bg_solutions[Phi0]
    kappa = np.sqrt(max(0, 1 - Om_bg**2))

    r_bg = sol_bg.x
    Phi_bg = sol_bg.y[0]

    modes_by_l = {}
    for l_val in range(6):
        omegas, _ = cavity_eigenvalues(r_bg, Phi_bg, g=1.0, l_val=l_val,
                                        N_grid=2000)
        if len(omegas) > 0:
            modes_by_l[l_val] = omegas

    line = f"  {Phi0:5.2f} {Om_bg:8.5f} {kappa:6.4f} |"
    for l_val in range(4):
        if l_val in modes_by_l:
            om_str = ", ".join(f"{o:.4f}" for o in modes_by_l[l_val][:3])
            line += f" {om_str:>20} |"
        else:
            line += f" {'---':>20} |"
    print(line)

    all_omegas = []
    all_labels = []
    for l_val in sorted(modes_by_l.keys()):
        for n, om in enumerate(modes_by_l[l_val]):
            all_omegas.append(om)
            all_labels.append(f"(n={n},l={l_val})")

    if len(all_omegas) >= 3:
        for combo_idx in combinations(range(len(all_omegas)), 3):
            oms = sorted([all_omegas[i] for i in combo_idx])
            labs = [all_labels[i] for i in combo_idx]
            if oms[0] > 0.001:
                Q = koide(*oms)
                all_results.append((Phi0, Q, oms, labs))

print(f"\n--- Step 3: Best Koide Q across all backgrounds ---\n")

if all_results:
    all_results.sort(key=lambda x: abs(x[1] - 2/3))
    for i in range(min(10, len(all_results))):
        Phi0, Q, oms, labs = all_results[i]
        print(f"  #{i+1}: Phi0={Phi0:.2f}, Q={Q:.8f}, |Q-2/3|={abs(Q-2/3):.2e}")
        print(f"       modes: {labs}")
        print(f"       omega: [{oms[0]:.6f}, {oms[1]:.6f}, {oms[2]:.6f}]")
        print(f"       ratios: 1 : {oms[1]/oms[0]:.2f} : {oms[2]/oms[0]:.2f}")
        print()

print("\n--- Step 4: Mass ratios check ---\n")

if all_results:
    Phi0, Q, oms, labs = all_results[0]
    print(f"  Best match: Phi0={Phi0:.2f}")
    print(f"  Q = {Q:.10f}")
    print(f"  |Q - 2/3| = {abs(Q - 2/3):.2e}")
    print(f"  Modes: {labs}")
    print(f"  omega_1 : omega_2 : omega_3 = {oms[0]:.6f} : {oms[1]:.6f} : {oms[2]:.6f}")
    print(f"  Mass ratios: 1 : {oms[1]/oms[0]:.4f} : {oms[2]/oms[0]:.4f}")
    print(f"  Target:      1 : {m_mu/m_e:.1f} : {m_tau/m_e:.1f}")
    print(f"\n  NOTE: In this model, mass ~ omega (not mass ~ E(omega))")
    print(f"  The 1D energy formula E(Om)=(1-Om^2)^(3/2)*(4Om^2+1)")
    print(f"  gives a DIFFERENT mass mapping with much wider ratios.")

    print(f"\n  Mass ratios using E(omega) instead:")
    def E1d(Om):
        return (1 - Om**2)**1.5 * (4*Om**2 + 1)
    E_vals = [E1d(o) for o in oms]
    if E_vals[0] > 0:
        print(f"  E ratios: 1 : {E_vals[1]/E_vals[0]:.2f} : {E_vals[2]/E_vals[0]:.2f}")
        Q_E = koide(*E_vals)
        print(f"  Koide Q(E) = {Q_E:.8f}  |Q-2/3| = {abs(Q_E - 2/3):.2e}")
