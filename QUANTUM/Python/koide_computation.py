"""
ISPG Oscillon Eigenvalue Spectrum and Koide Ratio Computation

Physics:
  The 1D oscillon equation (near-core approximation):
    Phi'' + (Omega^2 - 1)*Phi + Phi^2 = 0
  has localized solution Phi(x) = A * sech^2(Bx)
  with A = 3(1-Omega^2)/2, B = sqrt(1-Omega^2)/2.

  Linearized perturbations around this background satisfy:
    u'' + [E - ell(ell+1)/r^2 + 2*Phi_bg(r)] * u = 0
  where u = r*R(r), E = omega^2 - 1, and mass ~ omega.

  This is a Schrodinger eigenvalue problem with
  Poschl-Teller potential + centrifugal barrier.

  For ell=0: analytical eigenvalues (Poschl-Teller, s=3):
    E_n = -(1-Omega^2)/4 * (3-n)^2,  n=0,1,2

  For ell>0: numerical computation required.

Step 1: Linearized spectrum in the 1D sech^2 background (all ell)
Step 2: Full 3D nonlinear ground-state oscillon + linearized spectrum
Step 3: Koide ratio check
"""

import numpy as np
from scipy.linalg import eigvalsh_tridiagonal
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from itertools import combinations


def sech2_spectrum(Omega_bg, ell_max=5, N=6000, r_max=60.0):
    """
    Compute linearized eigenvalue spectrum in the 1D sech^2 background.
    Returns dict: ell -> array of bound-state eigenvalues E < 0.
    """
    beta2 = 1 - Omega_bg**2
    A = 1.5 * beta2
    B = 0.5 * np.sqrt(beta2)
    V0 = 2 * A  # = 3*beta2

    h = r_max / (N + 1)
    r = np.linspace(h, r_max - h, N)

    V_bg = V0 / np.cosh(B * r)**2

    results = {}
    for ell in range(ell_max + 1):
        V_cent = ell * (ell + 1) / r**2
        # Hamiltonian: H = -d^2/dr^2 + V_cent - V_bg
        diag = 2.0 / h**2 + V_cent - V_bg
        off_diag = -np.ones(N - 1) / h**2
        evals = eigvalsh_tridiagonal(diag, off_diag)
        bound = evals[evals < -1e-10]
        results[ell] = np.sort(bound)
    return results


def oscillon_3d_rhs(r, y, Omega):
    """RHS: Phi'' + (2/r)*Phi' + (Omega^2 - 1)*Phi + Phi^2 = 0"""
    Phi, dPhi = y
    if r < 1e-10:
        d2Phi = -(Omega**2 - 1) * Phi / 3.0 - Phi**2 / 3.0
    else:
        d2Phi = -(2.0 / r) * dPhi - (Omega**2 - 1) * Phi - Phi**2
    return [dPhi, d2Phi]


def integrate_oscillon_3d(Phi0, Omega, r_max=60.0):
    """Integrate 3D oscillon from r~0 to r_max."""
    r0 = 1e-6
    sol = solve_ivp(
        lambda r, y: oscillon_3d_rhs(r, y, Omega),
        [r0, r_max],
        [Phi0, 0.0],
        method='RK45',
        max_step=0.02,
        rtol=1e-11,
        atol=1e-13,
        dense_output=True
    )
    return sol


def find_3d_oscillon(Phi0, r_max=60.0):
    """
    For given central amplitude Phi0, find Omega such that Phi -> 0.
    Returns (Omega, sol) or (None, None).
    """
    def tail_value(Omega):
        sol = integrate_oscillon_3d(Phi0, Omega, r_max)
        return sol.y[0, -1]

    # Bracket search
    Omega_lo, Omega_hi = 0.01, 0.999
    try:
        f_lo = tail_value(Omega_lo)
        f_hi = tail_value(Omega_hi)
        if f_lo * f_hi > 0:
            # Try to find sign change
            for Om_test in np.linspace(0.1, 0.99, 50):
                f_test = tail_value(Om_test)
                if f_lo * f_test < 0:
                    Omega_hi = Om_test
                    f_hi = f_test
                    break
                elif f_test * f_hi < 0:
                    Omega_lo = Om_test
                    f_lo = f_test
                    break

        Omega_star = brentq(tail_value, Omega_lo, Omega_hi, xtol=1e-12)
        sol = integrate_oscillon_3d(Phi0, Omega_star, r_max)
        return Omega_star, sol
    except Exception as e:
        return None, None


def spectrum_from_profile(r_grid, Phi_profile, ell_max=5, N=6000, r_max=60.0):
    """
    Compute linearized spectrum using a numerically obtained background.
    """
    Phi_interp = interp1d(r_grid, Phi_profile, kind='cubic',
                          bounds_error=False, fill_value=0.0)

    h = r_max / (N + 1)
    r = np.linspace(h, r_max - h, N)
    V_bg = 2.0 * Phi_interp(r)
    V_bg = np.maximum(V_bg, 0.0)

    results = {}
    for ell in range(ell_max + 1):
        V_cent = ell * (ell + 1) / r**2
        diag = 2.0 / h**2 + V_cent - V_bg
        off_diag = -np.ones(N - 1) / h**2
        evals = eigvalsh_tridiagonal(diag, off_diag)
        bound = evals[evals < -1e-10]
        results[ell] = np.sort(bound)
    return results


def koide_ratio(m1, m2, m3):
    """Compute Koide ratio Q = (m1+m2+m3) / (sqrt(m1)+sqrt(m2)+sqrt(m3))^2"""
    s = m1 + m2 + m3
    sq = np.sqrt(m1) + np.sqrt(m2) + np.sqrt(m3)
    return s / sq**2


def find_best_koide(modes, target=2.0/3.0):
    """
    Among all combinations of 3 modes, find the one closest to Q=2/3.
    modes: list of (n, ell, E, omega) tuples.
    """
    best_Q = None
    best_combo = None
    for combo in combinations(modes, 3):
        masses = [m[3] for m in combo]
        if all(m > 0 for m in masses):
            Q = koide_ratio(*masses)
            if best_Q is None or abs(Q - target) < abs(best_Q - target):
                best_Q = Q
                best_combo = combo
    return best_Q, best_combo


def print_spectrum(spectrum, label=""):
    """Print bound-state eigenvalues and corresponding masses."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    all_modes = []
    for ell in sorted(spectrum.keys()):
        for n, E in enumerate(spectrum[ell]):
            omega_sq = 1.0 + E
            if omega_sq > 0:
                omega = np.sqrt(omega_sq)
                print(f"  (n={n}, ell={ell}): E = {E:12.8f},  omega = {omega:.8f}")
                all_modes.append((n, ell, E, omega))
            else:
                print(f"  (n={n}, ell={ell}): E = {E:12.8f},  [unphysical: omega^2 < 0]")
    return all_modes


# ============================================================
#  STEP 1: Analytical check (Poschl-Teller, ell=0)
# ============================================================
print("\n" + "="*60)
print("  STEP 1: Analytical Poschl-Teller eigenvalues (ell=0)")
print("="*60)
for Omega_bg in [0.8, 0.9, 0.95]:
    beta2 = 1 - Omega_bg**2
    print(f"\n  Omega_bg = {Omega_bg},  1-Omega^2 = {beta2:.4f}")
    for n_pt in range(3):
        E_analytic = -beta2 / 4.0 * (3 - n_pt)**2
        omega = np.sqrt(1.0 + E_analytic) if 1.0 + E_analytic > 0 else 0.0
        print(f"    n={n_pt}: E = {E_analytic:.8f}, omega = {omega:.8f}")

# ============================================================
#  STEP 2: Numerical spectrum in sech^2 background (all ell)
# ============================================================
print("\n\n" + "="*60)
print("  STEP 2: Numerical spectrum in sech^2 background")
print("="*60)

for Omega_bg in [0.5, 0.7, 0.8, 0.9, 0.95]:
    label = f"sech^2 background, Omega_bg = {Omega_bg}"
    spectrum = sech2_spectrum(Omega_bg, ell_max=4, N=6000, r_max=60.0)
    all_modes = print_spectrum(spectrum, label)

    if len(all_modes) >= 3:
        Q_best, combo_best = find_best_koide(all_modes)
        if combo_best:
            masses = [m[3] for m in combo_best]
            ratios = [m / min(masses) for m in masses]
            print(f"\n  >> Best Koide: Q = {Q_best:.8f}  (target = 0.666667)")
            print(f"     Modes: {[(m[0], m[1]) for m in combo_best]}")
            print(f"     omega: {[f'{m:.6f}' for m in masses]}")
            print(f"     ratios: {[f'{r:.4f}' for r in ratios]}")

# ============================================================
#  STEP 3: Full 3D nonlinear oscillon + linearized spectrum
# ============================================================
print("\n\n" + "="*60)
print("  STEP 3: 3D nonlinear oscillon ground state")
print("="*60)

for Phi0 in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    print(f"\n  Phi0 = {Phi0}")
    Omega_3d, sol_3d = find_3d_oscillon(Phi0, r_max=60.0)
    if Omega_3d is not None:
        print(f"    Ground-state Omega = {Omega_3d:.10f}")

        r_grid = sol_3d.t
        Phi_grid = sol_3d.y[0]

        # Verify: profile should be positive and decaying
        Phi_max = np.max(Phi_grid)
        Phi_end = Phi_grid[-1]
        print(f"    Phi_max = {Phi_max:.6f}, Phi(r_max) = {Phi_end:.2e}")

        # Linearized spectrum
        spectrum_3d = spectrum_from_profile(
            r_grid, Phi_grid, ell_max=4, N=6000, r_max=60.0
        )
        label = f"3D oscillon, Phi0={Phi0}, Omega={Omega_3d:.6f}"
        all_modes_3d = print_spectrum(spectrum_3d, label)

        if len(all_modes_3d) >= 3:
            Q_best, combo_best = find_best_koide(all_modes_3d)
            if combo_best:
                masses = [m[3] for m in combo_best]
                ratios = [m / min(masses) for m in masses]
                print(f"\n  >> Best Koide: Q = {Q_best:.8f}  (target = 0.666667)")
                print(f"     Modes: {[(m[0], m[1]) for m in combo_best]}")
                print(f"     omega: {[f'{m:.6f}' for m in masses]}")
                print(f"     ratios: {[f'{r:.4f}' for r in ratios]}")
    else:
        print("    [No bound state found]")

# ============================================================
#  STEP 4: Koide ratio as function of x = alpha/lambda0
# ============================================================
print("\n\n" + "="*60)
print("  STEP 4: Analytical Koide function Q(x)")
print("="*60)

x_values = np.linspace(0, 100, 1000)
Q_values = (3 + x_values) / (np.sqrt(1 + x_values) + 2)**2

x_koide = 33 + 24 * np.sqrt(2)
Q_at_x = (3 + x_koide) / (np.sqrt(1 + x_koide) + 2)**2
print(f"  x_Koide = 33 + 24*sqrt(2) = {x_koide:.6f}")
print(f"  Q(x_Koide) = {Q_at_x:.10f}")
print(f"  Target Q = {2/3:.10f}")
print(f"  Match: {abs(Q_at_x - 2/3) < 1e-8}")

# Check with actual lepton masses
m_e = 0.51099895   # MeV
m_mu = 105.6583755  # MeV
m_tau = 1776.86     # MeV
Q_lepton = koide_ratio(m_e, m_mu, m_tau)
print(f"\n  Actual lepton Koide ratio: Q = {Q_lepton:.10f}")
print(f"  Deviation from 2/3: {abs(Q_lepton - 2/3):.2e}")
