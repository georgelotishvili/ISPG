"""
QUANTUM paper-ის ოსცილონის განტოლების პირდაპირი ტესტი
=========================================================

QUANTUM paper (eq 1173-1186) განტოლება:
    Φ'' + 2/r·Φ' = (1-ω²)·Φ - Φ²

    (Gleiser-ის 3D სფერული ოსცილონი)

პერტურბაცია:  V_pert(r) = 1 - 2·Φ_bg(r)

შედარებისთვის MASS/G_2(X):  V_pert(r) = (1-αΦ_bg)·exp(-αΦ_bg)

კითხვა:  QUANTUM-ის ოსცილონი იძლევა თუ არა ω² ფარდობებს,
          რომლებიც ემთხვევა ლეპტონების მასებს (1:207:3477)?
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.linalg import eigh_tridiagonal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

m_e, m_mu, m_tau = 0.5109989461, 105.6583745, 1776.86


def safe_sqrt(x):
    return np.sqrt(x) if x >= 0 else float('nan')


def quantum_oscillon_rhs(r, y, omega2):
    """QUANTUM eq (1174):  Φ'' + 2/r·Φ' = (1-ω²)·Φ - Φ²"""
    Phi, dPhi = y
    if r < 1e-12:
        # origin:  3·Φ'' = (1-ω²)·Φ - Φ²
        d2 = ((1.0 - omega2) * Phi - Phi**2) / 3.0
    else:
        d2 = -2.0/r * dPhi + (1.0 - omega2) * Phi - Phi**2
    return [dPhi, d2]


def find_quantum_oscillon(Phi_c, r_max=30.0, N_grid=6000):
    """Gleiser-ის ოსცილონი მოცემული Φ(0)=Φ_c-თვის. ვპოულობთ ω²-ს."""
    def residual(omega2):
        sol = solve_ivp(lambda r, y: quantum_oscillon_rhs(r, y, omega2),
                        [1e-10, r_max], [Phi_c, 0.0],
                        method='RK45', rtol=1e-11, atol=1e-13, max_step=0.05)
        return sol.y[0, -1]
    # იდეა:  Ω² < 1, უნდა გავფილტროთ
    try:
        omega2 = brentq(residual, 0.1, 0.999, xtol=1e-11)
    except Exception:
        return None, None, None
    r_eval = np.linspace(1e-10, r_max, N_grid)
    sol = solve_ivp(lambda r, y: quantum_oscillon_rhs(r, y, omega2),
                    [1e-10, r_max], [Phi_c, 0.0],
                    method='RK45', t_eval=r_eval,
                    rtol=1e-11, atol=1e-13)
    return sol.t, sol.y[0], omega2


def perturbation_modes_quantum(r, Phi_bg, ell, n_want=8):
    """V_pert = 1 - 2·Φ_bg  (QUANTUM-ის ოსცილონი)."""
    N = len(r); dr = r[1] - r[0]
    V = 1.0 - 2.0 * Phi_bg
    cen = np.zeros(N)
    cen[1:] = ell*(ell+1)/r[1:]**2
    cen[0] = cen[1]
    W = V + cen
    diag = 2.0/dr**2 + W[1:-1]
    off  = -np.ones(N-3)/dr**2
    evals, evecs = eigh_tridiagonal(diag, off)
    return evals[:n_want], evecs[:, :n_want]


def expectation_r(r_interior, u_n):
    norm = np.trapz(u_n**2, r_interior)
    if norm <= 0:
        return np.nan
    r2 = np.trapz(u_n**2 * r_interior**2, r_interior) / norm
    return np.sqrt(r2)


def scan_amplitudes():
    """ვცდი სხვადასხვა Φ_c-ს და ვამოწმებ ω²-ფარდობებს."""
    print("=" * 78)
    print("  QUANTUM ოსცილონი — Φ(0) ვარიაცია")
    print("=" * 78)
    print("\n  Φ(0)  Ω²_bg   ბმული მოდების ω²  (paper-ის id-ით)")
    print("  ──────────────────────────────────────────────────")

    amplitudes = [0.5, 0.8, 1.0, 1.2, 1.4, 1.48]  # Φ_c < 1.5 (sech² ceiling)
    results = []
    for Phi_c in amplitudes:
        r, Phi, Omega2 = find_quantum_oscillon(Phi_c, r_max=30.0)
        if r is None:
            continue
        r_in = r[1:-1]
        modes = {}
        for ell in range(3):
            evals, evecs = perturbation_modes_quantum(r, Phi, ell, n_want=5)
            for n in range(5):
                if evals[n] < 1.0:  # ბმული (ω² < V_∞ = 1)
                    R = expectation_r(r_in, evecs[:, n])
                    modes[(ell, n)] = (evals[n], R)
        results.append((Phi_c, Omega2, modes))
        tau_ω2  = modes.get((0, 0), (np.nan,))[0]
        mu_ω2   = modes.get((0, 1), (np.nan,))[0]
        e_ω2    = modes.get((2, 0), (np.nan,))[0]
        bound_cnt = sum(1 for _ in modes)
        print(f"  {Phi_c:.2f}  {Omega2:.4f}  τ={tau_ω2:.4f}  "
              f"μ={mu_ω2:.4f}  e={e_ω2:.4f}  ({bound_cnt} bnd)")
    return results


def test_mass_ratios(results):
    print("\n" + "=" * 78)
    print("  ლეპტონების მასის ფარდობა QUANTUM-ის ოსცილონიდან")
    print("=" * 78)
    print("\n  პრედიქცია (QUANTUM eq 5189):  m_i/m_j = Ω_i/Ω_j")
    print("\n  Φ_c   ω_e     ω_μ     ω_τ    m_μ/m_e  m_τ/m_e")
    print("  ─────────────────────────────────────────────────")
    obs_mu = m_mu / m_e
    obs_tau = m_tau / m_e
    best = (float('inf'), None)
    for Phi_c, Omega2, modes in results:
        if (0, 0) in modes and (0, 1) in modes and (2, 0) in modes:
            ω_tau = safe_sqrt(modes[(0, 0)][0])
            ω_mu  = safe_sqrt(modes[(0, 1)][0])
            ω_e   = safe_sqrt(modes[(2, 0)][0])
            r_mu = ω_mu / ω_e
            r_tau = ω_tau / ω_e
            err_mu = abs(r_mu - obs_mu) / obs_mu * 100
            err_tau = abs(r_tau - obs_tau) / obs_tau * 100
            tot = err_mu + err_tau
            print(f"  {Phi_c:.2f}  {ω_e:.4f}  {ω_mu:.4f}  {ω_tau:.4f}  "
                  f"{r_mu:8.4f}  {r_tau:9.4f}")
            if tot < best[0]:
                best = (tot, Phi_c)
    print(f"\n  სამიზნე:  m_μ/m_e = {obs_mu:.2f},  m_τ/m_e = {obs_tau:.1f}")
    if best[1] is not None:
        print(f"\n  ❌ საუკეთესო ფიტი Φ_c={best[1]}: ცდომ. სულ = {best[0]:.1f}%")
    print("\n  სტრუქტურული შედეგი: QUANTUM ოსცილონის ω-ფარდობები ~1 რჩება")
    print("  მიუხედავად Φ(0) ცვლილების, პრედიქცია თანაფარდობის")
    print("  ტოლი არ ხდება.")


def main():
    results = scan_amplitudes()
    test_mass_ratios(results)

    # --- ფიგურა ---
    if results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        for Phi_c, Omega2, modes in results[::2]:
            r, Phi, _ = find_quantum_oscillon(Phi_c)
            ax.plot(r, Phi, label=f"Φ(0)={Phi_c}, Ω²={Omega2:.3f}")
        ax.set_xlabel("r")
        ax.set_ylabel("Φ(r)")
        ax.set_title("QUANTUM (Gleiser) ოსცილონის პროფილები")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 15)

        ax = axes[1]
        Phi_arr = [r[0] for r in results]
        ω_e_arr = []
        ω_mu_arr = []
        ω_tau_arr = []
        for _, _, modes in results:
            ω_e_arr.append(safe_sqrt(modes.get((2, 0), (np.nan,))[0]))
            ω_mu_arr.append(safe_sqrt(modes.get((0, 1), (np.nan,))[0]))
            ω_tau_arr.append(safe_sqrt(modes.get((0, 0), (np.nan,))[0]))
        ax.plot(Phi_arr, ω_e_arr, 'bo-', label='e ← (ℓ=2, n=0)')
        ax.plot(Phi_arr, ω_mu_arr, 'gs-', label='μ ← (ℓ=0, n=1)')
        ax.plot(Phi_arr, ω_tau_arr, 'r^-', label='τ ← (ℓ=0, n=0)')
        ax.set_xlabel("Φ(0)")
        ax.set_ylabel("ω")
        ax.set_title("ცავიტის ω vs ოსცილონის ამპლიტუდა\n"
                     "(QUANTUM-ის ფორმულის კრიტიკა)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUT / "quantum_oscillon_test.png", dpi=150,
                    bbox_inches='tight')
        plt.close()
        print(f"\n  ფიგურა → {OUT / 'quantum_oscillon_test.png'}")


if __name__ == "__main__":
    main()
