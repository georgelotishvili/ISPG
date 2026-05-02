"""
ISPG Oscillon q-Diagnostic
==========================
Semi-analytic support for the Mathieu parameter q
and the particle mass spectrum from the ISPG scalar-metric theory.

Addresses Weakness #1 (circularity in the mass spectrum):
  The fitted value q_fit = 1.853 is not independently recovered
  exactly here.  The script computes q_semi ≈ 1.9065 from the
  G₂(X) = -(M_Pl²/2)X + (α/M²)X² correction bundle, supporting
  the mechanism while leaving full lattice/PDE closure open.

Structure:
  Part 1 — Oscillon profile via shooting (3D spherical ODE)
  Part 2 — Cavity eigenvalue problem (bound states → mass ratios)
  Part 3 — δq correction estimates (metric + gradient + harmonics)
  Part 4 — Mathieu mass spectrum comparison at q_semi and q_fit
  Part 5 — Koide formula verification

References:
  ISPG_FrequencyToMass.tex  §2.1–2.6  (G₂, Mathieu, q derivation)
  ISPG_Quantum.tex          §5,§7     (oscillon, cavity spectrum)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import iv as besseli   # modified Bessel I_n
from scipy.linalg import eigh_tridiagonal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =====================================================================
#  Physical constants (dimensionless units: ℏ = c = R_core = 1)
# =====================================================================
ALPHA_NL  = 0.5       # ISPG coupling in F(Φ) = Φ(1 − e^{−αΦ})
PHI_C     = 2.35      # central oscillon amplitude  (ISPG_Quantum §7)
OMEGA_EXP = 0.866     # expected eigenfrequency      (ISPG_Quantum §7)
Q_FIT     = 1.853     # fitted Mathieu parameter      (FrequencyToMass)


# =====================================================================
#  PART 1 — Oscillon profile (3D spherically symmetric shooting)
# =====================================================================
def oscillon_rhs(r, y, Omega, alpha=ALPHA_NL):
    """RHS of the 3D radial ODE:
       Φ'' + (2/r)Φ' + Ω²Φ − Φ e^{−αΦ} = 0
    rewritten as a first-order system  y = [Φ, Φ'].
    Regularised at r = 0 via L'Hôpital: (2/r)Φ' → 2Φ''(0)/3."""
    Phi, dPhi = y
    if r < 1e-12:
        d2Phi = (Phi * np.exp(-alpha * Phi) - Omega**2 * Phi) / 3.0
    else:
        d2Phi = -2.0 / r * dPhi - Omega**2 * Phi + Phi * np.exp(-alpha * Phi)
    return [dPhi, d2Phi]


def find_oscillon(Phi_c=PHI_C, alpha=ALPHA_NL,
                  r_max=40.0, Omega_lo=0.30, Omega_hi=0.999):
    """Shooting method: find Ω so that Φ(r_max) → 0."""

    def residual(Omega):
        sol = solve_ivp(lambda r, y: oscillon_rhs(r, y, Omega, alpha),
                        [1e-10, r_max], [Phi_c, 0.0],
                        method='RK45', rtol=1e-12, atol=1e-14,
                        max_step=0.05)
        return sol.y[0, -1]

    Omega = brentq(residual, Omega_lo, Omega_hi, xtol=1e-12)

    r_eval = np.linspace(1e-10, r_max, 4000)
    sol = solve_ivp(lambda r, y: oscillon_rhs(r, y, Omega, alpha),
                    [1e-10, r_max], [Phi_c, 0.0],
                    method='RK45', t_eval=r_eval,
                    rtol=1e-12, atol=1e-14)
    return sol.t, sol.y[0], Omega


# =====================================================================
#  PART 2 — Cavity eigenvalue problem (linearised Schrödinger)
# =====================================================================
def cavity_eigenvalues(r, Phi_bg, alpha=ALPHA_NL, ell=0, n_want=6):
    """Solve  −u'' + [V(r) + ℓ(ℓ+1)/r²] u = ω² u
    on a uniform grid with  u(0)=0, u(r_max)=0.

    V(r) = (1 − αΦ_bg) e^{−αΦ_bg}   (linearised metric backreaction).

    Returns eigenvalues (ω²) and eigenvectors for the lowest n_want states.
    """
    N = len(r)
    dr = r[1] - r[0]

    aP = alpha * Phi_bg
    V = (1.0 - aP) * np.exp(-aP)

    centrifugal = np.zeros(N)
    centrifugal[1:] = ell * (ell + 1) / r[1:]**2
    centrifugal[0] = centrifugal[1]

    W = V + centrifugal     # total effective potential

    # interior points (Dirichlet BCs: u[0] = u[N-1] = 0)
    diag     = 2.0 / dr**2 + W[1:-1]
    off_diag = -np.ones(N - 3) / dr**2

    eigenvalues, eigenvectors = eigh_tridiagonal(diag, off_diag)

    # bound states: ω² < V(∞) = 1
    n_return = min(n_want, len(eigenvalues))
    return eigenvalues[:n_return], eigenvectors[:, :n_return]


# =====================================================================
#  PART 3 — δq corrections from the oscillon profile
# =====================================================================
def delta_q_metric(Phi0):
    """Metric backreaction:  δq_met = I₂(2Φ₀) / I₀(2Φ₀)
    (ISPG_FrequencyToMass eq:dq_metric)."""
    z = 2.0 * Phi0
    return float(besseli(2, z) / besseli(0, z))


def delta_q_gradient(r, Phi_bg, omega0=1.0):
    """Gradient correction:  δq_grad ~ ⟨(∇Φ)²⟩ / (Φ₀²ω₀²)
    integrated over the oscillon volume (4πr²dr weighting).
    (ISPG_FrequencyToMass eq:dq_grad)."""
    dPhi = np.gradient(Phi_bg, r)
    weight = 4.0 * np.pi * r**2

    num = np.trapz(dPhi**2 * weight, r)
    den = np.trapz(Phi_bg**2 * weight, r) * omega0**2
    if den == 0:
        return 0.0
    return num / den


def delta_q_harmonics(Phi0, alpha=ALPHA_NL):
    """Higher-harmonic correction from cos³θ coupling.
    F''(Φ₀) = α² Φ₀ e^{−αΦ₀}  gives the curvature of the
    nonlinearity that mediates the 3ω → ω down-conversion.
    (ISPG_FrequencyToMass eq:dq_harm — order-of-magnitude.)"""
    F_pp = alpha**2 * Phi0 * np.exp(-alpha * Phi0)
    return F_pp * Phi0 / 4.0


def compute_q(r, Phi_bg, Phi0):
    """Combine leading order + three corrections → q_total."""
    eps = 4.0 / 3.0                           # self-consistency (exact)
    q0  = eps / (2.0 * (2.0 - eps))           # = 1 (exact)

    dq_m = delta_q_metric(Phi0)
    dq_g = delta_q_gradient(r, Phi_bg)
    dq_h = delta_q_harmonics(Phi0)

    return dict(q_leading=q0, epsilon=eps,
                dq_metric=dq_m, dq_gradient=dq_g, dq_harmonics=dq_h,
                q_total=q0 + dq_m + dq_g + dq_h,
                beta_eff=(dq_m + dq_g + dq_h) / Phi0**2)


# =====================================================================
#  PART 4 — Mathieu characteristic values and mass spectrum
# =====================================================================
from scipy.special import mathieu_a as _mathieu_a
from scipy.special import mathieu_b as _mathieu_b

def mathieu_b(N, q):
    """b_N(q) — odd Mathieu characteristic value (scipy wrapper)."""
    return float(_mathieu_b(N, q))

def mathieu_a(N, q):
    """a_N(q) — even Mathieu characteristic value (scipy wrapper)."""
    return float(_mathieu_a(N, q))


def mass_spectrum_table(q, N_max=80):
    """Mass spectrum from Mathieu characteristic values b_N(q).
    Returns list of dicts with N, b_N, a_N, gap, stability flag."""
    rows = []
    for N in range(1, N_max + 1):
        bN = mathieu_b(N, q)
        aN = mathieu_a(N, q)
        gap = aN - bN
        rows.append(dict(N=N, b_N=bN, a_N=aN, gap=gap,
                         stable=(gap < 0.05)))
    return rows


# particle assignments  (N → name, observed mass in GeV)
# Confirmed from MASS/ISPG_FrequencyToMass.tex:
#   e  : N = 5    (line 1854)
#   μ  : N = 72   (line 1854)
#   τ  : N = 295  (line 1854, N_τ = 5 × 59 from √(m_τ/m_e))
#   u  : N = 10   (line 2799, m_u = 2.04 MeV = m_1 · b_10)
#   d  : N = 15   (line 2799, m_d = 4.59 MeV = m_1 · b_15)
#   N=3: predicted 186 keV resonance (line 2782-2786)
#   N=4: predicted 329 keV resonance (line 2767-2780)
# Heavier quarks and electroweak bosons do not have explicit m = m_1·b_N
# assignments in the manuscripts; previous dict values were speculative
# and gave 80–99.99% errors in unified_oscillon_predictions.py.
PARTICLES = {
    3:  ('N3',   0.000186),   # predicted resonance (paper)
    4:  ('N4',   0.000329),   # predicted resonance (paper)
    5:  ('e',    0.000511),
    10: ('u',    0.00204),    # paper prediction 2.04 MeV
    15: ('d',    0.00459),    # paper prediction 4.59 MeV
    72: ('μ',    0.10566),
    295:('τ',    1.77686),    # corrected from 312
}

# =====================================================================
#  PART 5 — Koide formula
# =====================================================================
def koide_ratio(m1, m2, m3):
    """Q = (m1+m2+m3) / (√m1+√m2+√m3)².  Exact value = 2/3."""
    s  = np.sqrt(m1) + np.sqrt(m2) + np.sqrt(m3)
    return (m1 + m2 + m3) / s**2


# =====================================================================
#  MAIN
# =====================================================================
def main():
    print("=" * 72)
    print("  ISPG OSCILLON q-DIAGNOSTIC")
    print("  Semi-analytic support for the Mathieu parameter q")
    print("=" * 72)

    # ------------------------------------------------------------------
    #  Step 1: oscillon profile
    # ------------------------------------------------------------------
    print("\n[1] Solving oscillon profile (3D spherical, shooting)...")
    print(f"    Φ₀ = {PHI_C},  α = {ALPHA_NL},  target Ω = {OMEGA_EXP}")

    r, Phi, Omega = find_oscillon()

    print(f"    Ω_computed = {Omega:.8f}")
    print(f"    Ω_expected = {OMEGA_EXP}")
    print(f"    |ΔΩ/Ω|    = {abs(Omega - OMEGA_EXP)/OMEGA_EXP*100:.4f} %")

    # ------------------------------------------------------------------
    #  Step 2: cavity eigenvalues  (ℓ = 0, 1, 2)
    # ------------------------------------------------------------------
    print("\n[2] Cavity eigenvalues (linearised Schrödinger)...")

    expected = [
        # (ℓ, n, particle, ω_exp, κ_exp)
        (0, 0, 'τ',         0.664, 0.748),
        (0, 1, 'μ',         0.970, 0.241),
        (2, 0, 'e',         0.999, 0.044),
        (1, 0, 'grav(ℓ=1)', 0.866, 0.500),
    ]

    all_states = []
    for ell in range(3):
        evals, evecs = cavity_eigenvalues(r, Phi, ell=ell, n_want=4)
        bound = evals[evals < 1.0]
        for n_idx, ev in enumerate(bound):
            omega_cav = np.sqrt(max(ev, 0.0))
            kappa_cav = np.sqrt(max(1.0 - ev, 0.0))
            all_states.append((ell, n_idx, omega_cav, kappa_cav, ev))

    all_states.sort(key=lambda s: s[2])

    print(f"    {'ℓ':>3} {'n':>3} {'ω':>10} {'κ':>10} {'ω²':>12}")
    print(f"    {'---':>3} {'---':>3} {'--------':>10} {'--------':>10} {'----------':>12}")
    for ell, n, om, ka, ev in all_states:
        print(f"    {ell:>3d} {n:>3d} {om:>10.6f} {ka:>10.6f} {ev:>12.6f}")

    print("\n    Expected (ISPG_Quantum.tex Table):")
    for ell, n, name, om_e, ka_e in expected:
        print(f"      ({ell},{n}) {name:>12s}:  ω = {om_e},  κ = {ka_e}")

    # Koide check — the mass proxy is the 1D binding energy E₁D
    # (ISPG_Quantum.tex Table in §7), not ω itself.
    # We compute E₁D as the cavity-mode energy integral.
    ell0_states = [(om, ka, ev) for ell, n, om, ka, ev in all_states if ell == 0]
    ell2_states = [(om, ka, ev) for ell, n, om, ka, ev in all_states if ell == 2]

    if len(ell0_states) >= 2 and len(ell2_states) >= 1:
        # Mass proportional to ω (eq:mass_from_omega)
        m_tau, m_mu, m_e = (ell0_states[0][0],
                            ell0_states[1][0],
                            ell2_states[0][0])
        Q_omega = koide_ratio(m_tau, m_mu, m_e)

        # Mass proportional to κ² = 1 - ω² (binding energy proxy)
        kappa_tau = ell0_states[0][1]
        kappa_mu  = ell0_states[1][1]
        kappa_e   = ell2_states[0][1]
        Q_kappa = koide_ratio(kappa_tau**2, kappa_mu**2, kappa_e**2)

        print(f"\n    Koide check (mass ∝ ω):   Q = {Q_omega:.6f}  "
              f"(2/3 = 0.666667, Δ = {abs(Q_omega-2/3)/(2/3)*100:.2f}%)")
        print(f"    Koide check (mass ∝ κ²):  Q = {Q_kappa:.6f}  "
              f"(2/3 = 0.666667, Δ = {abs(Q_kappa-2/3)/(2/3)*100:.2f}%)")
        print(f"    (Paper reports Q = 2/3 ± 0.009% using full E₁D;"
              f"\n     the exact E₁D requires the nonlinear eigenvalue"
              f" integral.)")

    # ------------------------------------------------------------------
    #  Step 3: Mathieu parameter q
    # ------------------------------------------------------------------
    print("\n[3] Mathieu parameter q: leading order plus correction estimates...")

    res = compute_q(r, Phi, PHI_C)

    print(f"    Leading order:  ε = {res['epsilon']:.6f}  →  q₀ = {res['q_leading']:.6f}")
    print(f"    Corrections:")
    print(f"      δq_metric   = {res['dq_metric']:.6f}   (I₂(2Φ₀)/I₀(2Φ₀))")
    print(f"      δq_gradient = {res['dq_gradient']:.6f}   (volume-averaged ⟨(∇Φ)²⟩)")
    print(f"      δq_harmonics= {res['dq_harmonics']:.6f}   (3ω → ω coupling)")
    print(f"    ─────────────────────────────────────")
    print(f"    q_computed  = {res['q_total']:.6f}")
    print(f"    q_fitted    = {Q_FIT}")
    print(f"    |Δq/q|     = {abs(res['q_total'] - Q_FIT)/Q_FIT*100:.2f}%")
    print(f"    β_eff       = {res['beta_eff']:.6f}   (q = 1 + β Φ₀²)")

    # ------------------------------------------------------------------
    #  Step 4: Mathieu mass spectrum
    # ------------------------------------------------------------------
    q_use = res['q_total']
    print(f"\n[4] Mass spectrum at q = {q_use:.4f}")

    spec = mass_spectrum_table(q_use, N_max=80)

    b5  = next(s['b_N'] for s in spec if s['N'] == 5)

    print(f"\n    {'N':>5}  {'b_N(q)':>12}  {'a_N(q)':>12}  {'gap':>12}  "
          f"{'m/m_e':>10}  {'status':>8}  {'particle':>8}")
    print(f"    {'─'*5}  {'─'*12}  {'─'*12}  {'─'*12}  "
          f"{'─'*10}  {'─'*8}  {'─'*8}")
    for s in spec:
        N = s['N']
        mass_ratio = s['b_N'] / b5 if b5 > 0 else 0
        status = "stable" if s['stable'] else "UNSTABLE"
        name = PARTICLES.get(N, ('', 0))[0]
        if N <= 10 or N in PARTICLES:
            print(f"    {N:>5d}  {s['b_N']:>12.4f}  {s['a_N']:>12.4f}  "
                  f"{s['gap']:>12.6f}  {mass_ratio:>10.4f}  "
                  f"{status:>8s}  {name:>8s}")

    # muon/electron and tau/electron ratios
    b72 = next((s['b_N'] for s in spec if s['N'] == 72), None)
    b312 = next((s['b_N'] for s in spec if s['N'] == 312), None) if len(spec) >= 312 else None

    if b72 and b5:
        print(f"\n    μ/e mass ratio:  b₇₂/b₅ = {b72/b5:.3f}   (observed: 206.768)")

    # also at fitted q for comparison
    spec_fit = mass_spectrum_table(Q_FIT, N_max=80)
    b5f  = next(s['b_N'] for s in spec_fit if s['N'] == 5)
    b72f = next(s['b_N'] for s in spec_fit if s['N'] == 72)
    print(f"    μ/e at q_fit:    b₇₂/b₅ = {b72f/b5f:.3f}")

    # ------------------------------------------------------------------
    #  Step 5: plots
    # ------------------------------------------------------------------
    print("\n[5] Generating figures...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle('ISPG Oscillon Diagnostic — Semi-analytic q support',
                 fontsize=15, fontweight='bold')

    # (a) oscillon profile
    ax = axes[0, 0]
    ax.plot(r, Phi, 'b-', lw=2,
            label=rf'$\Phi_0 = {PHI_C}$,  $\Omega = {Omega:.4f}$')
    Phi_sech = 1.5*(1 - Omega**2)/np.cosh(0.5*np.sqrt(1-Omega**2)*r)**2
    ax.plot(r, Phi_sech, 'r--', lw=1, alpha=0.6, label='sech² (1D analytic)')
    ax.set_xlabel(r'$r\;/\;R_c$')
    ax.set_ylabel(r'$\Phi(r)$')
    ax.set_title('(a) Oscillon radial profile')
    ax.legend()
    ax.set_xlim(0, 20)
    ax.grid(True, alpha=0.3)

    # (b) effective cavity potential
    ax = axes[0, 1]
    aP = ALPHA_NL * Phi
    V_eff = (1.0 - aP) * np.exp(-aP)
    ax.plot(r, V_eff, 'r-', lw=2, label=r'$V_{\rm eff}(r)$')
    ax.axhline(1.0, color='gray', ls=':', lw=1, label='continuum ($V=1$)')
    colors = ['tab:purple', 'tab:green', 'tab:orange', 'tab:cyan']
    for i, (ell, n, om, ka, ev) in enumerate(all_states[:4]):
        ax.axhline(ev, color=colors[i % len(colors)], ls='--', alpha=0.7,
                   label=rf'$\omega^2={ev:.4f}$  ($\ell={ell}$, $n={n}$)')
    ax.set_xlabel(r'$r\;/\;R_c$')
    ax.set_ylabel(r'$V_{\rm eff}$')
    ax.set_title('(b) Cavity potential & bound-state levels')
    ax.legend(fontsize=7.5, loc='lower right')
    ax.set_xlim(0, 15)
    ax.grid(True, alpha=0.3)

    # (c) δq_metric vs Φ₀
    ax = axes[1, 0]
    P_range = np.linspace(0.1, 4.0, 200)
    dqm = [delta_q_metric(P) for P in P_range]
    ax.plot(P_range, dqm, 'b-', lw=2,
            label=r'$\delta q_{\rm metric} = I_2(2\Phi_0)/I_0(2\Phi_0)$')
    ax.axhline(Q_FIT - 1.0, color='red', ls='--', lw=1.5,
               label=rf'$\delta q = {Q_FIT - 1:.3f}$ (target)')
    ax.axvline(PHI_C, color='green', ls=':', lw=1.5,
               label=rf'$\Phi_0 = {PHI_C}$')
    ax.scatter([PHI_C], [delta_q_metric(PHI_C)], s=80, c='green', zorder=5)
    ax.set_xlabel(r'$\Phi_0$')
    ax.set_ylabel(r'$\delta q$')
    ax.set_title(r'(c)  $\delta q$ corrections vs amplitude')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) Mathieu stability diagram
    ax = axes[1, 1]
    qq = np.linspace(0, 3.5, 200)
    for N in range(1, 8):
        bline = [mathieu_b(N, q_) for q_ in qq]
        aline = [mathieu_a(N, q_) for q_ in qq]
        ax.fill_between(qq, bline, aline, alpha=0.25,
                        label=f'N={N}' if N <= 5 else None)
    ax.axvline(q_use,  color='blue', ls='-',  lw=2,
               label=rf'$q_{{\rm comp}} = {q_use:.3f}$')
    ax.axvline(Q_FIT, color='red',  ls='--', lw=1.5,
               label=rf'$q_{{\rm fit}} = {Q_FIT}$')
    ax.set_xlabel('$q$')
    ax.set_ylabel('$a$  (Mathieu parameter)')
    ax.set_title('(d) Mathieu stability zones')
    ax.legend(fontsize=8)
    ax.set_ylim(-5, 80)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = 'ispg_oscillon_results.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"    Saved → {out_path}")

    # ------------------------------------------------------------------
    #  Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  Oscillon:   Φ₀ = {PHI_C},   Ω = {Omega:.6f}  "
          f"(expected {OMEGA_EXP})")
    print(f"  Leading q:  ε = 4/3 → q₀ = 1  (self-consistency, exact)")
    print(f"  Corrections:")
    print(f"    δq_metric    = {res['dq_metric']:.6f}")
    print(f"    δq_gradient  = {res['dq_gradient']:.6f}")
    print(f"    δq_harmonics = {res['dq_harmonics']:.6f}")
    dq_total = res['dq_metric'] + res['dq_gradient'] + res['dq_harmonics']
    print(f"    δq_total     = {dq_total:.6f}   (target: 0.853)")
    print(f"  q_semi         = {res['q_total']:.6f}")
    print(f"  q_fitted       = {Q_FIT}")
    print(f"  Gap            = {abs(res['q_total'] - Q_FIT):.4f}  "
          f"({abs(res['q_total'] - Q_FIT)/Q_FIT*100:.1f}%)")
    print(f"  β_eff          = {res['beta_eff']:.4f}")
    print()

    status = "PARTIAL" if abs(res['q_total'] - Q_FIT) > 0.1 else "GOOD"
    if abs(res['q_total'] - Q_FIT) < 0.01:
        status = "EXCELLENT"
    print(f"  Assessment: {status}")
    if status == "PARTIAL":
        print("  The semi-analytical δq formulas capture the dominant")
        print("  correction (metric Bessel ratio) but the gradient and")
        print("  harmonic terms require full lattice PDE evolution to")
        print("  compute exactly.  The leading q = 1 is rigorous.")
    print("=" * 72)

    return res


if __name__ == '__main__':
    main()
