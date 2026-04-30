"""
ISPG — 3D ოსცილონის ენერგიის შედარება E_1D(Ω) ფორმულასთან
===========================================================
დიაგნოსტიკა (არაფერს ცვლის .tex ფაილებში).

კითხვა:
  QUANTUM/Python/test_e1d_derivation.py-ში ფიგურირებს ფორმულა
      E_1D(Ω) = (1 − Ω²)^{3/2} · (4Ω² + 1)           (sech² ოსცილონის 1D ენერგია)
  რომელიც Koide-ს 2/3 აკმაყოფილებს 3D ცავიტის ω-ებზე, მაგრამ სკრიპტის
  საკუთარი კომენტარი ამას "empirical coincidence"-ს ეძახის.

  ამ სკრიპტის მიზანი:
  უშუალოდ გამოვთვალოთ 3D ოსცილონის ფუძე-პროფილის ენერგიის ინტეგრალი
      E_3D(Ω) = 4π ∫₀^∞ [½(Φ'₀)² + V(Φ₀) + ½Ω² Φ₀²] r² dr
  Ω-ის ფუნქციად, და შევადაროთ E_1D(Ω)-ს.

  თუ E_3D(Ω) / E_1D(Ω) ≈ მუდმივი Ω ∈ (0, 1)-ის ფართო ფართოებაზე,
  ფორმულა სტრუქტურულად ფესვები აქვს 3D ენერგიაში (მუდმივი = გეომ.ფაქტორი).
  თუ ფარდობა ცვალებადია → E_1D მხოლოდ ემპირიული ფიტია.

  ეს არის პირველი ნაბიჯი ფორმალური 3D→1D რედუქციის მხრივ.

ველის განტოლებები (test_e1d_derivation.py-დან):
  არაწრფივობა:  NL(Φ) = Φ · (1 − exp(−αΦ)),  α = 0.5
  ODE: Φ'' + (2/r)Φ' + (Ω² − 1)Φ + NL(Φ) = 0
  შესაბამისი პოტენციალი (V'(Φ) = Φ − NL(Φ) = Φ·e^{−αΦ}):
      V(Φ) = (1/α²) · [1 − (1 + αΦ) · e^{−αΦ}]
"""

import sys
import numpy as np
from scipy.integrate import solve_bvp, simpson
from scipy.interpolate import interp1d

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")

ALPHA = 0.5


# -----------------------------------------------------------------
#  ფიზიკური ფუნქციები
# -----------------------------------------------------------------

def nl_func(Phi):
    """ძალის არაწრფივი ნაწილი NL(Φ) — test_e1d_derivation.py-დან."""
    return Phi * (1.0 - np.exp(-ALPHA * Phi))


def V_potential(Phi):
    """პოტენციალი V(Φ) = (1/α²)[1 − (1+αΦ)e^{−αΦ}].
    იღებელია NL(Φ)-ის ინტეგრებით ველის განტოლების თანმიმდევრობად."""
    return (1.0 / ALPHA**2) * (1.0 - (1.0 + ALPHA * Phi) * np.exp(-ALPHA * Phi))


# -----------------------------------------------------------------
#  3D ოსცილონის ფუძე-პროფილის BVP ამოხსნა
#  (სრულად კოპირდება test_e1d_derivation.py-ის ალგორითმიდან)
# -----------------------------------------------------------------

def solve_osc(Phi0, r_max=60.0, r_prev=None, y_prev=None, p_prev=None):
    Om_guess = np.sqrt(max(0.01, 1.0 - min(Phi0 / 4.2, 0.95)))

    def ode(r, y, p):
        Om = p[0]
        Phi, dPhi = y
        r_safe = np.maximum(r, 1e-8)
        NL = nl_func(Phi)
        d2 = -(2.0 / r_safe) * dPhi - (Om**2 - 1) * Phi - NL
        d2_0 = -(Om**2 - 1) * Phi / 3.0 - NL / 3.0
        d2 = np.where(r < 1e-8, d2_0, d2)
        return np.vstack([dPhi, d2])

    def bc(ya, yb, p):
        return np.array([ya[0] - Phi0, ya[1], yb[0]])

    if r_prev is not None:
        N = max(500, len(r_prev))
        r = np.linspace(1e-6, r_max, N)
        f0 = interp1d(r_prev, y_prev[0], fill_value=0, bounds_error=False)
        f1 = interp1d(r_prev, y_prev[1], fill_value=0, bounds_error=False)
        sc = Phi0 / max(abs(y_prev[0][0]), 1e-30)
        y_init = np.vstack([f0(r) * sc, f1(r) * sc])
        Om_guess = p_prev[0]
    else:
        r = np.linspace(1e-6, r_max, 500)
        kg = np.sqrt(max(0.01, 1.0 - Om_guess**2))
        Phi_init = Phi0 / np.cosh(r * kg)**2
        y_init = np.vstack([Phi_init, np.gradient(Phi_init, r)])

    sol = solve_bvp(ode, bc, r, y_init, p=[Om_guess],
                    tol=1e-7, max_nodes=50000, verbose=0)
    if sol.success and 0.01 < sol.p[0] < 0.999:
        return sol.p[0], sol
    return None, None


# -----------------------------------------------------------------
#  ენერგიის ინტეგრალები
# -----------------------------------------------------------------

def energy_components(sol, Om):
    """აბრუნებს ცალ-ცალკე ინტეგრალების მნიშვნელობებს (4π-ს გარეშე)."""
    r = sol.x
    Phi = sol.y[0]
    dPhi = sol.y[1]
    vol = r**2

    I_grad = simpson(0.5 * dPhi**2 * vol, r)           # კინეტიკური (სივრცული)
    I_pot = simpson(V_potential(Phi) * vol, r)          # პოტენციალი V(Φ)
    I_norm = simpson(Phi**2 * vol, r)                    # ∫Φ² r² dr
    I_time = 0.5 * Om**2 * I_norm                       # კინეტ. (დროით. ½Ω²Φ²)

    return {
        "grad": 4 * np.pi * I_grad,
        "pot":  4 * np.pi * I_pot,
        "time": 4 * np.pi * I_time,
        "norm": 4 * np.pi * I_norm,
    }


def E_3D_avg(comps):
    """დროში-გასაშუალოებული ენერგია: <H> = grad + pot + time."""
    return comps["grad"] + comps["pot"] + comps["time"]


def E_3D_static(comps):
    """სტატიკური ენერგია (rotating frame): grad + pot − time."""
    return comps["grad"] + comps["pot"] - comps["time"]


def E_1D_formula(Om):
    """test_e1d_derivation.py:84 — κ³(4Ω² + 1)."""
    k2 = 1.0 - Om**2
    return k2**1.5 * (4.0 * Om**2 + 1.0) if k2 > 0 else 0.0


# -----------------------------------------------------------------
#  სკანი: Phi0 ∈ [0.10, 2.40] — ოსცილონების ფართო ოჯახი
# -----------------------------------------------------------------

def main():
    print("=" * 78)
    print("  3D ოსცილონის ენერგია vs E_1D(Ω) = κ³(4Ω²+1)")
    print("=" * 78)
    print()

    records = []
    prev = None
    Phi0_list = np.concatenate([
        np.arange(0.10, 0.80, 0.05),
        np.arange(0.80, 2.41, 0.10),
    ])

    for Phi0 in Phi0_list:
        if prev is not None:
            Om, sol = solve_osc(
                Phi0, r_prev=prev.x, y_prev=prev.y, p_prev=prev.p
            )
        else:
            Om, sol = solve_osc(Phi0)
        if Om is None or sol is None:
            continue
        comps = energy_components(sol, Om)
        e3d_a = E_3D_avg(comps)
        e3d_s = E_3D_static(comps)
        e1d = E_1D_formula(Om)
        records.append((Phi0, Om, comps, e3d_a, e3d_s, e1d))
        prev = sol

    if not records:
        print("  [შეცდომა] BVP არცერთ Phi0-ზე არ მოხსნა — შეამოწმე solver.")
        return

    # ცხრილი
    header = (
        f"  {'Phi0':>6} {'Ω':>8} "
        f"{'E_grad':>10} {'E_pot':>10} {'E_time':>10} "
        f"{'<H>':>10} {'H_stat':>10} {'E_1D':>10} "
        f"{'<H>/E1D':>10} {'H_st/E1D':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for Phi0, Om, c, avg, stat, e1d in records:
        r_avg = avg / e1d if abs(e1d) > 1e-12 else np.nan
        r_stat = stat / e1d if abs(e1d) > 1e-12 else np.nan
        print(
            f"  {Phi0:6.2f} {Om:8.4f} "
            f"{c['grad']:10.4f} {c['pot']:10.4f} {c['time']:10.4f} "
            f"{avg:10.4f} {stat:10.4f} {e1d:10.6f} "
            f"{r_avg:10.4f} {r_stat:10.4f}"
        )

    # სტატისტიკა
    avg_arr = np.array([r[3] for r in records])
    stat_arr = np.array([r[4] for r in records])
    e1d_arr = np.array([r[5] for r in records])
    ok = e1d_arr > 1e-10
    r_avg = avg_arr[ok] / e1d_arr[ok]
    r_stat = stat_arr[ok] / e1d_arr[ok]

    print()
    print("=" * 78)
    print("  ფარდობის სტატისტიკა")
    print("=" * 78)
    print(f"  <H>/E_1D:     mean = {r_avg.mean():10.4f}  std = {r_avg.std():9.4f}  "
          f"CV = {r_avg.std()/abs(r_avg.mean()):.4f}  "
          f"[min, max] = [{r_avg.min():.4f}, {r_avg.max():.4f}]")
    print(f"  H_stat/E_1D:  mean = {r_stat.mean():10.4f}  std = {r_stat.std():9.4f}  "
          f"CV = {r_stat.std()/abs(r_stat.mean()):.4f}  "
          f"[min, max] = [{r_stat.min():.4f}, {r_stat.max():.4f}]")

    print()
    print("  ინტერპრეტაცია:")
    print("    CV < 0.05  →  E_1D სტრუქტურული (ფარდობა ≈ მუდმივი გეომეტრიული ფაქტორი)")
    print("    CV > 0.20  →  E_1D ემპირიული დამთხვევა Koide Q-ზე, არა 3D ენერგიის ფორმა")
    print()

    # Ω-ზე დამოკიდებულების ფორმა
    print("=" * 78)
    print("  ფარდობის დამოკიდებულება Ω-ზე (სტრუქტურის დიაგნოსტიკა)")
    print("=" * 78)
    Om_arr = np.array([r[1] for r in records])[ok]
    # დავხარისხოთ Ω-ით
    idx = np.argsort(Om_arr)
    print(f"  {'Ω':>8} {'<H>/E_1D':>14} {'H_stat/E_1D':>14}")
    print("  " + "-" * 40)
    for i in idx[::2]:   # ყოველ მე-2 მონაცემს ვბეჭდავ
        print(f"  {Om_arr[i]:8.4f} {r_avg[i]:14.4f} {r_stat[i]:14.4f}")


if __name__ == "__main__":
    main()
