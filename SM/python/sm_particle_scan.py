"""
SM ნაწილაკებისთვის ოპტიმალური N-ის ძიება Mathieu ლადერზე
=============================================================
m_N = m_1 · b_N(q),   q = 1.853,   m_e = m_1 · b_5(q)
  →  b_N/b_5  =  m_particle / m_e

ყოველი ჩამოთვლილი non-neutrino massive SM ნაწილაკისთვის:
  1. target  = (m_obs / m_e) · b_5(q)
  2. optimal N  = argmin |b_N(q) − target|
  3. ფიტის ცდომილება  =  |b_N − target| / target

კრიტერიუმი:  ცდომილება < 1%  →  კარგი ფიტი
            ცდომილება < 5%  →  მისაღები
            ცდომილება > 5%  →  ლადერი ვერ იტევს
"""

import numpy as np
from scipy.special import mathieu_a, mathieu_b
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

Q_FIT = 1.853
N_MAX = 5000   # ძიების ზედა ზღვარი

# ----------------------------------------------------------------------
#  ჩამოთვლილი non-neutrino massive SM ნაწილაკების დაკვირვებადი მასები
#  (MeV, PDG 2024/2025 inputs used by this scan)
# ----------------------------------------------------------------------
PARTICLES = [
    # (სახელი, მასა MeV, კომენტარი)
    ("e",    0.51099895,  "ლეპტონი — ფიტის ანკერი N=5"),
    ("μ",    105.6583755, "ლეპტონი — ფიტის ანკერი N=72"),
    ("τ",    1776.86,     "ლეპტონი — პაპერი N=295"),
    ("u",    2.16,        "ქვარკი — PDG 2024, პაპერი 2.04 MeV"),
    ("d",    4.67,        "ქვარკი — PDG 2024, პაპერი 4.59 MeV"),
    ("s",    93.4,        "ქვარკი — PDG 2024"),
    ("c",    1273.0,      "ქვარკი — PDG 2024"),
    ("b",    4183.0,      "ქვარკი — PDG 2024"),
    ("t",    172570.0,    "ქვარკი — PDG 2024"),
    ("W",    80369.2,     "ვექტორული ბოზონი"),
    ("Z",    91188.0,     "ვექტორული ბოზონი"),
    ("H",    125250.0,    "Higgs სკალარი"),
]


def b_ladder(N_max, q):
    """b_N(q) ყველა N ∈ [1, N_max]-თვის."""
    return np.array([mathieu_b(n, q) for n in range(1, N_max + 1)])


def find_optimal_N(mass_ratio, b_arr, b5):
    """
    შენატანი: mass_ratio = m_particle / m_e
    უბრუნებს: (N_opt, b_N_opt, რელატიური_ცდომილება, N_continuous, significance)
    significance = half-gap / observed_err   (>1 ნიშნავს ფიტი არატრივიალურია)
    """
    target = mass_ratio * b5
    diffs = np.abs(b_arr - target)
    idx = int(np.argmin(diffs))
    N_opt = idx + 1
    b_opt = b_arr[idx]
    err = abs(b_opt - target) / target
    N_cont = np.sqrt(target)
    # half-gap = |b_{N+1} - b_{N-1}| / 2
    if 0 < idx < len(b_arr) - 1:
        half_gap = 0.5 * (b_arr[idx + 1] - b_arr[idx - 1]) / target
    else:
        half_gap = float('nan')
    # significance: რამდენჯერ უკეთესია ფიტი "შემთხვევით" ფიტთან შედარებით
    sig = (half_gap / 2) / err if err > 0 else float('inf')
    return N_opt, b_opt, err, N_cont, sig


def main():
    print("=" * 76)
    print(f"  SM ნაწილაკების N-ძიება Mathieu ლადერზე  (q = {Q_FIT})")
    print("=" * 76)

    print("\n  ლადერის გამოთვლა... ", end="", flush=True)
    b_arr = b_ladder(N_MAX, Q_FIT)
    print(f"b_1…b_{N_MAX} მზადაა.")

    m_e = PARTICLES[0][1]
    b5 = b_arr[4]  # N=5 → idx 4
    print(f"  m_e = {m_e:.6f} MeV,   b_5 = {b5:.6f}")
    print(f"  m_1 = m_e / b_5 = {m_e / b5:.6e} MeV")

    # ცხრილი
    print("\n" + "─" * 100)
    print(f"  {'ნაწ.':<4} {'m_obs [MeV]':>13} {'m/m_e':>13} "
          f"{'N_opt':>6} {'N_cont':>9} {'ცდომ. %':>9} {'ნახევ.გაპი %':>13} "
          f"{'სიგნიფ.':>9}")
    print("─" * 100)

    results = []
    for (name, m_obs, _) in PARTICLES:
        mr = m_obs / m_e
        N_opt, b_opt, err, N_cont, sig = find_optimal_N(mr, b_arr, b5)
        # half-gap გადაითვლება ხელახლა ცხრილისთვის
        idx = N_opt - 1
        if 0 < idx < len(b_arr) - 1:
            hg = 0.5 * (b_arr[idx + 1] - b_arr[idx - 1]) / (mr * b5)
        else:
            hg = float('nan')
        print(f"  {name:<4} {m_obs:>13.4f} {mr:>13.3f} "
              f"{N_opt:>6d} {N_cont:>9.2f} {err * 100:>8.3f}% {hg * 100:>12.3f}% "
              f"{sig:>8.1f}x")
        results.append((name, m_obs, mr, N_opt, N_cont, err, sig))

    print("─" * 100)

    # ---- სტატისტიკა ----
    errs = np.array([r[5] for r in results])
    sigs_e_excluded = np.array([r[6] for r in results[1:]])  # e გამოვაკლოთ (მასის ანკერია)
    sigs_calibration_excluded = np.array(
        [r[6] for r in results if r[0] not in ("e", "μ")]
    )
    good = np.sum(errs < 0.01)
    ok = np.sum(errs < 0.05)
    bad = np.sum(errs >= 0.05)
    print(f"\n  სტატისტიკა:")
    print(f"    კარგი ფიტი (<1%):  {good}/{len(results)}")
    print(f"    მისაღები (<5%):    {ok}/{len(results)}")
    print(f"    ვერ ფიტდება (≥5%): {bad}/{len(results)}")
    print(f"    საშ. ცდომილება:     {np.mean(errs) * 100:.2f}%")
    print(f"    მაქს. ცდომილება:    {np.max(errs) * 100:.2f}%")
    print(f"\n  სიგნიფიკანტობა (e mass anchor excluded; μ calibration still included):")
    print(f"    სიგნიფ. > 10x (არატრივიალური):  {np.sum(sigs_e_excluded > 10)}/{len(sigs_e_excluded)}")
    print(f"    სიგნიფ. > 3x (შესამჩნევი):       {np.sum(sigs_e_excluded > 3)}/{len(sigs_e_excluded)}")
    print(f"    სიგნიფ. < 1x (ტრივიალური):       {np.sum(sigs_e_excluded < 1)}/{len(sigs_e_excluded)}")
    print(f"    გეომ. საშ. სიგნიფ.:              {np.exp(np.mean(np.log(sigs_e_excluded[sigs_e_excluded > 0]))):.2f}x")

    print(f"\n  დამოუკიდებელი სიგნიფიკანტობა (e + μ calibration excluded):")
    print(f"    სიგნიფ. > 10x:                  {np.sum(sigs_calibration_excluded > 10)}/{len(sigs_calibration_excluded)}")
    print(f"    სიგნიფ. > 3x:                   {np.sum(sigs_calibration_excluded > 3)}/{len(sigs_calibration_excluded)}")
    print(f"    გეომ. საშ. სიგნიფ.:              {np.exp(np.mean(np.log(sigs_calibration_excluded[sigs_calibration_excluded > 0]))):.2f}x")

    # ---- N სპექტრის სტრუქტურა ----
    print(f"\n  N-ის სტრუქტურა:")
    N_vals = sorted([r[3] for r in results])
    print(f"    {N_vals}")
    print(f"\n  ინტერპრეტაცია:")
    print(f"    • ანკერები (e=5):                                  ზუსტი კონსტრუქციით")
    print(f"    • calibration point (μ=72):                         q ფიქსირდება აქ")
    print(f"    • დაბალი N პრედიქცია (τ=295, d=15, s=68):           ცდ. < 2%")
    print(f"    • მაღალი N (b, t, W, Z, H > 450):                  ლადერი ძალიან ხშირი")
    print(f"    • პრობლემატური: u (PDG 2.16 vs პაპერი 2.04)")

    # ---- ფიგურა ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    N_plot = np.arange(1, N_MAX + 1)
    ax.loglog(N_plot, b_arr, 'k-', lw=1, alpha=0.5, label='b_N(q=1.853)')
    colors = plt.cm.tab20(np.linspace(0, 1, len(results)))
    for (name, m_obs, mr, N_opt, N_cont, err, _sig), c in zip(results, colors):
        ax.plot(N_opt, mr * b5, 'o', color=c, markersize=8,
                label=f"{name} (N={N_opt}, {err*100:.2f}%)")
    ax.set_xlabel("N")
    ax.set_ylabel(r"$b_N(q)$ ~ $m/m_1$")
    ax.set_title("Mathieu ლადერი და SM ნაწილაკები")
    ax.legend(fontsize=8, ncol=2, loc='lower right')
    ax.grid(True, which='both', alpha=0.3)

    ax = axes[1]
    names = [r[0] for r in results]
    errs_pct = [r[5] * 100 for r in results]
    _ = [r[6] for r in results]  # sig (unused here, tuples now have 7 fields)
    bar_colors = ['g' if e < 1 else ('y' if e < 5 else 'r') for e in errs_pct]
    ax.bar(names, errs_pct, color=bar_colors, edgecolor='black')
    ax.axhline(1.0, color='g', ls='--', alpha=0.5, label='1% ზღვარი')
    ax.axhline(5.0, color='r', ls='--', alpha=0.5, label='5% ზღვარი')
    ax.set_yscale('log')
    ax.set_ylabel("რელატიური ცდომილება [%]")
    ax.set_title("ფიტის ცდომილება ყოველი ნაწილაკისთვის")
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    path = OUT / "sm_particle_scan.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ფიგურა → {path}")

    # ---- დასკვნა ----
    print("\n" + "=" * 76)
    print("  დასკვნა")
    print("=" * 76)
    if bad == 0:
        print("  ყველა ჩამოთვლილი non-neutrino massive SM state მოთავსებულია Mathieu ლადერზე < 5% ცდომ.-ით")
    elif bad <= 3:
        print(f"  {bad} ნაწილაკი ვერ ფიტდება — ლადერის საზღვრები")
    else:
        print(f"  {bad}/{len(results)} ნაწილაკი ვერ ფიტდება — ლადერი არასრულია")
    print("=" * 76)


if __name__ == "__main__":
    main()
