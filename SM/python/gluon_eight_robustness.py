"""
ISPG — 8-გლუონის პრედიქციის მტკიცეობის ტესტი
=================================================
gluon_eight.py აჩვენებს, რომ (g₀=2, V=Δ) ფიქსირებული
პარამეტრებით 8 მოდი გადარჩება. კითხვა:
  • ეს fine-tuning-ია თუ ფართო არეში მყარი პრედიქციაა?

ტესტი: 2D პარამეტრული სკანი (g₀, V) სივრცეში.
კრიტერიუმი: მოდი "confined" თუ scalar leak < 40%.

თუ N=8 რეგიონი ფართოა → ძლიერი octet-count მინიშნება.
თუ მხოლოდ ვიწრო ზოლია → fine-tuning, სუსტი მინიშნება.

Q.3 audit status: toy parameter scan at chosen leak threshold.
It tests partial robustness of the octet count; it is not a
first-principles G₂(X) prediction of SU(3)c.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

PI = np.pi
J21 = 5.76345919689      # j_{2,1} — პირველი spherical Bessel ℓ=2 ფესვი
E_T = J21**2             # tensor ℓ=2 ენერგია
E_S2 = (2 * PI)**2       # scalar n=2
DELTA = E_S2 - E_T       # რეზონანსული ცდომილება

LEAK_THRESHOLD = 0.40    # > 40% → deconfined


def mixing_leaks(g0, V):
    """სამი ჯგუფისთვის სკალარული leak %. უბრუნებს (A1, E_up, E_dn)."""
    modes_E = [E_T, E_T + V, E_T - V]  # A₁, E↑, E↓
    leaks = []
    for E_eff in modes_E:
        gap = E_eff - E_S2
        disc = np.sqrt(gap**2 + 4 * g0**2)
        if gap >= 0:
            cos2 = 0.5 * (1 + gap / disc)
        else:
            cos2 = 0.5 * (1 - gap / disc)
        leaks.append(1.0 - cos2)
    return leaks  # [A1, E_up, E_dn]


def count_confined(g0, V, thresh=LEAK_THRESHOLD):
    """უბრუნებს confined მოდების რიცხვს (0..10)."""
    leak_A1, leak_Eu, leak_Ed = mixing_leaks(g0, V)
    n = 5  # odd-parity ყოველთვის confined
    if leak_A1 < thresh: n += 1
    if leak_Eu < thresh: n += 2
    if leak_Ed < thresh: n += 2
    return n


def main():
    print("=" * 68)
    print("  8-გლუონის პრედიქციის მტკიცეობის ტესტი")
    print("=" * 68)
    print(" STATUS: toy scan at chosen leak threshold; partial octet-count robustness,")
    print("         not a first-principles G₂(X) SU(3)c derivation.\n")

    # --- სკანი ---
    g0_range = np.linspace(0.1, 8.0, 160)
    V_range  = np.linspace(0.1, 25.0, 250)
    G0, V = np.meshgrid(g0_range, V_range)

    N_confined = np.zeros_like(G0)
    for i in range(G0.shape[0]):
        for j in range(G0.shape[1]):
            N_confined[i, j] = count_confined(G0[i, j], V[i, j])

    # --- რეგიონალური სტატისტიკა ---
    total = N_confined.size
    frac = {n: np.sum(N_confined == n) / total for n in [5, 6, 7, 8, 9, 10]}

    print(f"\n პარამეტრული სივრცე:  g₀ ∈ [0.1, 8], V ∈ [0.1, 25]")
    print(f" ჯამი წერტილი: {total}")
    print(f" თრეშოლდი: leak < {LEAK_THRESHOLD*100:.0f}%  =  confined")
    print(f"\n confined მოდების განაწილება:")
    print(f"  ┌────────┬───────────┐")
    print(f"  │ N      │ %         │")
    print(f"  ├────────┼───────────┤")
    for n in [5, 6, 7, 8, 9, 10]:
        flag = "  ← SU(3)" if n == 8 else ""
        print(f"  │  {n:2d}    │  {frac[n]*100:6.2f} %{flag}")
    print(f"  └────────┴───────────┘")

    # --- N=8 რეგიონის შინაგანი ანალიზი ---
    mask_8 = (N_confined == 8)
    if mask_8.any():
        g0_in = G0[mask_8]
        V_in  = V[mask_8]
        print(f"\n N=8 რეგიონის ცენტრი და სიგანე:")
        print(f"  g₀: ცენტრი = {np.mean(g0_in):.2f},  "
              f"დიაპაზონი = [{np.min(g0_in):.2f}, {np.max(g0_in):.2f}]")
        print(f"   V: ცენტრი = {np.mean(V_in):.2f},  "
              f"დიაპაზონი = [{np.min(V_in):.2f}, {np.max(V_in):.2f}]")
        print(f"   (Δ = {DELTA:.2f}-ის რეფერენსი; gluon_eight.py-ს წერტილი"
              f" g₀=2, V={DELTA:.2f})")

    # --- თრეშოლდის მგრძნობელობა ---
    print(f"\n თრეშოლდი-სენსიტიურობა:")
    for t in [0.30, 0.35, 0.40, 0.45, 0.50]:
        N_at_t = np.array([[count_confined(g0_range[j], V_range[i], t)
                            for j in range(len(g0_range))]
                           for i in range(len(V_range))])
        frac_8 = np.sum(N_at_t == 8) / N_at_t.size
        print(f"   leak-threshold = {t*100:3.0f}%  →  N=8 ფრაქცია = "
              f"{frac_8*100:6.2f} %")

    # --- დიაგნოსტიკა ---
    print(f"\n" + "=" * 68)
    print(f" დასკვნა")
    print(f"=" * 68)
    frac_8 = frac[8]
    if frac_8 > 0.25:
        verdict = "მტკიცე — ფართო რეგიონი იძლევა ზუსტად 8 მოდს"
    elif frac_8 > 0.10:
        verdict = "ნაწილობრივ მტკიცე — საჭიროა სტრუქტურული მიზეზი"
    else:
        verdict = "fine-tuning — SU(3) პრედიქცია სუსტია"

    print(f"\n  N=8 დაფიქსირდა {frac_8*100:.1f}% სივრცეში")
    print(f"  ვერდიქტი: {verdict}")
    print(f"\n  კრიტიკული შენიშვნა:")
    print(f"    ეს toy 3-დონიანი Hamiltonian-ია. რეალური პრედიქცია")
    print(f"    უნდა გამოვიდეს ISPG-ს სრული 3D lattice ოსცილონის")
    print(f"    perturbation სპექტრიდან — არა ხელით არჩეული g₀-ით.")

    # --- ფიგურა ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("8-გლუონის პრედიქცია — პარამეტრული მტკიცეობა",
                 fontsize=13)

    ax = axes[0]
    cmap = plt.cm.get_cmap("viridis", 6)
    im = ax.pcolormesh(G0, V, N_confined, cmap=cmap,
                       vmin=5, vmax=10, shading='auto')
    cb = plt.colorbar(im, ax=ax, ticks=[5, 6, 7, 8, 9, 10])
    cb.set_label("confined modes", fontsize=11)
    ax.axhline(DELTA, color='r', ls=':', lw=1, label=f'V=Δ={DELTA:.2f}')
    ax.axvline(2.0, color='r', ls=':', lw=1, label='g₀=2')
    ax.set_xlabel(r"$g_0$", fontsize=12)
    ax.set_ylabel(r"$|V|$", fontsize=12)
    ax.set_title("(a) მოდების რიცხვი — (g₀, V) რუქა")
    ax.legend(loc='upper right', fontsize=9)

    ax = axes[1]
    mask_8_plot = (N_confined == 8).astype(float)
    ax.pcolormesh(G0, V, mask_8_plot, cmap='Greens',
                  vmin=0, vmax=1, shading='auto')
    ax.axhline(DELTA, color='r', ls=':', lw=1)
    ax.axvline(2.0, color='r', ls=':', lw=1)
    ax.plot(2.0, DELTA, 'r*', markersize=18, label='gluon_eight.py-ს წერტილი')
    ax.set_xlabel(r"$g_0$", fontsize=12)
    ax.set_ylabel(r"$|V|$", fontsize=12)
    ax.set_title(f"(b) N=8 რეგიონი  ({frac_8*100:.1f}% სივრცე)")
    ax.legend(fontsize=9, loc='upper left')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = OUT / "gluon_eight_robustness.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  ფიგურა → {path}")
    plt.close()


if __name__ == "__main__":
    main()
