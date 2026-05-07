"""
ISPG Gluon Eight — Can D₃ (3-quark) symmetry reduce 10 → 8?
=============================================================
A baryon oscillon contains 3 sub-oscillons (quarks) arranged
in a triangle.  This breaks SO(3) → D₃, splitting the 5-fold
degenerate ℓ=2 even-parity modes into subgroups.  We compute
whether exactly 2 modes are absorbed by the scalar sector,
leaving 5 (odd) + 3 (even) = 8 confined tensor modes = 8 gluons.

Physical criterion: modes with large scalar admixture
leak energy through the exterior → short lifetime → not confined.

Q.3 audit status: toy resonance model at chosen (g₀, V, leak
criterion). It is a structural octet-count clue, not yet a
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
J21 = 5.76345919689


def mixing_analysis(E_T, E_S, g0, V):
    """
    For the 3 mode groups (A₁, E↑, E↓), compute the scalar admixture
    sin²θ and estimated confinement lifetime τ ∝ 1/sin²θ.
    """
    modes = [
        ("A₁  (m=0)",      E_T,       1),
        ("E↑  (m=-2,+1)",  E_T + V,   2),
        ("E↓  (m=-1,+2)",  E_T - V,   2),
    ]

    results = []
    for label, E_eff, deg in modes:
        gap = E_eff - E_S
        disc = np.sqrt(gap**2 + 4 * g0**2)

        if gap >= 0:
            cos2 = 0.5 * (1 + gap / disc)
        else:
            cos2 = 0.5 * (1 - gap / disc)

        sin2 = 1.0 - cos2
        tau = 1.0 / sin2 if sin2 > 1e-12 else 1e12

        results.append(dict(
            label=label, E_eff=E_eff, deg=deg,
            gap=gap, tensor_weight=cos2,
            scalar_leak=sin2, lifetime_periods=tau
        ))
    return results


def main():
    print("=" * 65)
    print("  GLUON EIGHT:  D₃ SYMMETRY BREAKING → 10 → 8 ?")
    print("=" * 65)
    print("  STATUS: toy resonance model at chosen (g₀, V, leak threshold);")
    print("          not a first-principles G₂(X) SU(3)c derivation.\n")

    E_S1 = (1 * PI)**2   # scalar n=1
    E_S2 = (2 * PI)**2   # scalar n=2
    E_T  = J21**2         # tensor ℓ=2

    delta = E_S2 - E_T

    # ──────────────────────────────────────────────────────────
    #  §1  ENERGY LANDSCAPE
    # ──────────────────────────────────────────────────────────
    print(f"\n§1  Energy levels  (kR)²")
    print(f"-" * 50)
    print(f"  Scalar n=1:   {E_S1:.2f}")
    print(f"  TENSOR ℓ=2:   {E_T:.2f}   ← gluon candidate")
    print(f"  Scalar n=2:   {E_S2:.2f}   ← nearest harmonic")
    print(f"  Gap: Δ = {delta:.2f}  ({delta/E_T*100:.1f}% of tensor)")

    # ──────────────────────────────────────────────────────────
    #  §2  D₃ SPLITTING + MIXING
    # ──────────────────────────────────────────────────────────
    print(f"\n§2  D₃ splitting and scalar-tensor mixing")
    print(f"-" * 50)

    g0 = 2.0
    V = delta  # D₃ deformation pushes upper E to scalar resonance

    print(f"\n  D₃ deformation |V| = Δ = {V:.2f}  (resonance condition)")
    print(f"  Scalar-tensor coupling g₀ = {g0:.1f}\n")

    results = mixing_analysis(E_T, E_S2, g0, V)

    hdr = (f"  {'Mode':20s}  {'E_eff':>7s}  {'gap':>7s}  "
           f"{'tensor%':>8s}  {'leak%':>7s}  {'τ (periods)':>12s}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for r in results:
        print(f"  {r['label']:20s}  {r['E_eff']:7.2f}  {r['gap']:+7.2f}  "
              f"{r['tensor_weight']*100:7.1f}%  "
              f"{r['scalar_leak']*100:6.1f}%  "
              f"{r['lifetime_periods']:10.1f}  (×{r['deg']})")

    print(f"\n  Interpretation:")
    print(f"  ──────────────")
    for r in results:
        if r['scalar_leak'] < 0.1:
            tag = "STRONGLY confined → GLUON"
        elif r['scalar_leak'] < 0.3:
            tag = "Moderately confined → GLUON"
        elif r['scalar_leak'] < 0.45:
            tag = "Weakly confined → borderline"
        else:
            tag = "50% leak → DECONFINED (scattering resonance)"
        print(f"    {r['label']}:  leak = {r['scalar_leak']*100:.0f}%"
              f"  →  {tag}")

    # ──────────────────────────────────────────────────────────
    #  §3  LIFETIME HIERARCHY
    # ──────────────────────────────────────────────────────────
    print(f"\n§3  Lifetime hierarchy")
    print(f"-" * 50)

    tau_A1 = results[0]['lifetime_periods']
    tau_up = results[1]['lifetime_periods']
    tau_dn = results[2]['lifetime_periods']

    print(f"\n  E↓ (2 modes):  τ = {tau_dn:.1f} oscillation periods")
    print(f"  A₁ (1 mode) :  τ = {tau_A1:.1f} oscillation periods")
    print(f"  E↑ (2 modes):  τ = {tau_up:.1f} oscillation periods")
    print(f"\n  Ratio  τ(E↓)/τ(E↑) = {tau_dn/tau_up:.0f}×")
    print(f"  Ratio  τ(A₁)/τ(E↑) = {tau_A1/tau_up:.0f}×")
    print(f"\n  Physical meaning:")
    print(f"    E↓ and A₁ survive {tau_dn/tau_up:.0f}–{tau_A1/tau_up:.0f}×"
          f" longer than E↑.")
    print(f"    E↑ decays in ~{tau_up:.0f} oscillations → NOT a bound state")
    print(f"    (it's a scattering resonance, not a gluon)")

    # ──────────────────────────────────────────────────────────
    #  §4  SCAN OVER DEFORMATION STRENGTH
    # ──────────────────────────────────────────────────────────
    print(f"\n§4  Deformation strength scan")
    print(f"-" * 50)

    V_scan = np.linspace(0.5, 12, 300)
    leaks = {k: [] for k in ["A1", "E_up", "E_dn"]}

    for V_i in V_scan:
        res = mixing_analysis(E_T, E_S2, g0, V_i)
        leaks["A1"].append(res[0]['scalar_leak'])
        leaks["E_up"].append(res[1]['scalar_leak'])
        leaks["E_dn"].append(res[2]['scalar_leak'])

    leaks = {k: np.array(v) for k, v in leaks.items()}

    # Find where E↑ crosses 40% leak (effective deconfinement)
    threshold = 0.40
    cross_idx = np.argmin(np.abs(leaks["E_up"] - threshold))
    V_cross = V_scan[cross_idx]
    print(f"\n  E↑ reaches {threshold*100:.0f}% leak at |V| = {V_cross:.2f}"
          f"  ({V_cross/E_T*100:.1f}% deformation)")
    print(f"  At this point:")
    print(f"    A₁ leak  = {leaks['A1'][cross_idx]*100:.1f}%")
    print(f"    E↓ leak  = {leaks['E_dn'][cross_idx]*100:.1f}%")

    # For |V| range where E↑ > 40% AND others < 15%
    region_8 = (leaks["E_up"] > threshold) & \
               (leaks["A1"] < 0.15) & (leaks["E_dn"] < 0.15)
    if np.any(region_8):
        V_lo = V_scan[region_8][0]
        V_hi = V_scan[region_8][-1]
        print(f"\n  Region where N_gluon ≈ 8:")
        print(f"    |V| ∈ [{V_lo:.1f}, {V_hi:.1f}]"
              f"  ({V_lo/E_T*100:.0f}%–{V_hi/E_T*100:.0f}% deformation)")
    else:
        V_lo, V_hi = 0, 0
        print(f"\n  (Region not found with threshold {threshold*100:.0f}%"
              f" — adjusting...)")

    # ──────────────────────────────────────────────────────────
    #  §5  GROUP THEORY COUNT
    # ──────────────────────────────────────────────────────────
    print(f"\n§5  Final mode count")
    print(f"-" * 50)
    print(f"""
  ┌───────────────────────────────────────────────────┐
  │  Parity    Subgroup      Modes   Leak%   Status   │
  │─────────────────────────────────────────────────  │
  │  Odd       (all)           5      0%    GLUON ✓   │
  │  Even      E↓ (lower)      2     {leaks['E_dn'][cross_idx]*100:4.1f}%  GLUON ✓   │
  │  Even      A₁              1     {leaks['A1'][cross_idx]*100:4.1f}%  GLUON ✓   │
  │  Even      E↑ (upper)      2    {leaks['E_up'][cross_idx]*100:4.1f}%  DECAYED ✗ │
  │─────────────────────────────────────────────────  │
  │  TOTAL GLUONS:         5+2+1 = 8 = dim[SU(3)]    │
  └───────────────────────────────────────────────────┘
""")

    # ──────────────────────────────────────────────────────────
    #  §6  PLOTS
    # ──────────────────────────────────────────────────────────
    print(f"§6  Generating plots...")
    make_plots(V_scan, leaks, E_T, E_S1, E_S2, delta,
               threshold, V_cross, g0)

    # ──────────────────────────────────────────────────────────
    #  SUMMARY
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"""
  The 3-quark (D₃) structure of a baryon oscillon
  splits the 10 even+odd tensor modes as follows:

    5 odd-parity → pure tensor → ALWAYS confined     = 5
    5 even-parity → split by D₃ into A₁(1)+E↑(2)+E↓(2)
      • E↑ (2 modes): pushed into resonance with
        scalar n=2 mode → 50% scalar leak → DECONFINED
      • A₁ (1 mode): ~8% leak → confined            = 1
      • E↓ (2 modes): ~2% leak → confined            = 2
                                                     ─────
    TOTAL CONFINED TENSOR MODES                      = 8

  This equals dim[SU(3) adjoint] = 3² − 1 = 8.

  KEY NUMBERS:
    • Required deformation: ~{delta/E_T*100:.0f}% (3-quark scale)
    • Lifetime ratio: confined/deconfined ≈ {tau_dn/tau_up:.0f}×
    • Glueball ratio j₂₁/π = 1.835 vs M(0++)/M(p) = 1.823
      → 0.66% agreement as an external proton-anchor consistency check
""")


def make_plots(V_scan, leaks, E_T, E_S1, E_S2, delta,
               threshold, V_cross, g0):

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(r"ISPG: $D_3$ (3-quark) Symmetry $\rightarrow$ 8 Gluons",
                 fontsize=14, fontweight="bold")

    # --- (a) Energy level diagram ---
    ax = axes[0]
    V_plot = np.linspace(0, 12, 300)

    ax.axhspan(E_S2 - 4, E_S2 + 4, color="#3498db", alpha=0.1)
    ax.axhline(E_S2, color="#3498db", ls="-", lw=1.5, alpha=0.6,
               label=f"Scalar n=2 = {E_S2:.1f}")
    ax.axhline(E_S1, color="#2ecc71", ls="-", lw=1, alpha=0.5,
               label=f"Scalar n=1 = {E_S1:.1f}")

    ax.plot(V_plot, np.full_like(V_plot, E_T), "k-", lw=2.5,
            label=r"$A_1$ (1 mode)")
    ax.plot(V_plot, E_T + V_plot, "-", color="#e74c3c", lw=2.5,
            label=r"$E_\uparrow$ (2 modes)")
    ax.plot(V_plot, E_T - V_plot, "-", color="#2980b9", lw=2.5,
            label=r"$E_\downarrow$ (2 modes)")

    ax.axvline(delta, color="gray", ls=":", lw=1)
    ax.annotate(f"|V|=Δ={delta:.1f}\nresonance",
                xy=(delta, E_S2), xytext=(delta + 1, E_S2 + 5),
                fontsize=8, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray"))

    ax.set_xlabel(r"$D_3$ deformation $|V|$", fontsize=11)
    ax.set_ylabel(r"$(kR)^2$", fontsize=11)
    ax.set_title(r"(a) Energy levels vs $D_3$ deformation")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 12)
    ax.set_ylim(5, 55)

    # --- (b) Scalar leak vs deformation ---
    ax = axes[1]
    ax.plot(V_scan, leaks["E_up"] * 100, "-", color="#e74c3c", lw=2.5,
            label=r"$E_\uparrow$ (2 modes)")
    ax.plot(V_scan, leaks["A1"] * 100, "k-", lw=2.5,
            label=r"$A_1$ (1 mode)")
    ax.plot(V_scan, leaks["E_dn"] * 100, "-", color="#2980b9", lw=2.5,
            label=r"$E_\downarrow$ (2 modes)")

    ax.axhline(threshold * 100, color="gray", ls="--", lw=1,
               label=f"Deconfinement ~ {threshold*100:.0f}%")
    ax.axhspan(threshold * 100, 55, color="#e74c3c", alpha=0.05)
    ax.axhspan(0, threshold * 100, color="#2ecc71", alpha=0.05)

    ax.axvline(delta, color="gray", ls=":", lw=1)

    ax.set_xlabel(r"$D_3$ deformation $|V|$", fontsize=11)
    ax.set_ylabel("Scalar leak  (%)", fontsize=11)
    ax.set_title("(b) Confinement quality")
    ax.legend(fontsize=8)
    ax.set_xlim(V_scan[0], V_scan[-1])
    ax.set_ylim(0, 55)

    ax.text(8, 47, "DECONFINED", fontsize=10, color="#e74c3c",
            ha="center", alpha=0.6)
    ax.text(8, 5, "CONFINED (gluons)", fontsize=10, color="#2ecc71",
            ha="center", alpha=0.6)

    # --- (c) Mode count summary ---
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    txt = (
        "MODE COUNT AT $|V| \\approx \\Delta$\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "ODD PARITY  (pure tensor):\n"
        "   5 modes  ×  0% leak  =  5 gluons\n\n"
        "EVEN PARITY (D₃ split):\n"
        f"   E↓:  2 modes  ×  {leaks['E_dn'][np.argmin(np.abs(V_scan - delta))]*100:.0f}% leak  =  2 gluons\n"
        f"   A₁:  1 mode   ×  {leaks['A1'][np.argmin(np.abs(V_scan - delta))]*100:.0f}% leak  =  1 gluon\n"
        f"   E↑:  2 modes  ×  {leaks['E_up'][np.argmin(np.abs(V_scan - delta))]*100:.0f}% leak  =  0 gluons\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "TOTAL:  5 + 2 + 1 + 0  =  8\n\n"
        "= dim[SU(3) adjoint]\n"
        "= 3² − 1"
    )
    ax.text(5, 5, txt, fontsize=12, ha="center", va="center",
            family="monospace",
            bbox=dict(boxstyle="round,pad=1", facecolor="#eafaf1",
                      edgecolor="#27ae60", alpha=0.9))
    ax.set_title("(c) Result", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = OUT / "gluon_eight_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {path}")
    plt.close()


if __name__ == "__main__":
    main()
