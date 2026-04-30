"""
Phase AQ: quantitative estimate of the occupied-branch count N_cell
===================================================================

Goal:
  Put numbers on the mature-branch requirement

      N_cell >> 1

  that enters the fluctuation correction

      delta_N ~ sqrt(A_vort (1 - A_vort) / N_cell).

Conservative counting model:
  1. Local-emitter count:
       count the baryonic emitters geometrically contained in one coherent
       Bessel cell of transverse radius r / j_{0,1}.

  2. Enclosed-tail overlap:
       in the outer halo, local baryonic density becomes tiny, but the cell is
       crossed by long resonant tails emitted by the enclosed disk.
       Spread those enclosed packets over a cylindrical shell area
       A_shell ~ 4 pi r h_d and count the fraction intercepted by one cell.

  3. Lower bound:
       N_cell >= max(N_local, N_tail).

Notes:
  - This is intentionally conservative: it counts only baryonic emitters, not
    all entrained fluid elements of the vortex.
  - Taking a packet mass larger than 1 M_sun only rescales N_cell downward
    linearly; the script reports that robustness too.
"""

from pathlib import Path
import io
import sys

import numpy as np
from scipy.special import jn_zeros
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from constants import G, Msun, Omega_tr_conj, a0, kpc


SEP = "=" * 78
OUTDIR = Path(__file__).parent / "plots"
OUTDIR.mkdir(exist_ok=True)

BESSEL_ZERO = jn_zeros(0, 1)[0]
XI_MIN = 0.3
XI_MAX = 10.0
M_PACKET_DEFAULT = 1.0 * Msun


def galaxy_cases():
    return [
        {"name": "Dwarf disk", "M_msun": 1.0e9, "R_d_kpc": 1.5, "h_d_kpc": 0.3},
        {"name": "LSB disk", "M_msun": 1.0e10, "R_d_kpc": 8.0, "h_d_kpc": 0.6},
        {"name": "MW-like disk", "M_msun": 1.0e11, "R_d_kpc": 10.0, "h_d_kpc": 0.5},
        {"name": "HSB massive", "M_msun": 3.0e11, "R_d_kpc": 6.0, "h_d_kpc": 0.6},
    ]


def case_to_physical(case):
    m_si = case["M_msun"] * Msun
    r_d = case["R_d_kpc"] * kpc
    h_d = case["h_d_kpc"] * kpc
    r_m = np.sqrt(G * m_si / a0)
    eta = r_m / r_d
    return {
        "name": case["name"],
        "M": m_si,
        "R_d": r_d,
        "h_d": h_d,
        "r_M": r_m,
        "eta": eta,
    }


def enclosed_mass_fraction(xi, eta):
    xi = np.asarray(xi, dtype=float)
    y = eta * xi
    return 1.0 - (1.0 + y) * np.exp(-y)


def g_newton_dimless(xi, eta):
    xi = np.asarray(xi, dtype=float)
    out = np.zeros_like(xi)
    mask = xi > 0.0
    out[mask] = enclosed_mass_fraction(xi[mask], eta) / xi[mask] ** 2
    return out


def g_total_from_simple_mu(g_n):
    g_n = np.asarray(g_n, dtype=float)
    return 0.5 * (g_n + np.sqrt(g_n**2 + 4.0 * a0 * g_n))


def omega_orb(g, r):
    g = np.asarray(g, dtype=float)
    r = np.asarray(r, dtype=float)
    return np.sqrt(np.maximum(g / r, 0.0))


def a_vort(omega):
    omega = np.asarray(omega, dtype=float)
    return omega / (omega + Omega_tr_conj)


def bessel_cell_radius(r):
    r = np.asarray(r, dtype=float)
    return r / BESSEL_ZERO


def bessel_cell_area(r):
    rc = bessel_cell_radius(r)
    return np.pi * rc**2


def bessel_cell_volume(r, h_d):
    return bessel_cell_area(r) * (2.0 * h_d)


def surface_density(r, mass_total, r_d):
    sigma0 = mass_total / (2.0 * np.pi * r_d**2)
    return sigma0 * np.exp(-r / r_d)


def midplane_number_density(r, mass_total, r_d, h_d, m_packet=M_PACKET_DEFAULT):
    return surface_density(r, mass_total, r_d) / np.maximum(2.0 * h_d * m_packet, 1e-300)


def local_emitter_count(r, mass_total, r_d, h_d, m_packet=M_PACKET_DEFAULT):
    return (
        midplane_number_density(r, mass_total, r_d, h_d, m_packet=m_packet)
        * bessel_cell_volume(r, h_d)
    )


def enclosed_tail_overlap_count(xi, mass_total, r_d, h_d, r_m, eta, m_packet=M_PACKET_DEFAULT):
    xi = np.asarray(xi, dtype=float)
    r = xi * r_m
    n_enclosed = mass_total * enclosed_mass_fraction(xi, eta) / np.maximum(m_packet, 1e-300)
    shell_area = 4.0 * np.pi * r * h_d
    return n_enclosed * bessel_cell_area(r) / np.maximum(shell_area, 1e-300)


def conservative_ncell(xi, mass_total, r_d, h_d, r_m, eta, m_packet=M_PACKET_DEFAULT):
    r = np.asarray(xi, dtype=float) * r_m
    n_local = local_emitter_count(r, mass_total, r_d, h_d, m_packet=m_packet)
    n_tail = enclosed_tail_overlap_count(xi, mass_total, r_d, h_d, r_m, eta, m_packet=m_packet)
    return n_local, n_tail, np.maximum(n_local, n_tail)


def delta_n(a_here, n_cell):
    a_here = np.asarray(a_here, dtype=float)
    n_cell = np.asarray(n_cell, dtype=float)
    return np.sqrt(np.maximum(a_here * (1.0 - a_here), 0.0) / np.maximum(n_cell, 1e-300))


def solve_case(case, xi_grid, m_packet=M_PACKET_DEFAULT):
    phys = case_to_physical(case)
    r = xi_grid * phys["r_M"]
    g_n = a0 * g_newton_dimless(xi_grid, phys["eta"])
    g_tot = g_total_from_simple_mu(g_n)
    a_branch = a_vort(omega_orb(g_tot, r))
    n_local, n_tail, n_lower = conservative_ncell(
        xi_grid,
        phys["M"],
        phys["R_d"],
        phys["h_d"],
        phys["r_M"],
        phys["eta"],
        m_packet=m_packet,
    )
    return {
        **phys,
        "xi": xi_grid,
        "r": r,
        "A_vort": a_branch,
        "N_local": n_local,
        "N_tail": n_tail,
        "N_lower": n_lower,
        "delta_upper": 0.5 / np.sqrt(np.maximum(n_lower, 1e-300)),
        "delta_mature": delta_n(a_branch, n_lower),
    }


def print_case_table(results, xi_samples):
    print(f"  Case: {results['name']}")
    print(
        f"    M = {results['M']/Msun:.2e} Msun, "
        f"R_d = {results['R_d']/kpc:.2f} kpc, "
        f"h_d = {results['h_d']/kpc:.2f} kpc, "
        f"r_M = {results['r_M']/kpc:.2f} kpc, "
        f"eta = {results['eta']:.3f}"
    )
    print(
        "    xi      N_local        N_tail        N_lower        "
        "delta_upper    delta_mature"
    )
    for xi_here in xi_samples:
        idx = int(np.argmin(np.abs(results["xi"] - xi_here)))
        print(
            f"    {results['xi'][idx]:4.1f}   "
            f"{results['N_local'][idx]:11.3e}   "
            f"{results['N_tail'][idx]:11.3e}   "
            f"{results['N_lower'][idx]:11.3e}   "
            f"{results['delta_upper'][idx]:11.3e}   "
            f"{results['delta_mature'][idx]:12.3e}"
        )
    idx_min = int(np.argmin(results["N_lower"]))
    print(
        f"    min N_lower over {XI_MIN:.1f} <= xi <= {XI_MAX:.1f}: "
        f"{results['N_lower'][idx_min]:.3e} at xi={results['xi'][idx_min]:.3f}"
    )
    print(
        f"    max delta_upper over same range: "
        f"{np.max(results['delta_upper']):.3e}"
    )
    print(
        f"    max delta_mature over same range: "
        f"{np.max(results['delta_mature']):.3e}"
    )
    print()


def print_mass_robustness(cases):
    xi_grid = np.geomspace(XI_MIN, XI_MAX, 240)
    print("  Robustness against larger effective packet masses:")
    print("    m_packet/Msun    min N_lower across all cases")
    for m_packet_msun in [1.0, 10.0, 100.0]:
        minima = []
        for case in cases:
            res = solve_case(case, xi_grid, m_packet=m_packet_msun * Msun)
            minima.append(np.min(res["N_lower"]))
        print(f"    {m_packet_msun:12.1f}    {min(minima):18.3e}")
    print()


def make_plots(results_all):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for res in results_all:
        ax.loglog(res["xi"], res["N_local"], lw=2, label=res["name"])
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$N_{\rm local}$")
    ax.set_title("(a) Local emitters in one Bessel cell")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for res in results_all:
        ax.loglog(res["xi"], res["N_tail"], lw=2, label=res["name"])
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$N_{\rm tail}$")
    ax.set_title("(b) Enclosed-tail overlap count")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for res in results_all:
        ax.loglog(res["xi"], res["N_lower"], lw=2, label=res["name"])
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$N_{\rm cell,lower}$")
    ax.set_title("(c) Conservative occupied-branch count")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for res in results_all:
        ax.loglog(res["xi"], res["delta_upper"], lw=2, label=res["name"] + " (worst)")
        ax.loglog(res["xi"], res["delta_mature"], "--", lw=1.5, label=res["name"] + " (mature)")
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$\delta_N$")
    ax.set_title("(d) Finite-ensemble fluctuation bound")
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Quantitative Occupied-Branch Count Estimate", y=0.99)
    fig.tight_layout()
    outpath = OUTDIR / "phase_aq_ncell_estimate.png"
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outpath


def run_all():
    xi_grid = np.geomspace(XI_MIN, XI_MAX, 240)
    xi_samples = np.array([0.3, 1.0, 3.0, 10.0], dtype=float)
    cases = galaxy_cases()
    results_all = [solve_case(case, xi_grid) for case in cases]
    plot_path = make_plots(results_all)

    print(SEP)
    print("  PHASE AQ: Quantitative N_cell Estimate")
    print(SEP)
    print(
        "  Model choices:\n"
        "  - Bessel-cell radius: r / j_{0,1}\n"
        "  - local count: emitters geometrically inside one cell\n"
        "  - outer-halo count: enclosed emitters whose tails cross one local cell\n"
        "  - effective packet mass = 1 Msun (conservative stellar-scale choice)\n"
    )

    for res in results_all:
        print_case_table(res, xi_samples)

    global_min = min(np.min(res["N_lower"]) for res in results_all)
    global_delta = max(np.max(res["delta_upper"]) for res in results_all)
    global_delta_mature = max(np.max(res["delta_mature"]) for res in results_all)
    print_mass_robustness(cases)

    print("  Summary:")
    print(f"  - global minimum N_cell,lower  = {global_min:.3e}")
    print(f"  - global maximum delta_upper   = {global_delta:.3e}")
    print(f"  - global maximum delta_mature  = {global_delta_mature:.3e}")
    print(
        "\n  Reading:\n"
        "  - even the conservative lower bound stays far above unity across the\n"
        "    scored galactic window 0.3 <= xi <= 10;\n"
        "  - the local-emitter count is already enormous in ordinary disks;\n"
        "  - where the local baryonic density drops, enclosed resonant-tail overlap\n"
        "    keeps the occupation number huge;\n"
        "  - therefore the finite-ensemble correction delta_N is tiny, and the\n"
        "    mature occupied-branch assumption is quantitatively justified.\n"
    )
    print(f"  Plot saved to: {plot_path}")

    return {
        "results_all": results_all,
        "plot_path": plot_path,
        "global_min": global_min,
        "global_delta": global_delta,
        "global_delta_mature": global_delta_mature,
    }


if __name__ == "__main__":
    run_all()
