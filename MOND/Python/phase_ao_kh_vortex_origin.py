"""
Phase AO: KH growth and inertial-range estimate for cosmological vortex origin
==============================================================================

Goal:
  Quantify the "flows -> vortices" claim in the MOND vortex-origin section.

Checks:
  1. Kelvin-Helmholtz growth time:
         gamma_KH = k * DeltaV / 2 = pi * DeltaV / lambda
  2. Available formation window including the proper-time correction from the
     background-pressure history.
  3. Conservative Reynolds-number and Kolmogorov-cutoff estimate showing that
     10-100 kpc lies inside the inertial range.
"""

from pathlib import Path
import io
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from constants import Gyr, H0, kpc
from multiscale import H_of_z, clock_rate_ratio_from_quantum, dw0_from_master_formula


SEP = "=" * 78
OUTDIR = Path(__file__).parent / "plots"
OUTDIR.mkdir(exist_ok=True)
PDF_DIR = Path(__file__).parent.parent / "PDF"
PDF_DIR.mkdir(exist_ok=True)

DELTA_V = 600.0e3
L_DRIVING = 1.0e3 * kpc
R_VORTEX = 10.0 * kpc
ALPHA_Q = 1.0
K_MASTER = 8.0


def kh_growth_rate(length_m, delta_v=DELTA_V):
    length_m = np.asarray(length_m, dtype=float)
    return np.pi * delta_v / np.maximum(length_m, 1e-30)


def efold_time(length_m, delta_v=DELTA_V):
    return 1.0 / kh_growth_rate(length_m, delta_v=delta_v)


def formation_window(z_form):
    z = np.linspace(0.0, float(z_form), 5000)
    dt_dz = 1.0 / ((1.0 + z) * H_of_z(z))
    coord_time = np.trapz(dt_dz, z)
    dw0 = dw0_from_master_formula(z_form, alpha=ALPHA_Q, K=K_MASTER)
    clock_ratio = clock_rate_ratio_from_quantum(z, ALPHA_Q, dw0)
    proper_time = np.trapz(clock_ratio * dt_dz, z)
    return {
        "z_form": z_form,
        "dw0": dw0,
        "coord_time_s": coord_time,
        "proper_time_s": proper_time,
        "boost": proper_time / max(coord_time, 1e-30),
    }


def viscosity_upper_bound(radius=R_VORTEX):
    t_h = 1.0 / H0
    return radius**2 / t_h


def kolmogorov_cutoff(length_driving=L_DRIVING, delta_v=DELTA_V, nu=None):
    if nu is None:
        nu = viscosity_upper_bound()
    reynolds = length_driving * delta_v / max(nu, 1e-30)
    l_diss = length_driving * reynolds ** (-0.75)
    return reynolds, l_diss


def make_plots(sample_lengths, windows, re_min, l_diss):
    sample_lengths = np.asarray(sample_lengths, dtype=float)
    lambda_grid = np.geomspace(10.0 * kpc, 1.0e3 * kpc, 400)
    t_grid_gyr = efold_time(lambda_grid) / Gyr

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ax.loglog(lambda_grid / kpc, t_grid_gyr, lw=2, label=r"$t_{\rm e-fold}=\lambda/(\pi \Delta V)$")
    for window in windows:
        ax.axhline(
            window["proper_time_s"] / Gyr,
            lw=1.5,
            ls="--",
            label=rf"$\tau_{{\rm form}}(z_f={window['z_form']})$",
        )
    for lam in sample_lengths:
        ax.axvline(lam / kpc, color="gray", ls=":", lw=0.8)
    ax.set_xlabel(r"$\lambda$ (kpc)")
    ax.set_ylabel("time (Gyr)")
    ax.set_title("KH growth vs available process-time")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.axvspan(l_diss / kpc, L_DRIVING / kpc, color="tab:blue", alpha=0.18, label="inertial range")
    for lam in [10.0 * kpc, 30.0 * kpc, 100.0 * kpc]:
        ax.axvline(lam / kpc, color="tab:red", lw=1.3)
    ax.axvline(l_diss / kpc, color="k", ls="--", lw=1.4, label=rf"$\ell_{{\rm diss}}\approx {l_diss/kpc:.2f}$ kpc")
    ax.axvline(L_DRIVING / kpc, color="k", ls=":", lw=1.0, label=r"$L_{\rm driving}=1$ Mpc")
    ax.set_xscale("log")
    ax.set_xlim(1e-2, 2e3)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("scale (kpc)")
    ax.set_title(rf"Conservative cascade window (${{\rm Re}}_{{\min}}\approx {re_min:.2e}$)")
    ax.legend(fontsize=9, loc="upper left")

    fig.suptitle("Quantitative Vortex-Origin Diagnostics", y=0.99)
    fig.tight_layout()
    outpath = OUTDIR / "phase_ao_kh_vortex_origin.png"
    pdf_path = PDF_DIR / "fig_mond_vortex_origin.pdf"
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return outpath


def run_all():
    sample_lengths = np.array([10.0, 30.0, 100.0, 300.0, 1000.0]) * kpc
    windows = [formation_window(2.0), formation_window(10.0)]
    nu_max = viscosity_upper_bound()
    re_min, l_diss = kolmogorov_cutoff(nu=nu_max)
    plot_path = make_plots(sample_lengths, windows, re_min, l_diss)

    print(SEP)
    print("  PHASE AO: KH Growth and Vortex-Origin Quantification")
    print(SEP)
    print(f"  adopted shear speed DeltaV = {DELTA_V/1e3:.0f} km/s")
    print(f"  adopted driving scale      = {L_DRIVING/kpc:.0f} kpc")
    print(f"  conservative nu_max        = {nu_max:.3e} m^2/s")
    print(f"  conservative Re_min        = {re_min:.3e}")
    print(f"  Kolmogorov cutoff l_diss   = {l_diss/kpc:.3f} kpc")
    print()
    print("  KH growth times:")
    print(f"  {'lambda (kpc)':>12s}  {'gamma_KH (1/s)':>16s}  {'t_e-fold (Myr)':>16s}")
    print("  " + "-" * 52)
    for lam in sample_lengths:
        gamma = kh_growth_rate(lam)
        te_myr = efold_time(lam) / (1e6 * 3.15576e7)
        print(f"  {lam/kpc:12.1f}  {gamma:16.3e}  {te_myr:16.3f}")
    print()
    print("  Formation windows with proper-time weighting:")
    print(f"  {'z_f':>6s}  {'coord (Gyr)':>12s}  {'proper (Gyr)':>13s}  {'boost':>8s}  {'N_e(100 kpc)':>13s}  {'N_e(1 Mpc)':>11s}")
    print("  " + "-" * 78)
    te_100 = efold_time(100.0 * kpc)
    te_1000 = efold_time(1000.0 * kpc)
    for window in windows:
        n100 = window["proper_time_s"] / te_100
        n1000 = window["proper_time_s"] / te_1000
        print(
            f"  {window['z_form']:6.1f}  "
            f"{window['coord_time_s']/Gyr:12.3f}  "
            f"{window['proper_time_s']/Gyr:13.3f}  "
            f"{window['boost']:8.3f}  "
            f"{n100:13.1f}  "
            f"{n1000:11.1f}"
        )
    print()
    print("  Reading:")
    print("  - even on the 1 Mpc driving scale, KH growth is sub-Gyr;")
    print("  - the proper-time-weighted formation window supplies tens of e-foldings")
    print("    on 1 Mpc and hundreds on 100 kpc;")
    print("  - with the conservative viscosity ceiling from vortex persistence, the")
    print("    cascade reaches down to ~0.2 kpc, so 10-100 kpc lies safely inside")
    print("    the inertial range.")
    print(f"\n  Plot saved to: {plot_path}")

    return {
        "sample_lengths": sample_lengths,
        "windows": windows,
        "nu_max": nu_max,
        "re_min": re_min,
        "l_diss": l_diss,
        "plot_path": plot_path,
    }


if __name__ == "__main__":
    run_all()
