"""
Phase AN: blind universality scan for the self-calibrated nonlocal 2D kernel
=============================================================================

Goal:
  Complete the "sensitivity / universality" stage for the blind 2D MOND solve.

Method:
  - vary the physical galaxy parameters (M_gal, R_d, h_d),
  - convert them into the dimensionless disk shape parameters
        eta = r_M / R_d,   q = h_d / r_M ,
  - rebuild the blind self-calibrated source-side 2D kernel for each case,
  - extract mu_blind = g_N / g_eff,
  - compare against the universal MOND law mu = x / (1 + x),
  - and measure the cross-case collapse of mu(x).

Important:
  The source construction remains blind in every case: the baryonic
  normalization is fixed only by the far-field Newtonian Gauss law, and the
  halo prefactor is fixed only by the outer nonlocal asymptotic tail.
"""

from pathlib import Path
import io
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse.linalg import factorized

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from constants import G, Msun, a0, kpc
from phase_al_blind_selfcalibrated_kernel_2d import gauss_normalize_newtonian
from phase_u_axisymmetric_swirl_2d import (
    build_axisymmetric_operator,
    grad_xi,
    solve_linear_poisson,
)
from phase_ai_boundary_kernel_2d import (
    RHO_CORE_DERIVED,
    boundary_selected_kappa,
    build_tail_source,
    enclosed_mass_lookup,
    interior_residual,
)


SEP = "=" * 78
OUTDIR = Path(__file__).parent / "plots"
OUTDIR.mkdir(exist_ok=True)

XI_MIN = 0.3
XI_MAX = 10.0


def galaxy_cases():
    return [
        {"name": "Dwarf disk", "M_msun": 1.0e9, "R_d_kpc": 1.5, "h_d_kpc": 0.3},
        {"name": "LSB disk", "M_msun": 1.0e10, "R_d_kpc": 8.0, "h_d_kpc": 0.6},
        {"name": "MW-like disk", "M_msun": 1.0e11, "R_d_kpc": 10.0, "h_d_kpc": 0.5},
        {"name": "HSB massive", "M_msun": 3.0e11, "R_d_kpc": 6.0, "h_d_kpc": 0.6},
    ]


def derived_shape(case):
    m_si = case["M_msun"] * Msun
    r_d = case["R_d_kpc"] * kpc
    h_d = case["h_d_kpc"] * kpc
    r_m = np.sqrt(G * m_si / a0)
    eta = r_m / r_d
    h_ratio = h_d / r_m
    return {
        "r_M_kpc": r_m / kpc,
        "eta": eta,
        "h_ratio": h_ratio,
    }


def f_source_3d_param(xi, zeta, eta, h_ratio):
    xi = np.asarray(xi, dtype=float)
    zeta = np.asarray(zeta, dtype=float)
    h_safe = max(float(h_ratio), 1e-6)
    return eta**2 * np.exp(-eta * xi) * (1.0 / (2.0 * h_safe)) * np.exp(-np.abs(zeta) / h_safe)


def solve_case(
    eta,
    h_ratio,
    nr=121,
    nz=91,
    xi_max=20.0,
    zeta_max=10.0,
    rho_core=RHO_CORE_DERIVED,
    newton_fit_min=5.0,
    newton_fit_max=10.0,
    source_fit_min=10.0,
    source_fit_max=15.0,
):
    xi, zeta, mat = build_axisymmetric_operator(
        nr=nr, nz=nz, xi_max=xi_max, zeta_max=zeta_max
    )
    solver = factorized(mat)
    dxi = xi[1] - xi[0]

    xi_grid, zeta_grid = np.meshgrid(xi, zeta, indexing="ij")
    unit_source = f_source_3d_param(xi_grid, zeta_grid, eta=eta, h_ratio=h_ratio)

    u_unit = solve_linear_poisson(solver, unit_source, nr, nz)
    g_mid_unit = -grad_xi(u_unit, dxi)[:, 0]
    scale_n, g_point_target, fit_n_min, fit_n_max = gauss_normalize_newtonian(
        xi, g_mid_unit, fit_min=newton_fit_min, fit_max=newton_fit_max
    )

    f_bary = scale_n * unit_source
    g_n = scale_n * g_mid_unit

    rho_sorted, cumulative = enclosed_mass_lookup(xi, zeta, f_bary)
    base_source, m_enc_2d, _ = build_tail_source(
        xi, zeta, rho_core, rho_sorted, cumulative
    )

    u_base = solve_linear_poisson(solver, base_source, nr, nz)
    g_h_base = -grad_xi(u_base, dxi)[:, 0]
    m_enc_mid = m_enc_2d[:, 0]
    kappa_g, g_outer_target, fit_h_min, fit_h_max = boundary_selected_kappa(
        g_h_base, xi, m_enc_mid, fit_min=source_fit_min, fit_max=source_fit_max
    )

    source_extra = kappa_g * base_source
    source_total = f_bary + source_extra
    u_total = solve_linear_poisson(solver, source_total, nr, nz)
    g_eff = -grad_xi(u_total, dxi)[:, 0]

    mu_blind = np.divide(g_n, g_eff, out=np.ones_like(g_eff), where=g_eff > 1e-20)
    mu_mond = np.divide(g_eff, 1.0 + g_eff, out=np.zeros_like(g_eff), where=g_eff > 0)

    mask = (xi >= XI_MIN) & (xi <= XI_MAX)
    mu_rms = float(np.sqrt(np.mean((mu_blind[mask] - mu_mond[mask]) ** 2)))
    field_rms = float(
        np.sqrt(np.mean(((g_eff[mask] - (0.5 * (g_n[mask] + np.sqrt(g_n[mask] ** 2 + 4.0 * g_n[mask]))))
                         / np.maximum(0.5 * (g_n[mask] + np.sqrt(g_n[mask] ** 2 + 4.0 * g_n[mask])), 1e-30)) ** 2))
    )

    return {
        "xi": xi,
        "g_N": g_n,
        "g_eff": g_eff,
        "mu_blind": mu_blind,
        "mu_mond": mu_mond,
        "scale_N": scale_n,
        "kappa_G": kappa_g,
        "eta": eta,
        "h_ratio": h_ratio,
        "newton_fit_min": fit_n_min,
        "newton_fit_max": fit_n_max,
        "source_fit_min": fit_h_min,
        "source_fit_max": fit_h_max,
        "g_point_target": g_point_target,
        "g_outer_target": g_outer_target,
        "mu_rms": mu_rms,
        "field_rms": field_rms,
        "pde_residual": interior_residual(mat, u_total, source_total, nr, nz),
        "extra_pos": float(np.mean(source_extra[mask, :] >= -1e-12)),
        "total_pos": float(np.mean(source_total[mask, :] >= -1e-12)),
    }


def common_mu_collapse(case_results):
    curves = []
    xmins = []
    xmaxs = []
    for res in case_results:
        mask = (res["xi"] >= XI_MIN) & (res["xi"] <= XI_MAX)
        x = np.asarray(res["g_eff"][mask], dtype=float)
        mu = np.asarray(res["mu_blind"][mask], dtype=float)
        order = np.argsort(x)
        x_sorted = x[order]
        mu_sorted = mu[order]
        x_unique, idx = np.unique(x_sorted, return_index=True)
        mu_unique = mu_sorted[idx]
        curves.append((x_unique, mu_unique))
        xmins.append(np.min(x_unique))
        xmaxs.append(np.max(x_unique))

    x_common_min = max(xmins)
    x_common_max = min(xmaxs)
    x_common = np.geomspace(max(x_common_min, 1e-4), x_common_max, 160)

    mu_stack = []
    for x_curve, mu_curve in curves:
        mu_interp = np.interp(x_common, x_curve, mu_curve)
        mu_stack.append(mu_interp)
    mu_stack = np.asarray(mu_stack)

    mu_mean = np.mean(mu_stack, axis=0)
    spread_rms = float(np.sqrt(np.mean((mu_stack - mu_mean) ** 2)))
    target = x_common / (1.0 + x_common)
    target_rms = float(np.sqrt(np.mean((mu_mean - target) ** 2)))

    return {
        "x_common": x_common,
        "mu_stack": mu_stack,
        "mu_mean": mu_mean,
        "spread_rms": spread_rms,
        "target_rms": target_rms,
    }


def make_plots(cases, results, collapse):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for case, res in zip(cases, results):
        ax.semilogx(res["xi"], res["mu_blind"], lw=2, label=case["name"])
    ax.semilogx(results[0]["xi"], results[0]["mu_mond"], "k--", lw=1.5, label=r"$x/(1+x)$")
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$\mu$")
    ax.set_title(r"(a) Extracted blind $\mu(\xi)$")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    x_plot = np.geomspace(1e-3, 3.0, 300)
    ax.semilogx(x_plot, x_plot / (1.0 + x_plot), "k--", lw=1.5, label=r"$x/(1+x)$")
    for case, res in zip(cases, results):
        mask = (res["xi"] >= XI_MIN) & (res["xi"] <= XI_MAX)
        ax.semilogx(res["g_eff"][mask], res["mu_blind"][mask], lw=2, label=case["name"])
    ax.set_xlabel(r"$x = g/a_0$")
    ax.set_ylabel(r"$\mu$")
    ax.set_title(r"(b) Universality collapse $\mu(x)$")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    for case, res in zip(cases, results):
        mask = (res["xi"] >= XI_MIN) & (res["xi"] <= XI_MAX)
        delta = res["mu_blind"][mask] - res["mu_mond"][mask]
        ax.semilogx(res["g_eff"][mask], delta, lw=2, label=case["name"])
    ax.axhline(0.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel(r"$x = g/a_0$")
    ax.set_ylabel(r"$\Delta \mu$")
    ax.set_title(r"(c) Deviation from $x/(1+x)$")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    target = collapse["x_common"] / (1.0 + collapse["x_common"])
    for idx, case in enumerate(cases):
        ax.semilogx(
            collapse["x_common"],
            collapse["mu_stack"][idx],
            lw=1.5,
            alpha=0.85,
            label=case["name"],
        )
    ax.semilogx(collapse["x_common"], collapse["mu_mean"], color="tab:red", lw=2.5, label="mean collapse")
    ax.semilogx(collapse["x_common"], target, "k--", lw=1.5, label=r"$x/(1+x)$")
    ax.set_xlabel(r"$x = g/a_0$")
    ax.set_ylabel(r"$\mu$")
    ax.set_title("(d) Common-x collapse")
    ax.legend(fontsize=9)

    fig.suptitle("Blind Universality Scan for the Self-Calibrated Nonlocal 2D Kernel", y=0.98)
    fig.tight_layout()
    outpath = OUTDIR / "phase_an_blind_universality_scan.png"
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outpath


def run_scan():
    cases = galaxy_cases()
    results = []

    print(SEP)
    print("  PHASE AN: Blind Universality Scan")
    print(SEP)

    for case in cases:
        shape = derived_shape(case)
        res = solve_case(shape["eta"], shape["h_ratio"])
        res["shape"] = shape
        results.append(res)

    collapse = common_mu_collapse(results)
    plot_path = make_plots(cases, results, collapse)

    print()
    print(
        "  "
        f"{'Case':<14s} {'eta':>7s} {'h/r_M':>8s} {'mu rms':>10s} "
        f"{'field rms':>10s} {'kappa_G':>9s} {'scale_N':>9s}"
    )
    print("  " + "-" * 86)
    for case, res in zip(cases, results):
        print(
            "  "
            f"{case['name']:<14s} "
            f"{res['shape']['eta']:7.3f} "
            f"{res['shape']['h_ratio']:8.4f} "
            f"{res['mu_rms']:10.3e} "
            f"{res['field_rms']:10.3e} "
            f"{res['kappa_G']:9.3f} "
            f"{res['scale_N']:9.3f}"
        )

    print()
    print(f"  Cross-case collapse RMS around mean mu(x): {collapse['spread_rms']:.3e}")
    print(f"  Mean collapsed curve RMS vs x/(1+x):      {collapse['target_rms']:.3e}")
    print(f"  Plot saved to: {plot_path}")

    print(
        """

  Reading:
  - if the cross-case spread stays well below the per-case mismatch to the
    target law, then the extracted blind mu(x) is genuinely universal and the
    residual error is mostly a common geometry effect;
  - that is exactly the question this scan is designed to answer.
        """
    )

    return {
        "cases": cases,
        "results": results,
        "collapse": collapse,
        "plot_path": plot_path,
    }


if __name__ == "__main__":
    run_scan()
