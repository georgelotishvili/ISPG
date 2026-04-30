"""
Phase AR: K_sigma for pressure-supported Sersic systems
======================================================

Goal:
  Quantify the normalization

      sigma^4 = K_sigma G M a0

  for spherical isotropic pressure-supported systems with Sersic-like
  structure.

Method:
  1. Build a deprojected Sersic density using the Prugniel-Simien form.
  2. Solve the isotropic Jeans equation in the deep-MOND limit
         g(r) = sqrt(G a0 M_enc(r)) / r .
  3. Project to line-of-sight velocity dispersion.
  4. Measure aperture dispersions at:
       - R_e / 8   (classical elliptical-galaxy convention),
       - R_e       (half-light aperture),
       - infinity  (global projected aperture).

Important interpretation:
  The exact deep-MOND virial theorem fixes the global rms scale.  The
  profile dependence of K_sigma enters when translating that global scale
  into an observed projected aperture dispersion.
"""

from pathlib import Path
import io
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.special import gammaincinv


if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


SEP = "=" * 78
OUTDIR = Path(__file__).parent / "plots"
OUTDIR.mkdir(exist_ok=True)

R_MIN = 1.0e-4
R_MAX = 1.0e3
N_R = 5000
N_PROJ = 600
N_Z = 2600


def sersic_bn(n):
    return float(gammaincinv(2.0 * n, 0.5))


def prugniel_p(n):
    n = float(n)
    return 1.0 - 0.6097 / n + 0.05463 / (n**2)


def density_shape(r, n):
    b_n = sersic_bn(n)
    p_n = prugniel_p(n)
    return np.maximum(r, 1e-300) ** (-p_n) * np.exp(-b_n * np.maximum(r, 1e-300) ** (1.0 / n))


def normalize_density(r, rho_shape):
    mass_raw = 4.0 * np.pi * np.trapz(rho_shape * r**2, r)
    return rho_shape / max(mass_raw, 1e-300)


def cumulative_mass(r, rho):
    dmass = 4.0 * np.pi * rho * r**2
    m_enc = cumulative_trapezoid(dmass, r, initial=0.0)
    return m_enc / max(m_enc[-1], 1e-300)


def jeans_sigma_r2(r, rho, m_enc):
    integrand = rho * np.sqrt(np.maximum(m_enc, 0.0)) / np.maximum(r, 1e-300)
    tail_int = -cumulative_trapezoid(integrand[::-1], r[::-1], initial=0.0)[::-1]
    return tail_int / np.maximum(rho, 1e-300)


def interp_log(x_src, y_src, x_new):
    lx = np.log(np.maximum(x_src, 1e-300))
    lx_new = np.log(np.maximum(x_new, 1e-300))
    return np.interp(lx_new, lx, y_src)


def projected_profiles(r, rho, sigma_r2):
    r_proj = np.geomspace(1.0e-3, 1.0e2, N_PROJ)
    z_grid = np.concatenate(([0.0], np.geomspace(1.0e-6, 1.0e3, N_Z - 1)))

    sigma_proj = np.zeros_like(r_proj)
    second_proj = np.zeros_like(r_proj)

    for i, R in enumerate(r_proj):
        rr = np.sqrt(R**2 + z_grid**2)
        rho_line = interp_log(r, rho, rr)
        sig_line = interp_log(r, sigma_r2, rr)
        sigma_proj[i] = 2.0 * np.trapz(rho_line, z_grid)
        second_proj[i] = 2.0 * np.trapz(rho_line * sig_line, z_grid)

    sigma_los2 = second_proj / np.maximum(sigma_proj, 1e-300)
    mass2d = cumulative_trapezoid(2.0 * np.pi * sigma_proj * r_proj, r_proj, initial=0.0)
    mass2d /= max(mass2d[-1], 1e-300)

    return {
        "R": r_proj,
        "Sigma": sigma_proj,
        "Sigma_sigma2": second_proj,
        "sigma_los2": sigma_los2,
        "mass2d": mass2d,
    }


def radius_at_fraction(r, mfrac, target=0.5):
    return float(np.interp(target, mfrac, r))


def aperture_sigma2(R, Sigma, Sigma_sigma2, R_ap):
    mask = R <= R_ap
    num = np.trapz(2.0 * np.pi * R[mask] * Sigma_sigma2[mask], R[mask])
    den = np.trapz(2.0 * np.pi * R[mask] * Sigma[mask], R[mask])
    return num / max(den, 1e-300)


def global_rms_sigma2(r, rho, sigma_r2):
    return np.trapz(4.0 * np.pi * rho * sigma_r2 * r**2, r)


def solve_case(n):
    r = np.geomspace(R_MIN, R_MAX, N_R)
    rho = normalize_density(r, density_shape(r, n))
    m_enc = cumulative_mass(r, rho)
    sigma_r2 = jeans_sigma_r2(r, rho, m_enc)
    proj = projected_profiles(r, rho, sigma_r2)
    r_e = radius_at_fraction(proj["R"], proj["mass2d"], target=0.5)

    sigma2_global = global_rms_sigma2(r, rho, sigma_r2)
    sigma2_e8 = aperture_sigma2(proj["R"], proj["Sigma"], proj["Sigma_sigma2"], r_e / 8.0)
    sigma2_e = aperture_sigma2(proj["R"], proj["Sigma"], proj["Sigma_sigma2"], r_e)
    sigma2_inf = aperture_sigma2(proj["R"], proj["Sigma"], proj["Sigma_sigma2"], proj["R"][-1])

    return {
        "n": float(n),
        "r": r,
        "rho": rho,
        "m_enc": m_enc,
        "sigma_r2": sigma_r2,
        **proj,
        "R_e": r_e,
        "sigma2_global": sigma2_global,
        "sigma2_e8": sigma2_e8,
        "sigma2_e": sigma2_e,
        "sigma2_inf": sigma2_inf,
        "K_global": sigma2_global**2,
        "K_e8": sigma2_e8**2,
        "K_e": sigma2_e**2,
        "K_inf": sigma2_inf**2,
        "A_e8": sigma2_e8 / max(sigma2_global, 1e-300),
        "A_e": sigma2_e / max(sigma2_global, 1e-300),
        "A_inf": sigma2_inf / max(sigma2_global, 1e-300),
    }


def make_plots(results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for res in results:
        ax.loglog(res["r"] / res["R_e"], res["rho"] / np.max(res["rho"]), lw=2, label=fr"$n={res['n']:.0f}$")
    ax.set_xlabel(r"$r/R_e$")
    ax.set_ylabel(r"normalized $\rho$")
    ax.set_title("(a) Deprojected Sersic density")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    for res in results:
        ax.semilogx(res["R"] / res["R_e"], res["sigma_los2"] / np.sqrt(1.0), lw=2, label=fr"$n={res['n']:.0f}$")
    ax.set_xlabel(r"$R/R_e$")
    ax.set_ylabel(r"$\sigma_{\rm los}^2 / \sqrt{GMa_0}$")
    ax.set_title("(b) Projected dispersion profile")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    n_vals = [res["n"] for res in results]
    ax.plot(n_vals, [res["K_global"] for res in results], "o-", lw=2, label="global rms")
    ax.plot(n_vals, [res["K_e"] for res in results], "s-", lw=2, label=r"aperture $R_e$")
    ax.plot(n_vals, [res["K_e8"] for res in results], "^-", lw=2, label=r"aperture $R_e/8$")
    ax.set_xlabel("Sersic index n")
    ax.set_ylabel(r"$K_\sigma$")
    ax.set_title("(c) Faber-Jackson coefficient")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.axhline(4.0 / 81.0, color="k", ls="--", lw=1.5, label=r"exact global $1$D: $4/81$")
    ax.axhline(4.0 / 9.0, color="gray", ls=":", lw=1.5, label=r"exact $3$D rms: $4/9$")
    ax.plot(n_vals, [res["K_e8"] for res in results], "o-", lw=2, label=r"$K_\sigma(R_e/8)$")
    ax.plot(n_vals, [res["K_e"] for res in results], "s-", lw=2, label=r"$K_\sigma(R_e)$")
    ax.plot(n_vals, [res["K_inf"] for res in results], "^-", lw=2, label=r"$K_\sigma(\infty)$")
    ax.set_xlabel("Sersic index n")
    ax.set_ylabel(r"$K_\sigma$")
    ax.set_title("(d) Exact virial coefficient vs aperture conventions")
    ax.legend(fontsize=8)

    fig.suptitle("Pressure-Supported Sersic Systems in Deep MOND", y=0.99)
    fig.tight_layout()
    outpath = OUTDIR / "phase_ar_elliptical_ksigma.png"
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outpath


def print_results(results):
    print(SEP)
    print("  PHASE AR: K_sigma for Pressure-Supported Sersic Systems")
    print(SEP)
    print("  Deep-MOND virial baseline:")
    print("    exact global rms theorem => sigma_rms^4 = (4/81) G M a0")
    print("    equivalent 3D rms speed  => <v^2>^2 = (4/9) G M a0")
    print(f"    numerical check          => {results[0]['K_global']:.6f} (n=1 example)")
    print()
    print("  Sersic-model results:")
    print("    n      R_e(model)   K_global    K_sigma(R_e)   K_sigma(R_e/8)   A_sigma(R_e)")
    for res in results:
        print(
            f"    {res['n']:3.0f}     "
            f"{res['R_e']:9.4f}   "
            f"{res['K_global']:8.4f}      "
            f"{res['K_e']:8.4f}         "
            f"{res['K_e8']:8.4f}         "
            f"{res['A_e']:8.4f}"
        )
    print()
    print(
        "  Reading:\n"
        "  - the exact deep-MOND virial coefficient is profile-independent for the\n"
        "    global rms dispersion;\n"
        "  - the Sersic index enters when translating that global theorem into an\n"
        "    observed projected aperture dispersion;\n"
        "  - in this isotropic Sersic calculation, aperture conventions shift the\n"
        "    one-dimensional coefficient only by O(1), not by an order of magnitude;\n"
        "  - therefore one must distinguish carefully between the exact 3D virial\n"
        "    coefficient 4/9 and the projected 1D aperture coefficient K_sigma.\n"
    )


def run_all():
    results = [solve_case(n) for n in [1.0, 2.0, 4.0]]
    outpath = make_plots(results)
    print_results(results)
    print(f"  Plot saved to: {outpath}")
    return {"results": results, "plot_path": outpath}


if __name__ == "__main__":
    run_all()
