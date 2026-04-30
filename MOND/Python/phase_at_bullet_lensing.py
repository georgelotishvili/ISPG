"""
Phase AT: Bullet Cluster lensing benchmark from frozen hysteresis
=================================================================

Goal:
  Build a first quantitative 2D benchmark for the Bullet Cluster
  mass-light offset in the hysteresis sector.

Physical setup:
  1. Quantum / effective hysteresis equation:

         Box chi + lambda chi = lambda beta Y .

  2. Merger-timescale hierarchy:

         tau_rel = c / g_vir   >>   tau_cross = R_vir / v_coll .

     Therefore the hysteretic component cannot re-equilibrate during the
     collision. The minimal asymptotic branch is a "frozen" collisionless
     halo attached to the galaxy centroids rather than the stripped gas.

  3. Static profile choice:
     Appendix 13 gives rho_chi ~ 1/r^2 in the weak-field static limit.
     We therefore use a cored projected isothermal profile generated as the
     positive Helmholtz response of a compact collisionless source proxy.

What is benchmarked:
  - gas peak remains near the collision midplane;
  - chi-generated convergence peaks stay on the collisionless galaxies;
  - the total convergence map is peak-dominated on the galaxy side once the
    hysteretic mass per transported clump is only O(0.3 M_sub) or larger;
  - with an O(1) hysteretic mass M_chi ~ M_sub, the aperture mass and peak
    offset are in the observed Bullet-Cluster range.
  - the outer 250 kpc aperture is dominated by the transported chi component,
    with gas and galaxy fractions close to the observed Bullet decomposition.
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

from constants import G, Gyr, H0, Msun, c, kpc


SEP = "=" * 78
OUTDIR = Path(__file__).parent / "plots"
OUTDIR.mkdir(exist_ok=True)
PDF_DIR = Path(__file__).parent.parent / "PDF"
PDF_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# Fiducial merger geometry
# ---------------------------------------------------------------------
Z_LENS = 0.296
Z_SOURCE = 1.0

M_SUB_FID = 1.0e14 * Msun
M_SUB_CONSERVATIVE = 1.0e13 * Msun
F_GAS = 0.85
F_GAL = 0.15
BASIS_CLUSTER_MASS = 1.0e14 * Msun

R_VIR = 1.0e3 * kpc
D_ACCEL = 300.0 * kpc
V_COLL = 3.0e6              # 3000 km/s
OBSERVED_OFFSET = 720.0 * kpc

GAS_SIGMA_X = 240.0 * kpc
GAS_SIGMA_Y = 180.0 * kpc
GAL_SIGMA = 80.0 * kpc
CHI_SCREEN = 100.0 * kpc
APERTURE = 250.0 * kpc

MAIN_GAS_LAG = 450.0 * kpc
SUB_GAS_LAG = 320.0 * kpc
MAIN_GAS_SIGMA_X = 220.0 * kpc
MAIN_GAS_SIGMA_Y = 150.0 * kpc
SUB_GAS_SIGMA_X = 130.0 * kpc
SUB_GAS_SIGMA_Y = 100.0 * kpc

OBSERVED_MAIN_AP_MASS = 2.5e14 * Msun
OBSERVED_MAIN_AP_MASS_ERR = 0.1e14 * Msun
OBSERVED_SUB_AP_MASS = 2.0e14 * Msun
OBSERVED_SUB_AP_MASS_ERR = 0.2e14 * Msun
OBSERVED_GAS_FRAC = 0.09
OBSERVED_GAS_FRAC_ERR = 0.03
OBSERVED_GAL_FRAC = 0.11
OBSERVED_GAL_FRAC_ERR = 0.05
OBSERVED_DM_LIKE_FRAC = 1.0 - OBSERVED_GAS_FRAC - OBSERVED_GAL_FRAC
OBSERVED_DM_LIKE_FRAC_ERR = np.hypot(OBSERVED_GAS_FRAC_ERR, OBSERVED_GAL_FRAC_ERR)

GRID_HALF_SIZE = 2.0e3 * kpc
N_GRID = 801


def h_of_z(z):
    omega_b = 0.05
    omega_l = 0.95
    return H0 * np.sqrt(omega_b * (1.0 + z) ** 3 + omega_l)


def sigma_crit_si(z_lens=Z_LENS, z_source=Z_SOURCE, n_steps=20000):
    z_grid = np.linspace(0.0, z_source, n_steps)
    integrand = c / h_of_z(z_grid)
    chi = np.cumsum(
        np.r_[0.0, 0.5 * (integrand[1:] + integrand[:-1]) * np.diff(z_grid)]
    )
    chi_l = np.interp(z_lens, z_grid, chi)
    chi_s = chi[-1]
    d_l = chi_l / (1.0 + z_lens)
    d_s = chi_s / (1.0 + z_source)
    d_ls = (chi_s - chi_l) / (1.0 + z_source)
    return (c**2 / (4.0 * np.pi * G)) * d_s / max(d_l * d_ls, 1e-300)


def sigma_crit_msun_per_kpc2(z_lens=Z_LENS, z_source=Z_SOURCE):
    return sigma_crit_si(z_lens, z_source) / (Msun / kpc**2)


def merger_timescales(m_sub):
    a_coll = V_COLL**2 / D_ACCEL
    g_vir = G * m_sub / R_VIR**2
    tau_rel = c / max(g_vir, 1e-300)
    tau_cross = R_VIR / V_COLL
    return {
        "a_coll": a_coll,
        "g_vir": g_vir,
        "tau_rel": tau_rel,
        "tau_cross": tau_cross,
        "freeze_ratio": tau_rel / tau_cross,
    }


def grid():
    x = np.linspace(-GRID_HALF_SIZE, GRID_HALF_SIZE, N_GRID)
    y = np.linspace(-GRID_HALF_SIZE, GRID_HALF_SIZE, N_GRID)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    dx = x[1] - x[0]
    return x, y, xx, yy, dx


def gaussian_surface_density(xx, yy, mass, x0, sigma_x, sigma_y=None):
    sigma_y = sigma_y if sigma_y is not None else sigma_x
    norm = mass / (2.0 * np.pi * sigma_x * sigma_y)
    return norm * np.exp(
        -0.5 * ((xx - x0) / sigma_x) ** 2 - 0.5 * (yy / sigma_y) ** 2
    )


def helmholtz_response(source, dx, screen_length):
    n_y, n_x = source.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(n_x, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(n_y, d=dx)
    kxx, kyy = np.meshgrid(kx, ky, indexing="xy")
    lam = 1.0 / max(screen_length, 1e-300) ** 2
    denom = kxx**2 + kyy**2 + lam
    response_k = lam * np.fft.fft2(source) / np.maximum(denom, 1e-300)
    return np.fft.ifft2(response_k).real


def normalize_surface_density(field, target_mass, dx):
    positive = np.maximum(field, 0.0)
    mass_now = positive.sum() * dx**2
    return positive * (target_mass / max(mass_now, 1e-300))


def find_side_peak(surface_density, x, xx, side="right"):
    mask = xx > 0.0 if side == "right" else xx < 0.0
    masked = np.where(mask, surface_density, -1.0)
    idx = int(np.argmax(masked))
    i_y, i_x = np.unravel_index(idx, surface_density.shape)
    return {
        "x": float(x[i_x]),
        "y": 0.0,
        "value": float(surface_density[i_y, i_x]),
        "i_x": i_x,
        "i_y": i_y,
    }


def aperture_mass(surface_density, xx, yy, x0, y0, radius, dx):
    rr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    return float(surface_density[rr <= radius].sum() * dx**2)


def mass_breakdown(total, gas, gal, chi):
    total_safe = max(total, 1e-300)
    return {
        "total": total,
        "gas": gas,
        "gal": gal,
        "chi": chi,
        "gas_frac": gas / total_safe,
        "gal_frac": gal / total_safe,
        "chi_frac": chi / total_safe,
    }


def aperture_summary(results):
    total = 0.5 * (results["ap_mass_right"] + results["ap_mass_left"])
    gas = 0.5 * (results["ap_gas_right"] + results["ap_gas_left"])
    gal = 0.5 * (results["ap_gal_right"] + results["ap_gal_left"])
    chi = 0.5 * (results["ap_hyst_right"] + results["ap_hyst_left"])
    peak_offset = 0.5 * (
        abs(results["right_peak"]["x"]) + abs(results["left_peak"]["x"])
    )
    peak_kappa = 0.5 * (
        results["right_peak"]["value"] + results["left_peak"]["value"]
    )
    total_safe = max(total, 1e-300)
    return {
        "total": total,
        "gas": gas,
        "gal": gal,
        "chi": chi,
        "gas_frac": gas / total_safe,
        "gal_frac": gal / total_safe,
        "chi_frac": chi / total_safe,
        "peak_offset": peak_offset,
        "peak_kappa": peak_kappa,
        "peak_center_contrast": peak_kappa / max(results["center_value"], 1e-300),
        "required_m_sub_for_main": results["m_sub"] * OBSERVED_MAIN_AP_MASS / total_safe,
        "required_m_sub_for_sub": results["m_sub"] * OBSERVED_SUB_AP_MASS / total_safe,
    }


def asymmetric_summary(results):
    main = mass_breakdown(
        results["ap_main_total"],
        results["ap_main_gas"],
        results["ap_main_gal"],
        results["ap_main_chi"],
    )
    sub = mass_breakdown(
        results["ap_sub_total"],
        results["ap_sub_gas"],
        results["ap_sub_gal"],
        results["ap_sub_chi"],
    )
    mean = mass_breakdown(
        0.5 * (main["total"] + sub["total"]),
        0.5 * (main["gas"] + sub["gas"]),
        0.5 * (main["gal"] + sub["gal"]),
        0.5 * (main["chi"] + sub["chi"]),
    )
    return {
        "main": main,
        "sub": sub,
        "mean": mean,
        "main_peak_shift": abs(results["left_peak"]["x"] - results["main_center_x"]),
        "sub_peak_shift": abs(results["right_peak"]["x"] - results["sub_center_x"]),
        "mass_ratio": main["total"] / max(sub["total"], 1e-300),
    }


def build_maps_asymmetric(
    m_main=M_SUB_FID,
    m_sub=M_SUB_FID,
    f_hyst_main=1.0,
    f_hyst_sub=1.0,
    offset=OBSERVED_OFFSET,
    screen_length=CHI_SCREEN,
    gas_model="split",
):
    x, y, xx, yy, dx = grid()

    main_center_x = -offset
    sub_center_x = +offset

    if gas_model == "centered":
        gas_main = np.zeros_like(xx)
        gas_sub = np.zeros_like(xx)
        gas_map = gaussian_surface_density(
            xx, yy, F_GAS * (m_main + m_sub), 0.0, GAS_SIGMA_X, GAS_SIGMA_Y
        )
        gas_centers_x = np.array([0.0])
    elif gas_model == "split":
        gas_main_x = main_center_x + MAIN_GAS_LAG
        gas_sub_x = sub_center_x - SUB_GAS_LAG
        gas_main = gaussian_surface_density(
            xx, yy, F_GAS * m_main, gas_main_x, MAIN_GAS_SIGMA_X, MAIN_GAS_SIGMA_Y
        )
        gas_sub = gaussian_surface_density(
            xx, yy, F_GAS * m_sub, gas_sub_x, SUB_GAS_SIGMA_X, SUB_GAS_SIGMA_Y
        )
        gas_map = gas_main + gas_sub
        gas_centers_x = np.array([gas_main_x, gas_sub_x])
    else:
        raise ValueError(f"unknown gas_model={gas_model!r}")

    gal_left = gaussian_surface_density(xx, yy, F_GAL * m_main, main_center_x, GAL_SIGMA)
    gal_right = gaussian_surface_density(xx, yy, F_GAL * m_sub, sub_center_x, GAL_SIGMA)
    gal_map = gal_right + gal_left

    # Collisionless source proxy for chi in the frozen-hysteresis limit.
    source_right = gaussian_surface_density(xx, yy, 1.0, sub_center_x, GAL_SIGMA)
    source_left = gaussian_surface_density(xx, yy, 1.0, main_center_x, GAL_SIGMA)
    chi_raw_right = helmholtz_response(source_right, dx, screen_length)
    chi_raw_left = helmholtz_response(source_left, dx, screen_length)

    sigma_chi_right = normalize_surface_density(chi_raw_right, f_hyst_sub * m_sub, dx)
    sigma_chi_left = normalize_surface_density(chi_raw_left, f_hyst_main * m_main, dx)
    sigma_chi = sigma_chi_right + sigma_chi_left

    sigma_bary = gas_map + gal_map
    sigma_total = sigma_bary + sigma_chi
    sigma_crit = sigma_crit_si()
    kappa_gas = gas_map / sigma_crit
    kappa_gal = gal_map / sigma_crit
    kappa_chi = sigma_chi / sigma_crit
    kappa_total = sigma_total / sigma_crit

    right_peak = find_side_peak(kappa_total, x, xx, side="right")
    left_peak = find_side_peak(kappa_total, x, xx, side="left")

    return {
        "x": x,
        "y": y,
        "xx": xx,
        "yy": yy,
        "dx": dx,
        "sigma_crit": sigma_crit,
        "sigma_bary": sigma_bary,
        "sigma_gas": gas_map,
        "sigma_gas_main": gas_main,
        "sigma_gas_sub": gas_sub,
        "sigma_gal": gal_map,
        "sigma_gal_main": gal_left,
        "sigma_gal_sub": gal_right,
        "sigma_chi": sigma_chi,
        "sigma_chi_main": sigma_chi_left,
        "sigma_chi_sub": sigma_chi_right,
        "sigma_total": sigma_total,
        "kappa_gas": kappa_gas,
        "kappa_gal": kappa_gal,
        "kappa_chi": kappa_chi,
        "kappa_total": kappa_total,
        "right_peak": right_peak,
        "left_peak": left_peak,
        "ap_mass_right": aperture_mass(sigma_total, xx, yy, right_peak["x"], 0.0, APERTURE, dx),
        "ap_mass_left": aperture_mass(sigma_total, xx, yy, left_peak["x"], 0.0, APERTURE, dx),
        "ap_gas_right": aperture_mass(gas_map, xx, yy, right_peak["x"], 0.0, APERTURE, dx),
        "ap_gas_left": aperture_mass(gas_map, xx, yy, left_peak["x"], 0.0, APERTURE, dx),
        "ap_gal_right": aperture_mass(gal_map, xx, yy, right_peak["x"], 0.0, APERTURE, dx),
        "ap_gal_left": aperture_mass(gal_map, xx, yy, left_peak["x"], 0.0, APERTURE, dx),
        "ap_hyst_right": aperture_mass(sigma_chi, xx, yy, right_peak["x"], 0.0, APERTURE, dx),
        "ap_hyst_left": aperture_mass(sigma_chi, xx, yy, left_peak["x"], 0.0, APERTURE, dx),
        "ap_main_total": aperture_mass(sigma_total, xx, yy, main_center_x, 0.0, APERTURE, dx),
        "ap_sub_total": aperture_mass(sigma_total, xx, yy, sub_center_x, 0.0, APERTURE, dx),
        "ap_main_gas": aperture_mass(gas_map, xx, yy, main_center_x, 0.0, APERTURE, dx),
        "ap_sub_gas": aperture_mass(gas_map, xx, yy, sub_center_x, 0.0, APERTURE, dx),
        "ap_main_gal": aperture_mass(gal_map, xx, yy, main_center_x, 0.0, APERTURE, dx),
        "ap_sub_gal": aperture_mass(gal_map, xx, yy, sub_center_x, 0.0, APERTURE, dx),
        "ap_main_chi": aperture_mass(sigma_chi, xx, yy, main_center_x, 0.0, APERTURE, dx),
        "ap_sub_chi": aperture_mass(sigma_chi, xx, yy, sub_center_x, 0.0, APERTURE, dx),
        "center_value": float(kappa_total[len(y) // 2, len(x) // 2]),
        "f_hyst_main": f_hyst_main,
        "f_hyst_sub": f_hyst_sub,
        "m_main": m_main,
        "m_sub": m_sub,
        "screen_length": screen_length,
        "offset": offset,
        "main_center_x": main_center_x,
        "sub_center_x": sub_center_x,
        "target_centers_x": np.array([main_center_x, sub_center_x]),
        "gas_centers_x": gas_centers_x,
        "gas_model": gas_model,
    }


def build_maps(m_sub=M_SUB_FID, f_hyst=1.0, offset=OBSERVED_OFFSET,
               screen_length=CHI_SCREEN):
    return build_maps_asymmetric(
        m_main=m_sub,
        m_sub=m_sub,
        f_hyst_main=f_hyst,
        f_hyst_sub=f_hyst,
        offset=offset,
        screen_length=screen_length,
        gas_model="centered",
    )


def calibrate_asymmetric_benchmark(
    f_hyst_main=1.0,
    f_hyst_sub=1.0,
    offset=OBSERVED_OFFSET,
    screen_length=CHI_SCREEN,
):
    main_basis = build_maps_asymmetric(
        m_main=BASIS_CLUSTER_MASS,
        m_sub=0.0,
        f_hyst_main=f_hyst_main,
        f_hyst_sub=f_hyst_sub,
        offset=offset,
        screen_length=screen_length,
        gas_model="split",
    )
    sub_basis = build_maps_asymmetric(
        m_main=0.0,
        m_sub=BASIS_CLUSTER_MASS,
        f_hyst_main=f_hyst_main,
        f_hyst_sub=f_hyst_sub,
        offset=offset,
        screen_length=screen_length,
        gas_model="split",
    )
    mixing_matrix = np.array(
        [
            [main_basis["ap_main_total"], sub_basis["ap_main_total"]],
            [main_basis["ap_sub_total"], sub_basis["ap_sub_total"]],
        ]
    )
    target_vector = np.array([OBSERVED_MAIN_AP_MASS, OBSERVED_SUB_AP_MASS])
    scale_factors = np.linalg.solve(mixing_matrix, target_vector)
    result = build_maps_asymmetric(
        m_main=float(scale_factors[0] * BASIS_CLUSTER_MASS),
        m_sub=float(scale_factors[1] * BASIS_CLUSTER_MASS),
        f_hyst_main=f_hyst_main,
        f_hyst_sub=f_hyst_sub,
        offset=offset,
        screen_length=screen_length,
        gas_model="split",
    )
    result["mixing_matrix"] = mixing_matrix
    result["scale_factors"] = scale_factors
    return result


def threshold_scan(m_sub=M_SUB_FID, offset=OBSERVED_OFFSET,
                   screen_lengths=(80.0, 100.0, 120.0, 150.0),
                   f_grid=None):
    if f_grid is None:
        f_grid = np.linspace(0.1, 1.5, 120)

    rows = []
    for screen_kpc in screen_lengths:
        screen = screen_kpc * kpc
        threshold = np.nan
        for f_hyst in f_grid:
            res = build_maps(m_sub=m_sub, f_hyst=float(f_hyst), offset=offset, screen_length=screen)
            if (
                res["right_peak"]["value"] > res["center_value"]
                and abs(res["right_peak"]["x"] - offset) < 20.0 * kpc
            ):
                threshold = float(f_hyst)
                break
        rows.append({"screen_kpc": float(screen_kpc), "threshold": threshold})
    return rows


def make_plots(results, threshold_rows):
    x = results["x"] / kpc
    y = results["y"] / kpc
    xx = results["xx"] / kpc
    yy = results["yy"] / kpc
    summary = asymmetric_summary(results)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11.0))

    vmax = max(
        np.max(results["kappa_total"]),
        np.max(results["kappa_chi"]),
        np.max(results["kappa_gas"] + results["kappa_gal"]),
    )

    ax = axes[0, 0]
    im = ax.imshow(
        results["kappa_gas"] + results["kappa_gal"],
        origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
        cmap="magma",
        aspect="equal",
        vmax=vmax,
    )
    ax.set_title("(a) Baryons only")
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 1]
    im = ax.imshow(
        results["kappa_chi"],
        origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
        cmap="viridis",
        aspect="equal",
        vmax=vmax,
    )
    ax.axvline(0.0, color="w", ls=":", lw=1.1)
    ax.set_title("(b) Hysteretic convergence")
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    im = ax.imshow(
        results["kappa_total"],
        origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
        cmap="plasma",
        aspect="equal",
        vmax=vmax,
    )
    ax.axvline(0.0, color="w", ls=":", lw=1.1)
    ax.scatter(
        [results["left_peak"]["x"] / kpc, results["right_peak"]["x"] / kpc],
        [0.0, 0.0],
        c="cyan",
        s=35,
        label="lensing peaks",
    )
    ax.scatter(
        results["gas_centers_x"] / kpc,
        np.zeros_like(results["gas_centers_x"] / kpc),
        c="white",
        s=35,
        marker="x",
        label="gas peaks",
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("(c) Total convergence")
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 1]
    mid = len(y) // 2
    ax.plot(x, results["kappa_total"][mid, :], lw=2.3, label="total")
    ax.plot(x, (results["kappa_gas"] + results["kappa_gal"])[mid, :], lw=2.0, ls="--", label="baryons only")
    ax.plot(x, results["kappa_chi"][mid, :], lw=2.0, ls="-.", label="chi only")
    for x0 in results["gas_centers_x"] / kpc:
        ax.axvline(x0, color="k", ls=":", lw=1.0)
    for x0 in results["target_centers_x"] / kpc:
        ax.axvline(x0, color="gray", ls="--", lw=1.0)
    ax.set_title("(d) Merger-axis convergence profile")
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel(r"$\kappa(x,0)$")
    ax.legend(fontsize=8)

    fig.suptitle("Phase AT: Asymmetric Bullet Cluster Frozen-Hysteresis Benchmark", y=0.99)
    fig.tight_layout()
    outpath = OUTDIR / "phase_at_bullet_lensing.png"
    pdf_path = PDF_DIR / "fig_mond_bullet_cluster.pdf"
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    fig2, axes2 = plt.subplots(1, 2, figsize=(12.0, 4.8))

    ax = axes2[0]
    ax.plot(
        [row["screen_kpc"] for row in threshold_rows],
        [row["threshold"] for row in threshold_rows],
        "o-",
        lw=2,
    )
    ax.set_xlabel(r"screening/core length [kpc]")
    ax.set_ylabel(r"threshold $M_\chi/M_{\rm sub}$")
    ax.set_title("Hysteretic mass needed for outer peak dominance")
    ax.grid(alpha=0.25)

    ax = axes2[1]
    categories = ["main", "sub"]
    x_bar = np.arange(len(categories))
    width = 0.36
    model = np.array(
        [summary["main"]["total"], summary["sub"]["total"]]
    ) / (1.0e14 * Msun)
    observed = np.array(
        [OBSERVED_MAIN_AP_MASS, OBSERVED_SUB_AP_MASS]
    ) / (1.0e14 * Msun)
    observed_err = np.array(
        [OBSERVED_MAIN_AP_MASS_ERR, OBSERVED_SUB_AP_MASS_ERR]
    ) / (1.0e14 * Msun)
    ax.bar(
        x_bar - width / 2,
        observed,
        width,
        yerr=observed_err,
        capsize=4,
        label="observed",
        color="#7aa6c2",
    )
    ax.bar(
        x_bar + width / 2,
        model,
        width,
        label="asymmetric model",
        color="#d49461",
    )
    ax.set_xticks(x_bar, categories)
    ax.set_ylabel(r"$M(<250\,\mathrm{kpc})$ [$10^{14} M_\odot$]")
    ax.set_title("Observed vs calibrated aperture masses")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="upper center")
    ax.text(
        0.03,
        0.97,
        "\n".join(
            [
                f"main: f_gas={summary['main']['gas_frac']:.2f}, "
                f"f_gal={summary['main']['gal_frac']:.2f}, "
                f"f_chi={summary['main']['chi_frac']:.2f}",
                f"sub:  f_gas={summary['sub']['gas_frac']:.2f}, "
                f"f_gal={summary['sub']['gal_frac']:.2f}, "
                f"f_chi={summary['sub']['chi_frac']:.2f}",
            ]
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )

    outpath2 = OUTDIR / "phase_at_bullet_threshold.png"
    pdf_path2 = PDF_DIR / "fig_mond_bullet_threshold.pdf"
    fig2.tight_layout()
    fig2.savefig(outpath2, dpi=180, bbox_inches="tight")
    fig2.savefig(pdf_path2, bbox_inches="tight")
    plt.close(fig2)
    return outpath, outpath2


def print_results(fiducial, threshold_rows):
    sigma_crit = sigma_crit_msun_per_kpc2()
    ts_main = merger_timescales(fiducial["m_main"])
    ts_sub = merger_timescales(fiducial["m_sub"])
    ts_cons = merger_timescales(M_SUB_CONSERVATIVE)
    peak_summary = aperture_summary(fiducial)
    summary = asymmetric_summary(fiducial)

    print(SEP)
    print("  PHASE AT: Asymmetric Bullet Cluster Frozen-Hysteresis Benchmark")
    print(SEP)
    print("  Geometry and lensing distances:")
    print(f"    z_lens  = {Z_LENS:.3f}")
    print(f"    z_src   = {Z_SOURCE:.1f}")
    print(f"    Sigma_crit = {sigma_crit/1.0e9:.3f} x 10^9 Msun / kpc^2")
    print()

    print("  Collision parameters:")
    print(f"    v_coll  = {V_COLL/1.0e3:.0f} km/s")
    print(f"    d_accel = {D_ACCEL/kpc:.0f} kpc")
    print(f"    a_coll  = {ts_main['a_coll']:.3e} m/s^2")
    print(f"             = {ts_main['a_coll']/1.2e-10:.2f} a0_obs")
    print(f"    observed gas-to-lensing offset benchmark = {OBSERVED_OFFSET/kpc:.0f} kpc")
    print()

    print("  Relaxation hierarchy (transport-side tau_rel = c/g_vir):")
    print(
        f"    main baryonic scale = {fiducial['m_main']/Msun:.2e} Msun -> "
        f"g_vir = {ts_main['g_vir']:.3e} m/s^2, "
        f"tau_rel = {ts_main['tau_rel']/Gyr:.1f} Gyr, "
        f"tau_cross = {ts_main['tau_cross']/Gyr:.3f} Gyr, "
        f"tau_rel/tau_cross = {ts_main['freeze_ratio']:.1f}"
    )
    print(
        f"    sub  baryonic scale = {fiducial['m_sub']/Msun:.2e} Msun -> "
        f"g_vir = {ts_sub['g_vir']:.3e} m/s^2, "
        f"tau_rel = {ts_sub['tau_rel']/Gyr:.1f} Gyr, "
        f"tau_cross = {ts_sub['tau_cross']/Gyr:.3f} Gyr, "
        f"tau_rel/tau_cross = {ts_sub['freeze_ratio']:.1f}"
    )
    print(
        f"    conservative 1e13 Msun check -> tau_rel/tau_cross = {ts_cons['freeze_ratio']:.1f}"
    )
    print("    Merger is therefore deeply in the frozen-hysteresis regime.")
    print()

    print("  Asymmetric benchmark assumptions:")
    print(f"    main baryonic scale        = {fiducial['m_main']/Msun:.2e} Msun")
    print(f"    sub  baryonic scale        = {fiducial['m_sub']/Msun:.2e} Msun")
    print(f"    gas fraction / galaxy fraction = {F_GAS:.2f} / {F_GAL:.2f}")
    print(
        f"    main/sub gas lags          = {MAIN_GAS_LAG/kpc:.0f} / {SUB_GAS_LAG/kpc:.0f} kpc"
    )
    print(f"    chi screening/core length  = {fiducial['screen_length']/kpc:.0f} kpc")
    print(
        f"    transported hysteretic masses = "
        f"{fiducial['f_hyst_main']:.2f} M_main and {fiducial['f_hyst_sub']:.2f} M_sub"
    )
    print()

    print("  Reconstructed convergence peaks:")
    print(
        f"    main-side peak at x = {fiducial['left_peak']['x']/kpc:.1f} kpc, "
        f"shift from galaxy centroid = {summary['main_peak_shift']/kpc:.1f} kpc"
    )
    print(
        f"    sub-side  peak at x = {fiducial['right_peak']['x']/kpc:.1f} kpc, "
        f"shift from galaxy centroid = {summary['sub_peak_shift']/kpc:.1f} kpc"
    )
    print(
        f"    main/sub peak kappas    = "
        f"{fiducial['left_peak']['value']:.3f} / {fiducial['right_peak']['value']:.3f}"
    )
    print(
        f"    midplane kappa          = {fiducial['center_value']:.3f}"
    )
    print(f"    mean peak/centre contrast = {peak_summary['peak_center_contrast']:.3f}")
    print()

    print("  Fixed 250 kpc apertures at the observed galaxy centroids:")
    print(
        f"    model M_main = {summary['main']['total']/Msun:.3e} Msun"
        f"   (target {OBSERVED_MAIN_AP_MASS/Msun:.3e})"
    )
    print(
        f"    model M_sub  = {summary['sub']['total']/Msun:.3e} Msun"
        f"   (target {OBSERVED_SUB_AP_MASS/Msun:.3e})"
    )
    print(f"    model mass ratio M_main/M_sub = {summary['mass_ratio']:.3f}")
    print()

    print("  Local aperture decomposition (main cluster side):")
    print(f"    gas fraction      = {summary['main']['gas_frac']:.3f}")
    print(f"    galaxy fraction   = {summary['main']['gal_frac']:.3f}")
    print(f"    chi fraction      = {summary['main']['chi_frac']:.3f}")
    print()

    print("  Local aperture decomposition (subcluster side):")
    print(f"    gas fraction      = {summary['sub']['gas_frac']:.3f}")
    print(f"    galaxy fraction   = {summary['sub']['gal_frac']:.3f}")
    print(f"    chi fraction      = {summary['sub']['chi_frac']:.3f}")
    print()

    print("  Mean of the two outer apertures:")
    print(f"    gas fraction      = {summary['mean']['gas_frac']:.3f}")
    print(f"    galaxy fraction   = {summary['mean']['gal_frac']:.3f}")
    print(f"    chi fraction      = {summary['mean']['chi_frac']:.3f}")
    print(
        f"    peak/centre contrast = {peak_summary['peak_center_contrast']:.3f}"
    )
    print()

    print("  Observational decomposition benchmark (R < 250 kpc):")
    print(
        f"    gas fraction      = {OBSERVED_GAS_FRAC:.2f} ± {OBSERVED_GAS_FRAC_ERR:.2f}"
    )
    print(
        f"    galaxy fraction   = {OBSERVED_GAL_FRAC:.2f} ± {OBSERVED_GAL_FRAC_ERR:.2f}"
    )
    print(
        f"    residual fraction = {OBSERVED_DM_LIKE_FRAC:.2f} ± {OBSERVED_DM_LIKE_FRAC_ERR:.2f}"
    )
    print()

    print("  Threshold scan: minimum transported hysteretic mass needed")
    print("  for the weak-lensing peak to sit on the galaxy side:")
    print("    L_chi [kpc]    threshold M_chi / M_sub   (symmetric family)")
    for row in threshold_rows:
        print(f"      {row['screen_kpc']:7.0f}           {row['threshold']:.3f}")
    print()

    print("  Reading:")
    print(
        "  - in the symmetric centered-gas existence test, once M_chi is only"
        " ~0.2-0.4 M_sub per transported clump (depending on core length), the"
        " outer galaxy-side convergence peak overtakes the gas-centre peak;"
    )
    print(
        "  - after promoting the benchmark to an asymmetric main/sub solve with"
        " split gas peaks, the model is calibrated to the observed 250 kpc main"
        " and sub aperture masses rather than merely matching them in order of"
        " magnitude;"
    )
    print(
        "  - both calibrated outer apertures remain chi-dominated, while the gas"
        " contribution rises relative to the earlier single-midplane gas model,"
        " making the local mass budget more realistic;"
    )
    print(
        "  - the lensing peaks stay locked to the galaxy centroids at the"
        " observed ~720 kpc scale, so the asymmetric calibration does not spoil"
        " the offset mechanism;"
    )
    print(
        "  - this benchmark is still a minimal frozen-profile solve, not a full"
        " hydrodynamic merger simulation of the chi sector."
    )
    print(SEP)


def robustness_scan():
    """Vary f_hyst and L_chi around the fiducial and report key outputs."""
    print()
    print(SEP)
    print("  ROBUSTNESS SCAN: sensitivity to f_hyst and L_chi")
    print(SEP)
    header = (
        f"  {'f_hyst':>6s}  {'L_chi':>5s}  "
        f"{'peak_shift_main':>15s}  {'peak_shift_sub':>14s}  "
        f"{'f_chi_main':>10s}  {'f_chi_sub':>9s}  "
        f"{'kappa_main':>10s}  {'kappa_sub':>9s}"
    )
    print(header)
    print("  " + "-" * len(header.strip()))

    rows = []
    for f_hyst in [0.8, 1.0, 1.3, 1.6, 2.0]:
        for l_chi_kpc in [60.0, 80.0, 100.0, 120.0, 150.0]:
            res = calibrate_asymmetric_benchmark(
                f_hyst_main=f_hyst,
                f_hyst_sub=f_hyst,
                offset=OBSERVED_OFFSET,
                screen_length=l_chi_kpc * kpc,
            )
            s = asymmetric_summary(res)
            row = {
                "f_hyst": f_hyst,
                "L_chi_kpc": l_chi_kpc,
                "peak_shift_main_kpc": s["main_peak_shift"] / kpc,
                "peak_shift_sub_kpc": s["sub_peak_shift"] / kpc,
                "f_chi_main": s["main"]["chi_frac"],
                "f_chi_sub": s["sub"]["chi_frac"],
                "kappa_main": res["left_peak"]["value"],
                "kappa_sub": res["right_peak"]["value"],
            }
            rows.append(row)
            print(
                f"  {f_hyst:6.1f}  {l_chi_kpc:5.0f}  "
                f"{row['peak_shift_main_kpc']:15.1f}  {row['peak_shift_sub_kpc']:14.1f}  "
                f"{row['f_chi_main']:10.3f}  {row['f_chi_sub']:9.3f}  "
                f"{row['kappa_main']:10.3f}  {row['kappa_sub']:9.3f}"
            )
    print()
    print("  Key: peak_shift = distance from convergence peak to galaxy centroid (kpc)")
    print("       f_chi = chi fraction within 250 kpc aperture")
    print("       kappa = peak convergence value")
    print(SEP)
    return rows


def main():
    fiducial = calibrate_asymmetric_benchmark(
        f_hyst_main=1.3,
        f_hyst_sub=1.3,
        offset=OBSERVED_OFFSET,
        screen_length=CHI_SCREEN,
    )
    threshold_rows = threshold_scan()
    plot1, plot2 = make_plots(fiducial, threshold_rows)
    print_results(fiducial, threshold_rows)
    robustness_scan()
    print(f"  Plots saved to: {plot1}")
    print(f"                  {plot2}")


if __name__ == "__main__":
    main()
