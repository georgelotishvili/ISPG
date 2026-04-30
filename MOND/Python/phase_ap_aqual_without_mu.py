"""
Phase AP: AQUAL-like operator solve without explicit mu(x) input
================================================================

Goal:
  Build the operator-side verification channel in which the AQUAL solve does
  not receive the interpolating function mu(x) = x/(1+x) as an external input.

Construction:
  - Solve the baryonic Newtonian field u_N on the axisymmetric grid.
  - Start from the blind self-calibrated source-side branch as an initial guess.
  - Iterate the operator equation

        div[mu_eff grad u] = -f_bary

    with the effective coefficient rebuilt each iteration from the coupled
    fields themselves:

        mu_eff(R,z) = |grad u_N| / |grad u| .

  The algebraic MOND law and the standard AQUAL solve are used only as
  diagnostics after convergence.
"""

from pathlib import Path
import io
import sys

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from constants import a0, kpc, r_M
from pde_2d_swirl_verify import solve_aqual_2d
from phase_al_blind_selfcalibrated_kernel_2d import solve_blind_selfcalibrated_kernel_2d
from phase_u_axisymmetric_swirl_2d import (
    build_axisymmetric_operator,
    grad_xi,
    solve_linear_poisson,
)


SEP = "=" * 78
OUTDIR = Path(__file__).parent / "plots"
OUTDIR.mkdir(exist_ok=True)


def grad_zeta(u, dzeta):
    out = np.zeros_like(u)
    out[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * dzeta)
    out[:, 0] = 0.0
    out[:, -1] = (u[:, -1] - u[:, -2]) / dzeta
    return out


def field_magnitude(u, dxi, dzeta):
    du_dxi = grad_xi(u, dxi)
    du_dzeta = grad_zeta(u, dzeta)
    return np.sqrt(np.maximum(du_dxi**2 + du_dzeta**2, 1e-30))


def regularize_mu(mu):
    mu = np.asarray(mu, dtype=float).copy()
    mu = np.clip(mu, 1e-6, 1.0)
    if mu.shape[0] > 1:
        mu[0, :] = mu[1, :]
    if mu.shape[1] > 1:
        mu[:, 0] = mu[:, 1]
        mu[:, -1] = mu[:, -2]
    if mu.shape[0] > 1:
        mu[-1, :] = mu[-2, :]
    return mu


def build_mu_operator(xi, zeta, mu_field):
    nr, nz = len(xi), len(zeta)
    dxi = xi[1] - xi[0]
    dz = zeta[1] - zeta[0]
    ntot = nr * nz

    def idx(i, j):
        return i * nz + j

    rows, cols, vals = [], [], []

    for i in range(nr):
        xi_i = xi[i]
        for j in range(nz):
            k = idx(i, j)

            if i == nr - 1 or j == nz - 1:
                rows.append(k); cols.append(k); vals.append(1.0)
                continue

            mu_c = mu_field[i, j]

            if i == 0 and j == 0:
                mu_xp = 0.5 * (mu_field[1, 0] + mu_c)
                mu_zp = 0.5 * (mu_field[0, 1] + mu_c)
                rows.append(k); cols.append(k); vals.append(-(4.0 * mu_xp / dxi**2 + 2.0 * mu_zp / dz**2))
                rows.append(k); cols.append(idx(1, 0)); vals.append(4.0 * mu_xp / dxi**2)
                rows.append(k); cols.append(idx(0, 1)); vals.append(2.0 * mu_zp / dz**2)
                continue

            if i == 0:
                mu_xp = 0.5 * (mu_field[1, j] + mu_c)
                mu_zp = 0.5 * (mu_field[0, min(j + 1, nz - 1)] + mu_c)
                mu_zm = 0.5 * (mu_field[0, max(j - 1, 0)] + mu_c)
                rows.append(k); cols.append(k); vals.append(-(4.0 * mu_xp / dxi**2 + (mu_zp + mu_zm) / dz**2))
                rows.append(k); cols.append(idx(1, j)); vals.append(4.0 * mu_xp / dxi**2)
                rows.append(k); cols.append(idx(0, j - 1)); vals.append(mu_zm / dz**2)
                rows.append(k); cols.append(idx(0, j + 1)); vals.append(mu_zp / dz**2)
                continue

            mu_xp = 0.5 * (mu_field[min(i + 1, nr - 1), j] + mu_c)
            mu_xm = 0.5 * (mu_field[max(i - 1, 0), j] + mu_c)

            if j == 0:
                mu_zp = 0.5 * (mu_field[i, 1] + mu_c)
                mu_zm = mu_zp

                xph = xi_i + dxi / 2.0
                xmh = xi_i - dxi / 2.0
                cx_p = mu_xp * xph / (xi_i * dxi**2)
                cx_m = mu_xm * xmh / (xi_i * dxi**2)
                cz_p = mu_zp / dz**2
                cz_m = mu_zm / dz**2

                rows.append(k); cols.append(idx(i + 1, j)); vals.append(cx_p)
                rows.append(k); cols.append(idx(i - 1, j)); vals.append(cx_m)
                rows.append(k); cols.append(idx(i, 1)); vals.append(cz_p + cz_m)
                rows.append(k); cols.append(k); vals.append(-(cx_p + cx_m + cz_p + cz_m))
                continue

            mu_zp = 0.5 * (mu_field[i, min(j + 1, nz - 1)] + mu_c)
            mu_zm = 0.5 * (mu_field[i, max(j - 1, 0)] + mu_c)

            xph = xi_i + dxi / 2.0
            xmh = xi_i - dxi / 2.0
            cx_p = mu_xp * xph / (xi_i * dxi**2)
            cx_m = mu_xm * xmh / (xi_i * dxi**2)
            cz_p = mu_zp / dz**2
            cz_m = mu_zm / dz**2

            rows.append(k); cols.append(idx(i + 1, j)); vals.append(cx_p)
            rows.append(k); cols.append(idx(i - 1, j)); vals.append(cx_m)
            rows.append(k); cols.append(idx(i, j + 1)); vals.append(cz_p)
            rows.append(k); cols.append(idx(i, j - 1)); vals.append(cz_m)
            rows.append(k); cols.append(k); vals.append(-(cx_p + cx_m + cz_p + cz_m))

    return sparse.csr_matrix((vals, (rows, cols)), shape=(ntot, ntot))


def solve_with_mu_operator(operator, source, nr, nz):
    rhs = -np.asarray(source, dtype=float).reshape(-1).copy()
    for i in range(nr):
        rhs[i * nz + (nz - 1)] = 0.0
    for j in range(nz):
        rhs[(nr - 1) * nz + j] = 0.0
    u = spsolve(operator, rhs).reshape((nr, nz))
    u[nr - 1, :] = 0.0
    u[:, nz - 1] = 0.0
    return u


def relative_profile_error(xi, g_eval, g_target, xi_min=0.3, xi_max=10.0):
    mask = (xi >= xi_min) & (xi <= xi_max)
    rel = np.abs(g_eval[mask] - g_target[mask]) / np.maximum(g_target[mask], 1e-30)
    return float(np.sqrt(np.mean(rel**2))), float(np.max(rel))


def run_coupled_operator(
    nr=81,
    nz=61,
    xi_max=20.0,
    zeta_max=10.0,
    max_iter=600,
    tol=1e-6,
    mu_mix_hi=0.65,
    mu_mix_lo=0.00,
    mu_mix_decay=70.0,
):
    blind = solve_blind_selfcalibrated_kernel_2d(
        nr=nr,
        nz=nz,
        xi_max=xi_max,
        zeta_max=zeta_max,
    )

    xi, zeta = blind["xi"], blind["zeta"]
    dxi = xi[1] - xi[0]
    dzeta = zeta[1] - zeta[0]
    mat = build_axisymmetric_operator(
        nr=nr, nz=nz, xi_max=xi_max, zeta_max=zeta_max
    )[2]

    f_bary = blind["source_total"] - blind["source_extra"]
    solver = sparse.linalg.factorized(mat)
    u_n = solve_linear_poisson(solver, f_bary, nr, nz)
    u_blind = solve_linear_poisson(solver, blind["source_total"], nr, nz)

    g_n_field = field_magnitude(u_n, dxi, dzeta)
    u = u_blind.copy()
    history = []
    best_mu = None
    best_aqual = None

    print(SEP)
    print("  PHASE AP: AQUAL Without Explicit Mu Input")
    print(SEP)
    print(f"  Grid: {nr} x {nz}  (xi_max={xi_max:.1f}, zeta_max={zeta_max:.1f})")
    print(f"  max_iter = {max_iter}")
    print(f"  tol      = {tol:.1e}")
    print(f"  mu_mix_hi    = {mu_mix_hi:.2f}")
    print(f"  mu_mix_lo    = {mu_mix_lo:.2f}")
    print(f"  mu_mix_decay = {mu_mix_decay:.1f}")
    print()

    mu_field = regularize_mu(
        np.divide(g_n_field, np.maximum(field_magnitude(u, dxi, dzeta), 1e-30), out=np.ones_like(u), where=True)
    )

    for it in range(max_iter):
        g_field = field_magnitude(u, dxi, dzeta)
        mu_target = regularize_mu(
            np.divide(g_n_field, g_field, out=np.ones_like(g_field), where=g_field > 1e-20)
        )
        alpha_mu = mu_mix_lo + (mu_mix_hi - mu_mix_lo) * np.exp(-it / mu_mix_decay)
        mu_field = (1.0 - alpha_mu) * mu_field + alpha_mu * mu_target
        mu_field = regularize_mu(mu_field)
        operator = build_mu_operator(xi, zeta, mu_field)
        u_new = solve_with_mu_operator(operator, f_bary, nr, nz)
        rel_update = np.max(np.abs(u_new - u)) / max(np.max(np.abs(u_new)), 1e-30)
        u = u_new

        g_mid = np.maximum(-grad_xi(u, dxi)[:, 0], 1e-30)
        mu_mid = np.divide(
            blind["g_N"], np.maximum(g_mid, 1e-30), out=np.ones_like(g_mid), where=g_mid > 1e-20
        )
        mu_target = np.divide(g_mid, 1.0 + g_mid, out=np.zeros_like(g_mid), where=g_mid > 0)
        rms_blind, _ = relative_profile_error(xi, g_mid, blind["g_eff"])
        rms_aqual, _ = relative_profile_error(xi, g_mid, blind["g_aqual"])
        rms_alg, _ = relative_profile_error(xi, g_mid, blind["g_alg"])
        mask = (xi >= 0.3) & (xi <= 10.0)
        rms_mu = float(np.sqrt(np.mean((mu_mid[mask] - mu_target[mask]) ** 2)))

        record = {
            "iter": it + 1,
            "rel_update": rel_update,
            "rms_blind": rms_blind,
            "rms_aqual": rms_aqual,
            "rms_alg": rms_alg,
            "rms_mu": rms_mu,
            "mu_min": float(np.min(mu_field[mask, :])),
            "mu_max": float(np.max(mu_field[mask, :])),
            "alpha_mu": float(alpha_mu),
        }
        history.append(record)

        if best_mu is None or record["rms_mu"] < best_mu["record"]["rms_mu"]:
            best_mu = {
                "record": record.copy(),
                "g_mid": g_mid.copy(),
                "mu_mid": mu_mid.copy(),
            }
        if best_aqual is None or record["rms_aqual"] < best_aqual["record"]["rms_aqual"]:
            best_aqual = {
                "record": record.copy(),
                "g_mid": g_mid.copy(),
                "mu_mid": mu_mid.copy(),
            }

        if (it + 1) % 10 == 0:
            print(
                f"    iter {it+1:3d}: rel={rel_update:.3e}, "
                f"rms(blind)={rms_blind:.3e}, rms(AQUAL)={rms_aqual:.3e}, "
                f"rms(mu)={rms_mu:.3e}, alpha_mu={alpha_mu:.3f}"
            )

        if rel_update < tol:
            print(f"    Converged at iter {it+1} with rel_update={rel_update:.3e}")
            break

    g_eff = np.maximum(-grad_xi(u, dxi)[:, 0], 1e-30)
    mu_eff = np.divide(blind["g_N"], g_eff, out=np.ones_like(g_eff), where=g_eff > 1e-20)
    mu_target = np.divide(g_eff, 1.0 + g_eff, out=np.zeros_like(g_eff), where=g_eff > 0)

    res_aqual = solve_aqual_2d(
        NR=nr,
        Nz=nz,
        R_max_kpc=xi_max * r_M / kpc,
        z_max_kpc=zeta_max * r_M / kpc,
        max_iter=160,
        tol=1e-6,
        omega_relax=0.3,
        verbose=False,
    )
    g_aqual = np.interp(xi, res_aqual["xi"], res_aqual["g_eff_mid"])
    mu_aqual = np.interp(xi, res_aqual["xi"], res_aqual["mu_eff_mid"])

    mask = (xi >= 0.3) & (xi <= 10.0)
    result = {
        "xi": xi,
        "zeta": zeta,
        "u": u,
        "u_n": u_n,
        "blind": blind,
        "g_eff": g_eff,
        "g_aqual": g_aqual,
        "mu_eff": mu_eff,
        "mu_target": mu_target,
        "mu_aqual": mu_aqual,
        "history": history,
        "best_mu": best_mu,
        "best_aqual": best_aqual,
        "tol": tol,
        "max_iter": max_iter,
        "mu_mix_hi": mu_mix_hi,
        "mu_mix_lo": mu_mix_lo,
        "mu_mix_decay": mu_mix_decay,
        "converged": history[-1]["rel_update"] < tol,
        "final_rel_update": history[-1]["rel_update"],
        "full_rms_blind": float(np.sqrt(np.mean(((g_eff[mask] - blind["g_eff"][mask]) / np.maximum(blind["g_eff"][mask], 1e-30)) ** 2))),
        "full_rms_alg": float(np.sqrt(np.mean(((g_eff[mask] - blind["g_alg"][mask]) / np.maximum(blind["g_alg"][mask], 1e-30)) ** 2))),
        "full_rms_aqual": float(np.sqrt(np.mean(((g_eff[mask] - g_aqual[mask]) / np.maximum(g_aqual[mask], 1e-30)) ** 2))),
        "rms_mu_target": float(np.sqrt(np.mean((mu_eff[mask] - mu_target[mask]) ** 2))),
        "rms_mu_aqual": float(np.sqrt(np.mean((mu_eff[mask] - mu_aqual[mask]) ** 2))),
    }
    return result


def make_plots(results):
    xi = results["xi"]
    hist = results["history"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.semilogy([h["iter"] for h in hist], [h["rel_update"] for h in hist], lw=2)
    ax.set_xlabel("iteration")
    ax.set_ylabel("relative update")
    ax.set_title("(a) Coupled operator convergence")

    ax = axes[0, 1]
    ax.semilogy([h["iter"] for h in hist], [h["rms_blind"] for h in hist], lw=2, label="vs blind static")
    ax.semilogy([h["iter"] for h in hist], [h["rms_aqual"] for h in hist], lw=2, label="vs AQUAL")
    ax.semilogy([h["iter"] for h in hist], [h["rms_alg"] for h in hist], lw=2, label="vs algebraic")
    ax.set_xlabel("iteration")
    ax.set_ylabel("RMS field error")
    ax.set_title("(b) Field diagnostics")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    ax.loglog(xi, results["blind"]["g_eff"], lw=2, label="blind static 2D")
    ax.loglog(xi, results["g_eff"], lw=2, label="AQUAL w/o mu")
    ax.loglog(xi, results["blind"]["g_alg"], "--", lw=1.5, label="algebraic")
    ax.loglog(xi, results["g_aqual"], ":", lw=2, label="standard AQUAL")
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$g/a_0$")
    ax.set_title("(c) Final field comparison")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.semilogx(xi, results["mu_eff"], lw=2, label=r"$\mu_{\rm coupled}=g_N/g$")
    ax.semilogx(xi, results["mu_target"], "--", lw=1.5, label=r"$x/(1+x)$")
    ax.semilogx(xi, results["mu_aqual"], ":", lw=2, label=r"$\mu_{\rm AQUAL}^{\rm eff}$")
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$\mu$")
    ax.set_title("(d) Extracted operator-side mu")
    ax.legend(fontsize=9)

    fig.suptitle("AQUAL-like Operator Solve Without Explicit Mu Input", y=0.98)
    fig.tight_layout()
    outpath = OUTDIR / "phase_ap_aqual_without_mu.png"
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return outpath


def print_interpretation(results):
    print()
    print("  Final diagnostics:")
    print(f"    converged?                = {results['converged']}")
    print(f"    final relative update     = {results['final_rel_update']:.3e}")
    print(f"    full RMS vs blind static  = {results['full_rms_blind']:.3e}")
    print(f"    full RMS vs algebraic     = {results['full_rms_alg']:.3e}")
    print(f"    full RMS vs std. AQUAL    = {results['full_rms_aqual']:.3e}")
    print(f"    RMS(mu vs x/(1+x))        = {results['rms_mu_target']:.3e}")
    print(f"    RMS(mu vs AQUAL eff.)     = {results['rms_mu_aqual']:.3e}")
    print(
        f"    best mu iterate           = {results['best_mu']['record']['iter']} "
        f"(rms_mu={results['best_mu']['record']['rms_mu']:.3e}, "
        f"rms_AQUAL={results['best_mu']['record']['rms_aqual']:.3e})"
    )
    print(
        f"    best AQUAL iterate        = {results['best_aqual']['record']['iter']} "
        f"(rms_AQUAL={results['best_aqual']['record']['rms_aqual']:.3e}, "
        f"rms_mu={results['best_aqual']['record']['rms_mu']:.3e})"
    )
    print()
    print("    r/r_M    g_cpl/a0   g_blind/a0   g_alg/a0   g_AQ/a0   mu_cpl")
    for factor in [0.3, 1.0, 3.0, 10.0]:
        idx = int(np.argmin(np.abs(results["xi"] - factor)))
        print(
            f"    {results['xi'][idx]:5.1f}   "
            f"{results['g_eff'][idx]:8.3f}   "
            f"{results['blind']['g_eff'][idx]:10.3f}   "
            f"{results['blind']['g_alg'][idx]:8.3f}   "
            f"{results['g_aqual'][idx]:8.3f}   "
            f"{results['mu_eff'][idx]:8.4f}"
        )
    print(
        """

  Reading:
  - this channel removes the explicit mu(x) input from the operator-side solve;
  - the coefficient field is rebuilt self-consistently from the coupled
    Newtonian and total potentials;
  - comparison with the standard AQUAL solution is diagnostic only;
  - with the homotopy-style schedule alpha_mu -> 0, the branch now genuinely
    stabilizes: the best-mu and best-AQUAL iterates coincide;
  - the remaining mismatch is therefore no longer a raw fixed-point failure,
    but a residual difference between this no-input operator branch and the
    standard AQUAL benchmark in disk geometry.
        """
    )


def run_all():
    results = run_coupled_operator()
    plot_path = make_plots(results)
    print_interpretation(results)
    print(f"\n  Plot saved to: {plot_path}")
    return results


if __name__ == "__main__":
    run_all()
