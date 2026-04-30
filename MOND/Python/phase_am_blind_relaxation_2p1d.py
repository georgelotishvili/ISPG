"""
Phase AM: blind time-dependent relaxation on the self-calibrated 2D kernel
===========================================================================

Goal:
  Take the strong blind static benchmark of Phase AL and verify that the
  quasi-static 2+1D relaxation PDE converges dynamically to the same branch,
  without ever feeding a pointwise algebraic mu input into the source.

Equation:
    dU/dtau - L[U] = S_total(R,z),

with S_total inherited from the blind self-calibrated nonlocal 2D kernel.
The algebraic and AQUAL branches are used only as external diagnostics.
"""

from pathlib import Path
import io
import sys

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from constants import H0, c, r_M
from multiscale import self_consistent_solution
from phase_u_axisymmetric_swirl_2d import build_axisymmetric_operator, grad_xi
from phase_al_blind_selfcalibrated_kernel_2d import solve_blind_selfcalibrated_kernel_2d


SEP = "=" * 78
OUTDIR = Path(__file__).parent / "plots"
OUTDIR.mkdir(exist_ok=True)


def physical_time_unit():
    """Natural quasi-static relaxation unit from 3H U_t - c^2 ΔU = S."""
    return 3.0 * H0 * r_M**2 / c**2


def boundary_zero(arr):
    arr = np.asarray(arr, dtype=float).copy()
    arr[-1, :] = 0.0
    arr[:, -1] = 0.0
    return arr


def relative_profile_error(xi, g_eval, g_target, xi_min=0.3, xi_max=10.0):
    mask = (xi >= xi_min) & (xi <= xi_max)
    rel = np.abs(g_eval[mask] - g_target[mask]) / np.maximum(g_target[mask], 1e-30)
    return float(np.sqrt(np.mean(rel**2))), float(np.max(rel))


def run_relaxation(
    nr=121,
    nz=91,
    xi_max=20.0,
    zeta_max=10.0,
    dt_hat=0.4,
    n_steps=400,
    snapshot_steps=(0, 1, 2, 5, 10, 20, 40, 80, 120, 180, 240, 320, 399),
):
    static = solve_blind_selfcalibrated_kernel_2d(
        nr=nr,
        nz=nz,
        xi_max=xi_max,
        zeta_max=zeta_max,
    )

    xi, zeta = static["xi"], static["zeta"]
    mat = build_axisymmetric_operator(
        nr=nr, nz=nz, xi_max=xi_max, zeta_max=zeta_max
    )[2]
    source_total = boundary_zero(static["source_total"])
    source_vec = source_total.reshape(-1)

    ident = sparse.identity(nr * nz, format="csc")
    step_matrix = ident - dt_hat * mat
    step_solver = factorized(step_matrix)

    u = np.zeros((nr, nz), dtype=float)
    history = []
    snapshots = {}
    time_unit_s = physical_time_unit()
    dxi = xi[1] - xi[0]

    _, xi_tr, _, _, _, _, g_transport, _, _ = self_consistent_solution()
    order_tr = np.argsort(xi_tr)
    g_transport_interp = np.interp(
        xi, xi_tr[order_tr], g_transport[order_tr], left=np.nan, right=np.nan
    )

    print(SEP)
    print("  PHASE AM: Blind Time-Dependent Relaxation")
    print(SEP)
    print(f"  Grid: {nr} x {nz}  (xi_max={xi_max:.1f}, zeta_max={zeta_max:.1f})")
    print(f"  dt_hat   = {dt_hat:.3f}")
    print(f"  time unit = {time_unit_s/86400:.2f} days")
    print(f"  physical dt = {dt_hat * time_unit_s / (365.25*86400):.3f} yr")
    print(f"  inherited scale_N = {static['scale_N']:.6f}")
    print(f"  inherited kappa_G = {static['kappa_G']:.6f}")
    print()

    for step in range(n_steps):
        rhs = u.reshape(-1) + dt_hat * source_vec
        for i in range(nr):
            rhs[i * nz + (nz - 1)] = 0.0
        for j in range(nz):
            rhs[(nr - 1) * nz + j] = 0.0

        u_new = step_solver(rhs).reshape((nr, nz))
        u_new = boundary_zero(u_new)

        rel_update = np.max(np.abs(u_new - u)) / max(np.max(np.abs(u_new)), 1e-30)
        u = u_new

        g_mid = -grad_xi(u, dxi)[:, 0]
        rms_static, max_static = relative_profile_error(xi, g_mid, static["g_eff"])
        rms_alg, max_alg = relative_profile_error(xi, g_mid, static["g_alg"])
        rms_aqual, max_aqual = relative_profile_error(xi, g_mid, static["g_aqual"])

        valid_transport = np.isfinite(g_transport_interp)
        xi_t = xi[valid_transport]
        g_t = g_mid[valid_transport]
        g_tt = g_transport_interp[valid_transport]
        if len(xi_t) > 0:
            rms_transport, max_transport = relative_profile_error(
                xi_t, g_t, g_tt, xi_min=0.3, xi_max=min(10.0, xi_t.max())
            )
        else:
            rms_transport, max_transport = np.nan, np.nan

        record = {
            "step": step + 1,
            "t_hat": (step + 1) * dt_hat,
            "t_years": (step + 1) * dt_hat * time_unit_s / (365.25 * 86400),
            "rel_update": rel_update,
            "rms_static": rms_static,
            "max_static": max_static,
            "rms_alg": rms_alg,
            "max_alg": max_alg,
            "rms_aqual": rms_aqual,
            "max_aqual": max_aqual,
            "rms_transport": rms_transport,
            "max_transport": max_transport,
        }
        history.append(record)

        if step in snapshot_steps:
            snapshots[step] = {
                "u": u.copy(),
                "g_mid": g_mid.copy(),
                "record": record,
            }

    final = history[-1]
    print("  Final diagnostics:")
    print(f"    rel update                = {final['rel_update']:.3e}")
    print(f"    RMS(dynamic vs blind 2D)  = {final['rms_static']:.3e}")
    print(f"    RMS(dynamic vs algebraic) = {final['rms_alg']:.3e}")
    print(f"    RMS(dynamic vs AQUAL)     = {final['rms_aqual']:.3e}")
    print(f"    RMS(dynamic vs transport) = {final['rms_transport']:.3e}")
    print(f"    elapsed physical time     = {final['t_years']:.2f} yr")
    print()
    print("    r/r_M    g_dyn/a0   g_blind/a0   g_alg/a0   g_AQUAL/a0")
    for factor in [0.3, 1.0, 3.0, 10.0]:
        idx = np.argmin(np.abs(xi - factor))
        print(
            f"    {xi[idx]:5.1f}   "
            f"{snapshots[max(snapshots.keys())]['g_mid'][idx]:9.3f}   "
            f"{static['g_eff'][idx]:10.3f}   "
            f"{static['g_alg'][idx]:8.3f}   "
            f"{static['g_aqual'][idx]:10.3f}"
        )

    return {
        "xi": xi,
        "zeta": zeta,
        "static": static,
        "g_transport_interp": g_transport_interp,
        "history": history,
        "snapshots": snapshots,
        "time_unit_s": time_unit_s,
        "dt_hat": dt_hat,
        "n_steps": n_steps,
    }


def make_plots(results):
    xi = results["xi"]
    hist = results["history"]
    static = results["static"]
    g_transport = results["g_transport_interp"]

    t_years = np.array([h["t_years"] for h in hist])
    rms_static = np.array([h["rms_static"] for h in hist])
    rms_alg = np.array([h["rms_alg"] for h in hist])
    rms_aqual = np.array([h["rms_aqual"] for h in hist])
    rms_transport = np.array([h["rms_transport"] for h in hist])
    rel_update = np.array([h["rel_update"] for h in hist])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.semilogy(t_years, rel_update, lw=2)
    ax.set_xlabel("time (yr)")
    ax.set_ylabel("relative update")
    ax.set_title("(a) Blind relaxation convergence")

    ax = axes[0, 1]
    ax.semilogy(t_years, rms_static, lw=2, label="vs blind static 2D")
    ax.semilogy(t_years, rms_alg, lw=2, label="vs algebraic")
    ax.semilogy(t_years, rms_aqual, lw=2, label="vs AQUAL")
    ax.semilogy(t_years, rms_transport, lw=2, label="vs transport")
    ax.set_xlabel("time (yr)")
    ax.set_ylabel("RMS field error")
    ax.set_title("(b) Midplane error history")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    ax.loglog(xi, static["g_alg"], "k--", lw=1.5, label="algebraic")
    ax.loglog(xi, static["g_aqual"], ":", color="tab:purple", lw=2, label="AQUAL")
    ax.loglog(xi, static["g_eff"], color="tab:red", lw=2, label="blind static 2D")
    ax.loglog(xi, g_transport, color="tab:blue", lw=2, label="transport")
    last_snapshot = results["snapshots"][max(results["snapshots"].keys())]
    ax.loglog(xi, last_snapshot["g_mid"], color="tab:green", lw=2, label="blind dynamic")
    ax.set_xlabel(r"$\xi = r/r_M$")
    ax.set_ylabel(r"$g/a_0$")
    ax.set_title("(c) Final midplane field")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    for step in sorted(results["snapshots"].keys()):
        snap = results["snapshots"][step]
        if step not in {0, 5, 20, max(results["snapshots"].keys())}:
            continue
        ax.loglog(
            xi,
            snap["g_mid"],
            lw=1.5,
            label=f"{snap['record']['t_years']:.1f} yr",
        )
    ax.loglog(xi, static["g_eff"], "k--", lw=1.5, label="blind static 2D")
    ax.set_xlabel(r"$\xi = r/r_M$")
    ax.set_ylabel(r"$g/a_0$")
    ax.set_title("(d) Blind time snapshots")
    ax.legend(fontsize=9)

    fig.suptitle("Blind Time-Dependent Relaxation on the Self-Calibrated Nonlocal 2D Kernel", y=0.98)
    fig.tight_layout()
    outpath = OUTDIR / "phase_am_blind_relaxation_2p1d.png"
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {outpath}")


def print_interpretation(results):
    final = results["history"][-1]
    print("\n" + SEP)
    print("  Interpretation")
    print(SEP)
    print(
        f"""
  This calculation upgrades the blind static kernel of Phase AL to a genuine
  time-dependent 2+1D relaxation test.

  Numerical outcome:
  - dynamic vs blind static RMS  = {final['rms_static']:.3e}
  - dynamic vs algebraic RMS     = {final['rms_alg']:.3e}
  - dynamic vs AQUAL RMS         = {final['rms_aqual']:.3e}
  - dynamic vs transport RMS     = {final['rms_transport']:.3e}
  - elapsed physical time        = {final['t_years']:.2f} yr

  Reading:
  - the strong blind source construction is not just a static Poisson artifact:
    the time-dependent relaxation flows to the same branch,
  - that branch stays close to the same-baseline algebraic profile and the
    independent AQUAL benchmark,
  - so the remaining gap is again localized to the full nonlocal Green-kernel
    geometry rather than to any hidden pointwise mu insertion.
        """
    )


def run_all():
    results = run_relaxation()
    make_plots(results)
    print_interpretation(results)
    return results


if __name__ == "__main__":
    run_all()
