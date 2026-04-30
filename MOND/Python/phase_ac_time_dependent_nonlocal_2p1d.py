"""
Phase AC: Time-dependent 2+1D relaxation of the nonlocal source-side closure
=============================================================================

Purpose:
  Evolve the axisymmetric scalar field in (R, z, t) for the already-derived
  nonlocal source-side channel and verify that the time-dependent relaxation
  converges to the same MOND-supporting branch as the static 2D solve.

Important scope note:
  This is the first explicit time-dependent 2+1D AXISYMMETRIC RELAXATION solve
  of the derived source-side closure in the quasi-static galactic regime:

      3H dU/dt - c^2 nabla^2 U = S_total(R,z) .

  It does NOT yet claim a full undropped hyperbolic solve of

      d^2 U/dt^2 + 3H dU/dt - c^2 nabla^2 U = S_total(R,z,t) .
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

from constants import H0, c, r_M, kpc
from multiscale import self_consistent_solution
from phase_u_axisymmetric_swirl_2d import build_axisymmetric_operator, grad_xi
from phase_ai_boundary_kernel_2d import RHO_CORE_DERIVED, solve_boundary_selected_kernel

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


def run_relaxation(nr=81, nz=61, xi_max=10.0, zeta_max=5.0,
                   rho_core=RHO_CORE_DERIVED, dt_hat=0.4, n_steps=240,
                   snapshot_steps=(0, 1, 2, 5, 10, 20, 40, 80, 120, 180, 239)):
    """Implicit-Euler evolution of the quasi-static 2+1D relaxation PDE."""
    static = solve_boundary_selected_kernel(
        rho_core=rho_core, nr=nr, nz=nz, xi_max=xi_max, zeta_max=zeta_max
    )

    xi, zeta = static["xi"], static["zeta"]
    mat = build_axisymmetric_operator(nr=nr, nz=nz, xi_max=xi_max, zeta_max=zeta_max)[2]
    source_total = boundary_zero(static["source_total"])
    source_vec = source_total.reshape(-1)

    ident = sparse.identity(nr * nz, format="csc")
    step_matrix = ident - dt_hat * mat
    step_solver = factorized(step_matrix)

    u = np.zeros((nr, nz), dtype=float)
    history = []
    snapshots = {}
    time_unit_s = physical_time_unit()

    # Transport-side spectral comparison on the midplane.
    _, xi_tr, _, _, _, _, g_transport, _, _ = self_consistent_solution()
    order_tr = np.argsort(xi_tr)
    g_transport_interp = np.interp(
        xi, xi_tr[order_tr], g_transport[order_tr], left=np.nan, right=np.nan
    )

    print(SEP)
    print("  PHASE AC: Time-Dependent 2+1D Relaxation")
    print(SEP)
    print(f"  Grid: {nr} x {nz}  (xi_max={xi_max:.1f}, zeta_max={zeta_max:.1f})")
    print(f"  rho_core = {rho_core:.3f} r_M")
    print(f"  dt_hat   = {dt_hat:.3f}")
    print(f"  time unit = {time_unit_s/86400:.2f} days")
    print(f"  physical dt = {dt_hat * time_unit_s / (365.25*86400):.3f} yr")
    print()

    for step in range(n_steps):
        rhs = u.reshape(-1) + dt_hat * source_vec
        # Keep outer/top boundaries pinned to zero.
        for i in range(nr):
            rhs[i * nz + (nz - 1)] = 0.0
        for j in range(nz):
            rhs[(nr - 1) * nz + j] = 0.0

        u_new = step_solver(rhs).reshape((nr, nz))
        u_new = boundary_zero(u_new)

        rel_update = np.max(np.abs(u_new - u)) / max(np.max(np.abs(u_new)), 1e-30)
        u = u_new

        g_mid = -grad_xi(u, xi[1] - xi[0])[:, 0]
        rms_static, max_static = relative_profile_error(xi, g_mid, static["g_eff"])
        rms_alg, max_alg = relative_profile_error(xi, g_mid, static["g_alg"])
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
    print(f"    RMS(dynamic vs static 2D) = {final['rms_static']:.3e}")
    print(f"    RMS(dynamic vs algebraic) = {final['rms_alg']:.3e}")
    print(f"    RMS(dynamic vs transport) = {final['rms_transport']:.3e}")
    print(f"    elapsed physical time     = {final['t_years']:.2f} yr")
    print()
    print("    r/r_M    g_dyn/a0   g_static/a0   g_alg/a0   g_transport/a0")
    for factor in [0.3, 1.0, 3.0, 10.0]:
        idx = np.argmin(np.abs(xi - factor))
        print(
            f"    {xi[idx]:5.1f}   "
            f"{snapshots[max(snapshots.keys())]['g_mid'][idx]:9.3f}   "
            f"{static['g_eff'][idx]:11.3f}   "
            f"{static['g_alg'][idx]:8.3f}   "
            f"{g_transport_interp[idx]:14.3f}"
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
    rms_transport = np.array([h["rms_transport"] for h in hist])
    rel_update = np.array([h["rel_update"] for h in hist])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.semilogy(t_years, rel_update, lw=2)
    ax.set_xlabel("time (yr)")
    ax.set_ylabel("relative update")
    ax.set_title("(a) Relaxation convergence")

    ax = axes[0, 1]
    ax.semilogy(t_years, rms_static, lw=2, label="vs static 2D")
    ax.semilogy(t_years, rms_alg, lw=2, label="vs algebraic")
    ax.semilogy(t_years, rms_transport, lw=2, label="vs transport")
    ax.set_xlabel("time (yr)")
    ax.set_ylabel("RMS field error")
    ax.set_title("(b) Midplane error history")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    ax.loglog(xi, static["g_alg"], "k--", lw=1.5, label="algebraic")
    ax.loglog(xi, static["g_eff"], color="tab:red", lw=2, label="static source 2D")
    ax.loglog(xi, g_transport, color="tab:blue", lw=2, label="transport")
    last_snapshot = results["snapshots"][max(results["snapshots"].keys())]
    ax.loglog(xi, last_snapshot["g_mid"], color="tab:green", lw=2, label="time-dependent")
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
    ax.loglog(xi, static["g_alg"], "k--", lw=1.5, label="algebraic")
    ax.set_xlabel(r"$\xi = r/r_M$")
    ax.set_ylabel(r"$g/a_0$")
    ax.set_title("(d) Time snapshots")
    ax.legend(fontsize=9)

    fig.suptitle("Time-Dependent 2+1D Relaxation of the Nonlocal Source-Side Closure", y=0.98)
    fig.tight_layout()
    outpath = OUTDIR / "phase_ac_time_dependent_nonlocal_2p1d.png"
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
  This calculation is the first explicit time-dependent 2+1D axisymmetric
  relaxation solve of the derived nonlocal source-side closure.

  Numerical outcome:
  - dynamic vs static-source RMS   = {final['rms_static']:.3e}
  - dynamic vs algebraic RMS       = {final['rms_alg']:.3e}
  - dynamic vs transport-side RMS  = {final['rms_transport']:.3e}
  - elapsed physical time          = {final['t_years']:.2f} yr

  Reading:
  - the time-dependent relaxation flows toward the same branch selected by the
    static nonlocal source-side solve,
  - that late-time branch remains close to the algebraic and transport-side
    MOND closures,
  - and the remaining independent refinement is still the fully hyperbolic
    undropped solve with the d^2U/dt^2 term retained.
        """
    )


if __name__ == "__main__":
    results = run_relaxation()
    make_plots(results)
    print_interpretation(results)
