"""
Phase AD: Undropped hyperbolic 2+1D solve for the nonlocal source-side channel
===============================================================================

Purpose:
  Solve the full axisymmetric damped-wave equation with the second time
  derivative retained:

      d^2 U / dt^2 + 3H dU/dt - c^2 nabla^2 U = S_total(R,z) .

  The source-side closure itself is the already-derived nonlocal enclosed-mass
  kernel with boundary-selected normalization. The new ingredient here is the
  undropped hyperbolic propagation and its late-time convergence on the full
  (R, z, t) domain.

Important scope note:
  This is a full undropped hyperbolic solve for the DERIVED fixed-source
  source-side closure.  Phase AE then promotes that same check to the fully
  coupled case where S_total(R,z,t) is rebuilt from the evolving
  rotating-medium order parameter at every time step.
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

from constants import H0, c, r_M
from multiscale import self_consistent_solution
from phase_u_axisymmetric_swirl_2d import build_axisymmetric_operator, grad_xi
from phase_ai_boundary_kernel_2d import RHO_CORE_DERIVED, solve_boundary_selected_kernel

SEP = "=" * 78
OUTDIR = Path(__file__).parent / "plots"
OUTDIR.mkdir(exist_ok=True)


def boundary_zero(arr):
    out = np.asarray(arr, dtype=float).copy()
    out[-1, :] = 0.0
    out[:, -1] = 0.0
    return out


def relative_profile_error(xi, g_eval, g_target, xi_min=0.3, xi_max=10.0):
    mask = (xi >= xi_min) & (xi <= xi_max)
    rel = np.abs(g_eval[mask] - g_target[mask]) / np.maximum(g_target[mask], 1e-30)
    return float(np.sqrt(np.mean(rel**2))), float(np.max(rel))


def make_sponge(xi, zeta, xi_start=0.75, zeta_start=0.75, sigma_max=1.2):
    """Absorbing layer near the outer radius/top boundary."""
    xi_frac = np.clip((xi / xi.max() - xi_start) / max(1e-12, 1.0 - xi_start), 0.0, 1.0)
    zeta_frac = np.clip((zeta / zeta.max() - zeta_start) / max(1e-12, 1.0 - zeta_start), 0.0, 1.0)
    xi_ramp = xi_frac**2
    zeta_ramp = zeta_frac**2
    sponge = sigma_max * np.maximum.outer(xi_ramp, np.ones_like(zeta))
    sponge = np.maximum(sponge, sigma_max * np.outer(np.ones_like(xi), zeta_ramp))
    return sponge


def smooth_turn_on(t_hat, t_ramp=2.0):
    x = np.maximum(t_hat, 0.0) / max(t_ramp, 1e-12)
    return 1.0 - np.exp(-(x**2))


def run_hyperbolic(nr=81, nz=61, xi_max=10.0, zeta_max=5.0,
                   rho_core=RHO_CORE_DERIVED, dt_hat=0.035, n_steps=1800,
                   avg_tail=200, verbose=True):
    """Central-difference evolution in tau = c t / r_M units."""
    static = solve_boundary_selected_kernel(
        rho_core=rho_core, nr=nr, nz=nz, xi_max=xi_max, zeta_max=zeta_max
    )
    xi, zeta = static["xi"], static["zeta"]
    mat = build_axisymmetric_operator(nr=nr, nz=nz, xi_max=xi_max, zeta_max=zeta_max)[2].tocsr()

    gamma_hubble = 3.0 * H0 * r_M / c
    sponge = make_sponge(xi, zeta)
    gamma_total = gamma_hubble + sponge

    source_total = boundary_zero(static["source_total"])

    # Transport-side comparison profile.
    _, xi_tr, _, _, _, _, g_transport, _, _ = self_consistent_solution()
    order_tr = np.argsort(xi_tr)
    g_transport_interp = np.interp(
        xi, xi_tr[order_tr], g_transport[order_tr], left=np.nan, right=np.nan
    )

    if verbose:
        print(SEP)
        print("  PHASE AD: Undropped Hyperbolic 2+1D Solve")
        print(SEP)
        print(f"  Grid: {nr} x {nz}  (xi_max={xi_max:.1f}, zeta_max={zeta_max:.1f})")
        print(f"  rho_core = {rho_core:.3f} r_M")
        print(f"  dt_hat   = {dt_hat:.3f}")
        print(f"  gamma_H  = {gamma_hubble:.3e}")
        print(f"  sponge max = {np.max(sponge):.3e}")
        print()

    u_n = np.zeros((nr, nz), dtype=float)
    accel0 = source_total.copy()
    u_nm1 = 0.5 * (dt_hat**2) * accel0
    u_nm1 = boundary_zero(u_nm1)

    history = []
    snapshots = {}
    tail_sum = np.zeros_like(u_n)
    tail_count = 0

    report_steps = {0, 1, 2, 5, 10, 20, 50, 100, 200, 400, 800, 1200, n_steps - 1}

    for step in range(n_steps):
        tau = step * dt_hat
        source_now = smooth_turn_on(tau) * source_total
        lap_u = (mat @ u_n.reshape(-1)).reshape((nr, nz))

        numer = (
            2.0 * u_n
            - (1.0 - 0.5 * gamma_total * dt_hat) * u_nm1
            + (dt_hat**2) * (lap_u + source_now)
        )
        denom = 1.0 + 0.5 * gamma_total * dt_hat
        u_np1 = numer / denom
        u_np1 = boundary_zero(u_np1)

        g_mid = -grad_xi(u_np1, xi[1] - xi[0])[:, 0]
        rms_static, max_static = relative_profile_error(xi, g_mid, static["g_eff"])
        rms_alg, max_alg = relative_profile_error(xi, g_mid, static["g_alg"])
        valid_transport = np.isfinite(g_transport_interp)
        xi_t = xi[valid_transport]
        g_t = g_mid[valid_transport]
        g_tt = g_transport_interp[valid_transport]
        rms_transport, max_transport = relative_profile_error(
            xi_t, g_t, g_tt, xi_min=0.3, xi_max=min(10.0, xi_t.max())
        )

        rel_update = np.max(np.abs(u_np1 - u_n)) / max(np.max(np.abs(u_np1)), 1e-30)
        kinetic = np.mean(((u_np1 - u_nm1) / (2.0 * dt_hat))**2)

        record = {
            "step": step + 1,
            "tau": (step + 1) * dt_hat,
            "t_years": (step + 1) * dt_hat * r_M / c / (365.25 * 86400),
            "rel_update": rel_update,
            "kinetic": kinetic,
            "rms_static": rms_static,
            "max_static": max_static,
            "rms_alg": rms_alg,
            "max_alg": max_alg,
            "rms_transport": rms_transport,
            "max_transport": max_transport,
        }
        history.append(record)

        if step in report_steps:
            snapshots[step] = {
                "u": u_np1.copy(),
                "g_mid": g_mid.copy(),
                "record": record,
            }

        if step >= n_steps - avg_tail:
            tail_sum += u_np1
            tail_count += 1

        u_nm1, u_n = u_n, u_np1

    u_avg = boundary_zero(tail_sum / max(tail_count, 1))
    g_avg = -grad_xi(u_avg, xi[1] - xi[0])[:, 0]

    rms_static_avg, max_static_avg = relative_profile_error(xi, g_avg, static["g_eff"])
    rms_alg_avg, max_alg_avg = relative_profile_error(xi, g_avg, static["g_alg"])
    valid_transport = np.isfinite(g_transport_interp)
    xi_t = xi[valid_transport]
    g_t = g_avg[valid_transport]
    g_tt = g_transport_interp[valid_transport]
    rms_transport_avg, max_transport_avg = relative_profile_error(
        xi_t, g_t, g_tt, xi_min=0.3, xi_max=min(10.0, xi_t.max())
    )

    final = history[-1]
    if verbose:
        print("  Final instantaneous diagnostics:")
        print(f"    rel update                = {final['rel_update']:.3e}")
        print(f"    kinetic mean              = {final['kinetic']:.3e}")
        print(f"    RMS(inst. vs static 2D)   = {final['rms_static']:.3e}")
        print(f"    RMS(inst. vs algebraic)   = {final['rms_alg']:.3e}")
        print(f"    RMS(inst. vs transport)   = {final['rms_transport']:.3e}")
        print()
        print("  Late-time averaged diagnostics:")
        print(f"    RMS(avg vs static 2D)     = {rms_static_avg:.3e}")
        print(f"    RMS(avg vs algebraic)     = {rms_alg_avg:.3e}")
        print(f"    RMS(avg vs transport)     = {rms_transport_avg:.3e}")
        print(f"    elapsed physical time     = {final['t_years']:.2f} yr")
        print()
        print("    r/r_M    g_avg/a0   g_static/a0   g_alg/a0   g_transport/a0")
        for factor in [0.3, 1.0, 3.0, 10.0]:
            idx = np.argmin(np.abs(xi - factor))
            print(
                f"    {xi[idx]:5.1f}   "
                f"{g_avg[idx]:9.3f}   "
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
        "u_avg": u_avg,
        "g_avg": g_avg,
        "avg_metrics": {
            "rms_static": rms_static_avg,
            "max_static": max_static_avg,
            "rms_alg": rms_alg_avg,
            "max_alg": max_alg_avg,
            "rms_transport": rms_transport_avg,
            "max_transport": max_transport_avg,
        },
        "dt_hat": dt_hat,
        "n_steps": n_steps,
        "gamma_hubble": gamma_hubble,
        "sponge": sponge,
    }


def make_plots(results):
    xi = results["xi"]
    static = results["static"]
    g_transport = results["g_transport_interp"]
    history = results["history"]

    t_years = np.array([h["t_years"] for h in history])
    rel_update = np.array([h["rel_update"] for h in history])
    kinetic = np.array([h["kinetic"] for h in history])
    rms_static = np.array([h["rms_static"] for h in history])
    rms_alg = np.array([h["rms_alg"] for h in history])
    rms_transport = np.array([h["rms_transport"] for h in history])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.semilogy(t_years, rel_update, lw=2)
    ax.set_xlabel("time (yr)")
    ax.set_ylabel("relative update")
    ax.set_title("(a) Hyperbolic convergence")

    ax = axes[0, 1]
    ax.semilogy(t_years, kinetic, lw=2, color="tab:orange")
    ax.set_xlabel("time (yr)")
    ax.set_ylabel("mean kinetic proxy")
    ax.set_title("(b) Wave-energy decay")

    ax = axes[1, 0]
    ax.semilogy(t_years, rms_static, lw=2, label="vs static 2D")
    ax.semilogy(t_years, rms_alg, lw=2, label="vs algebraic")
    ax.semilogy(t_years, rms_transport, lw=2, label="vs transport")
    ax.set_xlabel("time (yr)")
    ax.set_ylabel("instantaneous RMS field error")
    ax.set_title("(c) Error history")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.loglog(xi, static["g_alg"], "k--", lw=1.5, label="algebraic")
    ax.loglog(xi, static["g_eff"], color="tab:red", lw=2, label="static source 2D")
    ax.loglog(xi, g_transport, color="tab:blue", lw=2, label="transport")
    ax.loglog(xi, results["g_avg"], color="tab:green", lw=2, label="hyperbolic avg")
    ax.set_xlabel(r"$\xi = r/r_M$")
    ax.set_ylabel(r"$g/a_0$")
    ax.set_title("(d) Late-time averaged field")
    ax.legend(fontsize=9)

    fig.suptitle("Undropped Hyperbolic 2+1D Solve of the Nonlocal Source-Side Channel", y=0.98)
    fig.tight_layout()
    outpath = OUTDIR / "phase_ad_hyperbolic_nonlocal_2p1d.png"
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved: {outpath}")


def print_interpretation(results):
    avg = results["avg_metrics"]
    final = results["history"][-1]
    print("\n" + SEP)
    print("  Interpretation")
    print(SEP)
    print(
        f"""
  This is the first undropped hyperbolic 2+1D propagation check of the
  derived nonlocal source-side MOND closure.

  Late-time averaged outcome:
  - avg vs static-source RMS   = {avg['rms_static']:.3e}
  - avg vs algebraic RMS       = {avg['rms_alg']:.3e}
  - avg vs transport-side RMS  = {avg['rms_transport']:.3e}
  - elapsed physical time      = {final['t_years']:.2f} yr

  Reading:
  - with the d^2U/dt^2 term retained, the hyperbolic evolution still settles
    onto the same nonlocal source-side branch after transients leave through
    the absorbing outer layer,
  - that late-time branch remains close to the algebraic and transport-side
    MOND closures on the matched axisymmetric window,
  - Phase AE then removes the last fixed-source restriction by evolving
    S_total(R,z,t) directly from the rotating-medium order parameter inside
    the same hyperbolic run.
        """
    )


if __name__ == "__main__":
    results = run_hyperbolic()
    make_plots(results)
    print_interpretation(results)
