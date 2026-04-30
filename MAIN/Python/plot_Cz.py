"""
Plot C(z) and tau_proc(z) for Appendix 18.

Model:  phi_bg(z) = nu * ln(1+z)
        C(z) = exp[phi_bg(z)/2] = (1+z)^{nu/2}

nu is calibrated from the MOND paper benchmark:
  C(10) = 3.1  =>  nu/2 = ln(3.1)/ln(11) = 0.4718  =>  nu = 0.9436

H(z) uses standard LambdaCDM (the dynamic DE correction to the
Friedmann equation is sub-leading for the time integrals).
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- cosmological parameters ---
Om  = 0.315
OL  = 0.685
Or  = 9.15e-5
H0  = 67.4                        # km/s/Mpc
H0_invGyr = H0 * 1.0222e-3        # Gyr^{-1}

# --- process-time model ---
C_bench = 3.1
z_bench = 10.0
nu_half = np.log(C_bench) / np.log(1 + z_bench)   # nu/2
nu = 2 * nu_half

def Cz(z):
    return (1 + z)**nu_half

def Ez(z):
    """E(z) = H(z)/H0 in LambdaCDM"""
    return np.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + OL)

# --- redshift grid ---
z = np.linspace(0, 10, 5000)

C_arr = Cz(z)
E_arr = Ez(z)

# lookback time:  dt = dz / [(1+z) H(z)]
integrand_lb  = 1.0 / ((1+z) * E_arr * H0_invGyr)
# process time:   dtau = C(z) * dt
integrand_tau = C_arr / ((1+z) * E_arr * H0_invGyr)

t_lb   = np.concatenate([[0], cumulative_trapezoid(integrand_lb,  z)])
tau_pr = np.concatenate([[0], cumulative_trapezoid(integrand_tau, z)])

# average enhancement
A_arr = np.where(t_lb > 0, tau_pr / t_lb, 1.0)

print(f"nu = {nu:.4f},  nu/2 = {nu_half:.4f}")
print(f"C(0.58) = {Cz(0.58):.3f}")
print(f"C(2)    = {Cz(2.0):.3f}")
print(f"C(10)   = {Cz(10.0):.3f}")
print(f"t_lookback(10) = {t_lb[-1]:.2f} Gyr")
print(f"tau_proc(10)   = {tau_pr[-1]:.2f} Gyr")
print(f"Enhancement x  = {tau_pr[-1]/t_lb[-1]:.2f}")

# average enhancement at z=0.58
idx058 = np.argmin(np.abs(z - 0.58))
print(f"A(0.58) = tau/t  = {A_arr[idx058]:.3f}")

# ===================== FIGURE =====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))

# --- Panel (a): C(z) ---
ax1.plot(z, C_arr, 'k-', lw=2)
ax1.axhline(1, color='grey', ls='--', lw=0.7)
ax1.set_xlabel(r'$z$', fontsize=14)
ax1.set_ylabel(r'$\mathcal{C}(z) = (1{+}z)^{\,\nu/2}$', fontsize=13)
ax1.set_title(r'(a)  Clock-rate factor $\mathcal{C}(z)$', fontsize=12)
ax1.set_xlim(0, 10)
ax1.set_ylim(0.9, 3.5)

bpts = [(0.58, 1.24, (-60, 22)),
        (2.0,  None, (-55, 18)),
        (10.0, None, (-80, 12))]
for zb, _, offs in bpts:
    cv = Cz(zb)
    ax1.plot(zb, cv, 'ro', ms=5, zorder=5)
    ax1.annotate(rf'$z\!=\!{zb}$: {cv:.2f}',
                 xy=(zb, cv), fontsize=9,
                 xytext=offs, textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='red', lw=0.8),
                 color='red')

ax1.fill_between(z, 1, C_arr, where=C_arr >= 1, alpha=0.08, color='blue')
ax1.text(5, 1.55, r'past: $\mathcal{C}>1$' '\n' r'clocks faster',
         fontsize=9, ha='center', color='blue', style='italic')
ax1.grid(True, alpha=0.25)

# --- Panel (b): cumulative times ---
ax2.plot(z, t_lb,   'b-',  lw=2, label=r'$t_{\mathrm{lookback}}(z)$  (coordinate)')
ax2.plot(z, tau_pr,  'r-',  lw=2, label=r'$\tau_{\mathrm{proc}}(z)$  (process time)')
ax2.fill_between(z, t_lb, tau_pr, alpha=0.12, color='red')
ax2.set_xlabel(r'$z$', fontsize=14)
ax2.set_ylabel('Gyr', fontsize=13)
ax2.set_title(r'(b)  Coordinate vs.\ process time', fontsize=12)
ax2.set_xlim(0, 10)
ax2.legend(fontsize=10, loc='lower right')
ax2.grid(True, alpha=0.25)

# annotate enhancement
enh = tau_pr[-1] / t_lb[-1]
ax2.annotate(rf'${enh:.1f}\times$ enhancement',
             xy=(8.0, (t_lb[-1] + tau_pr[-1]) / 2),
             fontsize=11, color='red', fontweight='bold', ha='center')

plt.tight_layout()
plt.savefig('../PDF/fig_Cz_processtime.pdf', bbox_inches='tight')
plt.savefig('../PDF/fig_Cz_processtime.png', dpi=200, bbox_inches='tight')
print("\nSaved ../PDF/fig_Cz_processtime.pdf / .png")
