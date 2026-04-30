"""
Physical constants, galaxy model, and derived quantities for the
MOND PDE numerical verification (ISPG Appendix 12).

All quantities in SI unless noted otherwise.
Dimensionless quantities follow the conventions of ISPG_MOND.tex §11.
"""

import numpy as np

# =====================================================================
# I.  Fundamental physical constants (SI)
# =====================================================================
G    = 6.67430e-11        # gravitational constant  [m^3 kg^-1 s^-2]
c    = 2.99792458e8       # speed of light          [m/s]
H0   = 67.4e3 / 3.0857e22  # Hubble constant (67.4 km/s/Mpc)  [s^-1]
Msun = 1.98848e30         # solar mass              [kg]
kpc  = 3.08568e19         # kiloparsec              [m]
yr   = 3.15576e7          # Julian year             [s]
Gyr  = 1e9 * yr           # gigayear                [s]

# =====================================================================
# II.  Galaxy model (fiducial: Milky-Way-like exponential disk)
# =====================================================================
M_gal = 1.0e11 * Msun     # total baryonic mass          [kg]
R_d   = 10.0  * kpc        # disk scale length            [m]
h_d   = 0.5   * kpc        # disk scale height            [m]

# =====================================================================
# III.  Critical acceleration
# =====================================================================
a0      = c * H0 / (2 * np.pi)   # predicted MOND scale  [m/s^2]
a0_obs  = 1.2e-10                 # observed (McGaugh 2016)  [m/s^2]

# =====================================================================
# IV.  Derived scales
# =====================================================================
r_M   = np.sqrt(G * M_gal / a0)          # MOND transition radius  [m]
r_s   = 2 * G * M_gal / c**2             # Schwarzschild radius    [m]
eps   = 3 * np.pi * r_s / r_M            # dimensionless ε (eq:epsilon)

# Coherence length and Hubble radius
lambda_H = 2 * np.pi * c / H0            # coherence length = c/a0  [m]
r_H      = c / H0                         # Hubble radius           [m]

# =====================================================================
# V.  Timescales
# =====================================================================
T_H = 1.0 / H0                            # Hubble time            [s]

# Keplerian orbital period at r_M
v_circ_rM = np.sqrt(G * M_gal / r_M)      # circular velocity at r_M  [m/s]
Omega_rM  = v_circ_rM / r_M               # angular velocity at r_M   [rad/s]
T_gal     = 2 * np.pi / Omega_rM          # orbital period at r_M     [s]

# Number of orbits in a Hubble time
N_rot = T_H / T_gal

# =====================================================================
# VI.  Transport coefficients (self-consistent, see wkb_closure.py)
# =====================================================================
Omega_tr_conj = a0 / c                    # transport angular velocity  [rad/s]
tau_rel_rM    = c / (G * M_gal / r_M**2)  # relaxation time at r_M     [s]
                                           # (= c / g_N(r_M))

# =====================================================================
# VII.  Computational domain (dimensionless)
# =====================================================================
xi_min = 1e-2     # inner boundary  (well inside Newtonian regime)
xi_max = 1e2      # outer boundary  (deep MOND)
N_cheb = 200      # default Chebyshev order

s_min  = np.log(xi_min)   # log-coordinate lower bound
s_max  = np.log(xi_max)   # log-coordinate upper bound


# =====================================================================
#  Print everything
# =====================================================================
def print_constants():
    """Pretty-print all constants, galaxy parameters, and derived quantities."""
    sep = "=" * 65

    print(sep)
    print("  MOND PDE — Constants & Derived Quantities")
    print(sep)

    print("\n--- I. Fundamental constants (SI) ---")
    print(f"  G      = {G:.5e}  m^3 kg^-1 s^-2")
    print(f"  c      = {c:.8e}  m/s")
    print(f"  H0     = {H0:.5e}  s^-1  ({67.4} km/s/Mpc)")
    print(f"  M_sun  = {Msun:.5e}  kg")
    print(f"  kpc    = {kpc:.5e}  m")

    print("\n--- II. Galaxy model ---")
    print(f"  M_gal  = {M_gal/Msun:.1e} M_sun  = {M_gal:.4e} kg")
    print(f"  R_d    = {R_d/kpc:.1f} kpc  = {R_d:.4e} m")
    print(f"  h_d    = {h_d/kpc:.1f} kpc  = {h_d:.4e} m")

    print("\n--- III. Critical acceleration ---")
    print(f"  a0 (predicted)  = {a0:.5e}  m/s^2")
    print(f"  a0 (observed)   = {a0_obs:.5e}  m/s^2")
    print(f"  ratio pred/obs  = {a0/a0_obs:.4f}")

    print("\n--- IV. Derived scales ---")
    print(f"  r_M      = {r_M:.4e} m  = {r_M/kpc:.2f} kpc")
    print(f"  r_s      = {r_s:.4e} m")
    print(f"  epsilon  = {eps:.4e}  (= 3*pi*r_s/r_M)")
    print(f"  lambda_H = {lambda_H:.4e} m  = {lambda_H/kpc:.1f} kpc")
    print(f"  r_H      = {r_H:.4e} m  = {r_H/(1e6*kpc):.0f} Mpc")

    print("\n--- V. Timescales ---")
    print(f"  T_H      = {T_H:.4e} s  = {T_H/Gyr:.2f} Gyr")
    print(f"  v_circ(r_M) = {v_circ_rM/1e3:.2f} km/s")
    print(f"  Omega(r_M)  = {Omega_rM:.4e} rad/s")
    print(f"  T_gal    = {T_gal:.4e} s  = {T_gal/Gyr:.3f} Gyr")
    print(f"  N_rot    = {N_rot:.1f}  orbits in T_H")

    print("\n--- VI. Transport coefficients (self-consistent) ---")
    print(f"  Omega_tr = a0/c = {Omega_tr_conj:.4e} rad/s")
    print(f"  tau_rel(r_M)     = c/g  = {tau_rel_rM:.4e} s = {tau_rel_rM/Gyr:.2f} Gyr")
    print(f"  Omega_tr * tau_rel(r_M) = {Omega_tr_conj * tau_rel_rM:.4e}")
    g_N_rM = G * M_gal / r_M**2
    print(f"  a0 / g_N(r_M)           = {a0 / g_N_rM:.4e}  (should match above)")

    print("\n--- VII. Computational domain ---")
    print(f"  xi_min = {xi_min:.0e}   (s_min = {s_min:.2f})")
    print(f"  xi_max = {xi_max:.0e}   (s_max = {s_max:.2f})")
    print(f"  N_cheb = {N_cheb}")

    # Sanity checks
    print(f"\n--- Sanity checks ---")
    print(f"  eps ~ 3*pi*r_s/r_M          = {3*np.pi*r_s/r_M:.4e}  [OK]")
    print(f"  eps ~ 3*H*r_M/c             = {3*H0*r_M/c:.4e}  [OK]")
    print(f"  r_M^2 * a0 / (G*M)          = {r_M**2 * a0 / (G*M_gal):.6f}  (should be 1)")
    print(f"  a0 * lambda_H / c^2         = {a0 * lambda_H / c**2:.6f}  (should be 1)")

    print(f"\n{sep}")
    print("  Step 1.1 complete.")
    print(sep)


if __name__ == "__main__":
    print_constants()
