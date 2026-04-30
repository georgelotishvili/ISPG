"""Compute H_eff: time-averaged Hubble parameter over galaxy lifetime."""
import numpy as np

H0 = 67.4  # km/s/Mpc
H0_si = H0 * 1e3 / 3.0857e22  # s^-1
c_si = 2.998e8  # m/s
Omega_m = 0.315
Omega_L = 0.685
Gyr = 3.156e16  # seconds

def H_of_z(z):
    return H0_si * np.sqrt(Omega_m*(1+z)**3 + Omega_L)

def t_of_z(z, nsteps=10000):
    """Lookback time to redshift z."""
    zz = np.linspace(0, z, nsteps)
    integrand = 1.0 / ((1+zz) * np.array([H_of_z(zi) for zi in zz]))
    return np.trapz(integrand, zz)

t0 = t_of_z(50)
a0_theory = c_si * H0_si / (2 * np.pi)

print("Age of universe: %.2f Gyr" % (t0/Gyr))
print("H0 = %.4e s^-1" % H0_si)
print("a0(H0) = cH0/(2pi) = %.4e m/s^2" % a0_theory)
print("a0_obs = 1.20e-10 m/s^2")
print("Ratio needed: H_eff/H0 = %.4f" % (1.2e-10 / a0_theory))
print()

# KEY IDENTITY:
# H_eff = (1/T) * int_0^T H(t) dt
# Change variable t -> z: dt = dz / ((1+z)*H(z))
# int H(t) dt = int H * dz/((1+z)*H) = int dz/(1+z) = ln(1+z_form)
# T = t_of_z(z_form)
# Therefore: H_eff = ln(1+z_form) / t_of_z(z_form)

print("=" * 70)
print("  Vortex-lag effect: H_eff = <H>_time over galaxy lifetime")
print("=" * 70)
header = "%8s %12s %10s %14s %10s" % ("z_form", "t_form(Gyr)", "H_eff/H0", "a0_eff(m/s2)", "a0_eff/a0obs")
print(header)
print("-" * 70)

for z_form in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    T = t_of_z(z_form)
    t_form_gyr = (t0 - T) / Gyr  # when galaxy formed (cosmic age)
    H_eff = np.log(1 + z_form) / T
    a0_eff = c_si * H_eff / (2 * np.pi)
    ratio = H_eff / H0_si
    
    print("%8.1f %12.2f %10.4f %14.4e %10.4f" % (
        z_form, t_form_gyr, ratio, a0_eff, a0_eff / 1.2e-10))

print()
print("Finding z_form that gives a0_eff = 1.20e-10 exactly:")
target = 1.2e-10
best_z = None
best_err = 1e10
for z_test in np.arange(0.01, 8.0, 0.01):
    T = t_of_z(z_test)
    H_eff = np.log(1 + z_test) / T
    a0_eff = c_si * H_eff / (2 * np.pi)
    err = abs(a0_eff - target)
    if err < best_err:
        best_err = err
        best_z = z_test

T = t_of_z(best_z)
H_eff = np.log(1 + best_z) / T
a0_eff = c_si * H_eff / (2 * np.pi)
print("  z_form = %.2f (galaxy formed %.2f Gyr after Big Bang)" % (best_z, (t0 - T)/Gyr))
print("  H_eff/H0 = %.4f" % (H_eff / H0_si))
print("  a0_eff = %.4e m/s^2" % a0_eff)

# Also: what if we use EXPONENTIAL weighting (recent H matters more)?
print()
print("=" * 70)
print("  Alternative: exponentially-weighted H (memory kernel e^{-s/tau})")
print("  with tau = 2*pi/H0 (one coherence-cell crossing time)")
print("=" * 70)

tau_mem = 2 * np.pi / H0_si
nsteps = 50000
zz = np.linspace(0, 10, nsteps)
dz = zz[1] - zz[0]

# t(z) from z=0
tt = np.zeros(nsteps)
for i in range(1, nsteps):
    tt[i] = tt[i-1] + dz / ((1 + zz[i-1]) * H_of_z(zz[i-1]))

Hvals = np.array([H_of_z(z) for z in zz])
weight = np.exp(-tt / tau_mem)
H_eff_exp = np.trapz(Hvals * weight, tt) / np.trapz(weight, tt)
a0_exp = c_si * H_eff_exp / (2 * np.pi)
print("  tau_mem = 2pi/H0 = %.2f Gyr" % (tau_mem / Gyr))
print("  H_eff_exp/H0 = %.4f" % (H_eff_exp / H0_si))
print("  a0_exp = %.4e m/s^2 (ratio to 1.2e-10: %.4f)" % (a0_exp, a0_exp/1.2e-10))
