import numpy as np
from scipy.optimize import minimize, brentq, minimize_scalar
import sys
sys.stdout.reconfigure(line_buffering=True)

m_e, m_mu, m_tau = 0.51099895, 105.6583755, 1776.86

def koide(m1, m2, m3):
    return (m1+m2+m3) / (np.sqrt(m1)+np.sqrt(m2)+np.sqrt(m3))**2

Q_exp = koide(m_e, m_mu, m_tau)
print(f"Lepton Koide: Q = {Q_exp:.10f}")
print(f"Ratios: 1 : {m_mu/m_e:.1f} : {m_tau/m_e:.1f}")

def E(Om):
    if Om <= 0 or Om >= 1: return 0.0
    b2 = 1 - Om**2
    return b2**1.5 * (4*Om**2 + 1)

# Peak
res = minimize_scalar(lambda x: -E(x), bounds=(0.01, 0.99), method='bounded')
Om_peak = res.x
E_peak = E(Om_peak)
print(f"\nE peak: Omega={Om_peak:.6f}, E={E_peak:.6f}")

# ---- TEST 1: Grid scan for best Q=2/3 ----
print("\n--- SCAN: best Q=2/3 (free ratios) ---")
Oms = np.linspace(0.02, 0.998, 600)
Es = np.array([E(o) for o in Oms])

best_Q_diff = 1.0
best = None
N = len(Oms)
for i in range(N):
    if Es[i] <= 0: continue
    for j in range(i+1, N):
        if Es[j] <= 0: continue
        for k in range(j+1, N):
            if Es[k] <= 0: continue
            Q = koide(Es[i], Es[j], Es[k])
            d = abs(Q - 2/3)
            if d < best_Q_diff:
                best_Q_diff = d
                best = (Oms[i], Oms[j], Oms[k], Es[i], Es[j], Es[k], Q)

Om1, Om2, Om3, E1, E2, E3, Q = best
Es_s = sorted([E1, E2, E3])
print(f"Q = {Q:.10f}  |Q-2/3| = {abs(Q-2/3):.2e}")
print(f"Omega: {Om1:.6f}, {Om2:.6f}, {Om3:.6f}")
print(f"Ratios: 1 : {Es_s[1]/Es_s[0]:.1f} : {Es_s[2]/Es_s[0]:.1f}")

# ---- TEST 2: Fix lepton ratios, find Omega values ----
print("\n--- FIXED lepton ratios ---")

def find_Om_right(E_target):
    if E_target >= E_peak or E_target <= 0: return None
    return brentq(lambda x: E(x) - E_target, Om_peak, 0.99999, xtol=1e-15)

def find_Om_left(E_target):
    if E_target >= E_peak or E_target <= 0: return None
    return brentq(lambda x: E(x) - E_target, 0.001, Om_peak, xtol=1e-15)

# Tau on left branch, mu and e on right branch
for Om_tau in np.linspace(0.05, Om_peak-0.01, 200):
    Et = E(Om_tau)
    Em = Et * m_mu / m_tau
    Ee = Et * m_e / m_tau
    if Em >= E_peak or Ee >= E_peak: continue
    Om_mu = find_Om_right(Em)
    Om_e = find_Om_right(Ee)
    if Om_mu and Om_e:
        Q = koide(Ee, Em, Et)
        print(f"Om_tau={Om_tau:.4f} Om_mu={Om_mu:.6f} Om_e={Om_e:.8f}"
              f"  E_tau={Et:.6f} E_mu={Em:.6f} E_e={Ee:.8f}"
              f"  Q={Q:.10f}")
        break  # just show one example

# Tau on right branch too
for Om_tau in np.linspace(Om_peak+0.01, 0.95, 200):
    Et = E(Om_tau)
    Em = Et * m_mu / m_tau
    Ee = Et * m_e / m_tau
    if Em >= E_peak or Ee >= E_peak: continue
    Om_mu = find_Om_right(Em)
    Om_e = find_Om_right(Ee)
    if Om_mu and Om_e:
        Q = koide(Ee, Em, Et)
        print(f"Om_tau={Om_tau:.4f} Om_mu={Om_mu:.6f} Om_e={Om_e:.8f}"
              f"  E_tau={Et:.6f} E_mu={Em:.6f} E_e={Ee:.8f}"
              f"  Q={Q:.10f}")
        break

# ---- TEST 3: Optimize ----
print("\n--- OPTIMIZE for Q=2/3 ---")
np.random.seed(42)
best_opt = None
for trial in range(200):
    x0 = np.sort(np.random.uniform(0.02, 0.98, 3))
    try:
        res = minimize(lambda p: (koide(E(p[0]), E(p[1]), E(p[2])) - 2/3)**2,
                       x0, method='Nelder-Mead',
                       options={'xatol':1e-13, 'fatol':1e-22, 'maxiter':5000})
        if best_opt is None or res.fun < best_opt.fun:
            best_opt = res
    except: pass
    if trial % 50 == 49:
        print(f"  trial {trial+1}/200, best so far: {best_opt.fun:.2e}")

p = sorted(best_opt.x)
Ep = [E(o) for o in p]
Ep_s = sorted(Ep)
Q_final = koide(*Ep_s)
print(f"\nFINAL Q = {Q_final:.12f}")
print(f"|Q-2/3| = {abs(Q_final - 2/3):.2e}")
print(f"Omega: {p[0]:.8f}, {p[1]:.8f}, {p[2]:.8f}")
print(f"E:     {Ep_s[0]:.10f}, {Ep_s[1]:.8f}, {Ep_s[2]:.8f}")
print(f"Ratios: 1 : {Ep_s[1]/Ep_s[0]:.2f} : {Ep_s[2]/Ep_s[0]:.2f}")
print(f"Target: 1 : {m_mu/m_e:.2f} : {m_tau/m_e:.2f}")

for name, Om, Ev in zip(['e','mu','tau'], p, sorted(Ep)):
    b2 = 1-Om**2
    A = 1.5*b2
    B = 0.5*np.sqrt(b2)
    print(f"  {name:3s}: Om={Om:.8f}, A={A:.6f}, B={B:.6f}, E={Ev:.8f}")
