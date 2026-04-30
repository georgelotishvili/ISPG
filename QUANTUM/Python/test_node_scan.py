"""Scan Omega for node count transitions — find ALL solution branches."""
import sys
sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

import numpy as np
from scipy.integrate import solve_ivp

def shoot(Phi0, Omega, r_max=30.0):
    """Integrate 3D oscillon ODE from r=0 and count nodes."""
    kappa2 = 1.0 - Omega**2
    if kappa2 <= 0:
        return np.inf, -1, None

    def rhs(r, y):
        Phi, dPhi = y
        if r < 1e-10:
            d2Phi = -(Omega**2 - 1) * Phi / 3.0 - Phi**2 / 3.0
        else:
            d2Phi = -(2.0 / r) * dPhi - (Omega**2 - 1) * Phi - Phi**2
        return [dPhi, d2Phi]

    sol = solve_ivp(rhs, [1e-10, r_max], [Phi0, 0.0],
                    rtol=1e-10, atol=1e-12,
                    max_step=0.1, dense_output=True)

    if not sol.success:
        return np.inf, -1, None

    Phi_vals = sol.y[0]
    nodes = np.sum(np.diff(np.sign(Phi_vals)) != 0)
    return sol.y[0, -1], nodes, sol


print("=" * 70)
print("  Node count scan: Omega vs number of nodes")
print("  (looking for excited state branches)")
print("=" * 70)

for Phi0 in [0.5, 1.0, 1.5, 2.0]:
    print(f"\n  Phi0 = {Phi0}")
    print(f"  {'Omega':>8} {'Phi_end':>12} {'nodes':>6}")
    print(f"  {'-'*30}")

    prev_nodes = -1
    transitions = []

    for Omega in np.linspace(0.999, 0.02, 500):
        Phi_end, nodes, sol = shoot(Phi0, Omega, r_max=30.0)

        if nodes != prev_nodes and prev_nodes >= 0 and nodes >= 0:
            transitions.append((Omega, prev_nodes, nodes))
            print(f"  {Omega:8.5f} {Phi_end:12.4e} {nodes:6d}  <- transition from {prev_nodes}")

        prev_nodes = nodes

    if transitions:
        print(f"\n  Found {len(transitions)} transitions:")
        for Om, n_from, n_to in transitions:
            print(f"    Omega={Om:.5f}: {n_from} -> {n_to} nodes")
    else:
        print(f"\n  No node transitions found")

    print(f"\n  Quick check of node count at key Omega values:")
    for Om in [0.99, 0.95, 0.90, 0.80, 0.70, 0.50, 0.30, 0.10]:
        Phi_end, nodes, _ = shoot(Phi0, Om, r_max=30.0)
        print(f"    Om={Om:.2f}: nodes={nodes}, Phi_end={Phi_end:.4e}")
