"""
ISPG — shell-struqturis hipoteza
==================================
kitxva: jdebian tu ara SM fermionebi N-is nakrebebshi?
"""
import numpy as np

masses = {
    "e":   0.51099895,
    "u":   2.16,
    "d":   4.67,
    "s":   93.4,
    "mu":   105.6583755,
    "c":   1270.0,
    "tau":   1776.86,
    "b":   4180.0,
    "t":  172760.0,
}

charges = {
    "e":  (1.0,    -0.5, "lep",  1),
    "u":  (2.0/3,   0.5, "up",   1),
    "d":  (1.0/3,  -0.5, "dn",   1),
    "mu":  (1.0,    -0.5, "lep",  2),
    "c":  (2.0/3,   0.5, "up",   2),
    "s":  (1.0/3,  -0.5, "dn",   2),
    "tau":  (1.0,    -0.5, "lep",  3),
    "t":  (2.0/3,   0.5, "up",   3),
    "b":  (1.0/3,  -0.5, "dn",   3),
}

m_e = masses["e"]

print("=" * 78)
print("  k_f = sqrt(m_f/m_e)")
print("=" * 78)
print(f"\n  {'part':>4} {'m_f MeV':>11} {'m_f/m_e':>13} {'k_f':>10} "
      f"{'5*k':>9} {'family':>10} {'Q':>6}")
print("  " + "-" * 70)

rows = []
for name, m in masses.items():
    r = m / m_e
    k = np.sqrt(r)
    N_predicted = 5 * k
    N_assigned = round(N_predicted)
    k_frac = k - int(k)
    is_integer = abs(k - round(k)) < 0.05
    family = "A_5k" if is_integer else "B_mid"
    Q = charges[name][0]
    rows.append((name, N_assigned, k, k_frac, family, Q))
    print(f"  {name:>4} {m:>11.3f} {r:>13.3f} {k:>10.4f} "
          f"{N_predicted:>9.2f} {family:>10} {Q:>6.3f}")

print("\n" + "=" * 78)
print("  FAMILY GROUPING")
print("=" * 78)

family_A = [(n, N, Q) for n, N, k, kf, fam, Q in rows if fam.startswith("A")]
family_B = [(n, N, Q) for n, N, k, kf, fam, Q in rows if fam.startswith("B")]

print(f"\n  Family A (k_f integer, N = 5*k):")
for n, N, Q in family_A:
    print(f"     {n} (N={N:>4}, |Q|={Q:.3f})")

print(f"\n  Family B (k_f non-integer):")
for n, N, Q in family_B:
    print(f"     {n} (N={N:>4}, |Q|={Q:.3f})")

print("\n" + "=" * 78)
print("  Family B fractional parts")
print("=" * 78)
print(f"\n  {'part':>4} {'k_f':>10} {'frac':>12} {'5*frac':>10}")
print("  " + "-" * 50)
for n, N, k, kf, fam, Q in rows:
    if fam.startswith("B"):
        print(f"  {n:>4} {k:>10.4f} {kf:>12.4f} {5*kf:>10.3f}")

# shell structure
print("\n" + "=" * 78)
print("  SHELL STRUCTURE (clusters at similar N)")
print("=" * 78)
sorted_rows = sorted(rows, key=lambda x: x[1])
print(f"\n  {'part':>4} {'N':>5} {'log N':>8} {'delta':>10}")
prev_log = None
for n, N, k, kf, fam, Q in sorted_rows:
    log_N = np.log10(N)
    delta = log_N - prev_log if prev_log is not None else 0.0
    marker = "  <-- BIG JUMP" if delta > 0.3 else ""
    print(f"  {n:>4} {N:>5} {log_N:>8.3f} {delta:>10.3f}{marker}")
    prev_log = log_N

shells = []
current_shell = [sorted_rows[0]]
for i in range(1, len(sorted_rows)):
    delta = np.log10(sorted_rows[i][1]) - np.log10(sorted_rows[i-1][1])
    if delta > 0.3:
        shells.append(current_shell)
        current_shell = [sorted_rows[i]]
    else:
        current_shell.append(sorted_rows[i])
shells.append(current_shell)

print("\n  Shells (log-gap > 0.3):")
for i, shell in enumerate(shells):
    names = [s[0] for s in shell]
    Ns = [s[1] for s in shell]
    # determine what SM features this shell shares
    charges_in_shell = [charges[s[0]] for s in shell]
    gens = sorted(set(c[3] for c in charges_in_shell))
    types = sorted(set(c[2] for c in charges_in_shell))
    print(f"    Shell {i+1}:  {names}")
    print(f"              N = {Ns}")
    print(f"              gens = {gens},  types = {types}")
