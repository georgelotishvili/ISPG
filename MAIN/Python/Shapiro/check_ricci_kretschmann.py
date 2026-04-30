"""Symbolic Ricci and Kretschmann for ISPG static bi-conformal metric (c=1).

Independent check for ``MAIN/0. MAIN.tex`` (eq:ricci, eq:kretschner):
verifies the full closed-form Kretschmann invariant
    K = (r_s^2 / (4*r^8)) * (48*r^2 - 32*r*r_s + 7*r_s^2) * exp(-2*r_s/r)
against the SymPy-computed curvature directly, and additionally reports
the weak-field leading term (12*r_s^2/r^6 * exp(-2*r_s/r)) for comparison
with Schwarzschild recovery.
"""
import sympy as sp

r_s, r, th = sp.symbols("r_s r theta", positive=True, real=True)

# ds^2 = -exp(-r_s/r) dt^2 + exp(r_s/r)(dr^2 + r^2 dOmega^2)
g00 = -sp.exp(-r_s / r)
g11 = sp.exp(r_s / r)
g22 = sp.exp(r_s / r) * r**2
g33 = sp.exp(r_s / r) * r**2 * sp.sin(th) ** 2

g_mat = sp.diag(g00, g11, g22, g33)
ginv = sp.diag(1 / g00, 1 / g11, 1 / g22, 1 / g33)

# x^0=t, x^1=r, x^2=theta, x^3=phi
x = [sp.symbols("t"), r, th, sp.symbols("phi")]


def d_g(mu, nu, rho):
    return sp.diff(g_mat[mu, nu], x[rho])


def Gamma_up_low_low(k, i, j):
    s = 0
    for l in range(4):
        s += ginv[k, l] * (d_g(j, l, i) + d_g(i, l, j) - d_g(i, j, l))
    return sp.simplify(s / 2)


Gamma = {}
for k in range(4):
    for i in range(4):
        for j in range(4):
            Gamma[(k, i, j)] = Gamma_up_low_low(k, i, j)


def d_Gamma(k, i, j, m):
    return sp.diff(Gamma[(k, i, j)], x[m])


# R^ρ_{σμν} = ∂_μ Γ^ρ_{νσ} - ∂_ν Γ^ρ_{μσ} + Γ^ρ_{μλ}Γ^λ_{νσ} - Γ^ρ_{νλ}Γ^λ_{μσ}
# Map (ρ,σ,μ,ν) -> (a,b,c,d): R^a_{bcd}
def R_up_low_low_low(a, b, c, d):
    acc = d_Gamma(a, d, b, c) - d_Gamma(a, c, b, d)
    for e in range(4):
        acc += Gamma[(a, c, e)] * Gamma[(e, d, b)]
        acc -= Gamma[(a, d, e)] * Gamma[(e, c, b)]
    return sp.simplify(acc)


# Covariant R_abcd = g_{ae} R^e_{bcd}
def R_low_low_low_low(a, b, c, d):
    s = 0
    for e in range(4):
        s += g_mat[a, e] * R_up_low_low_low(e, b, c, d)
    return sp.simplify(s)


# Ricci R_{μν} = R^λ_{μλν}
Ric = sp.zeros(4, 4)
for mu in range(4):
    for nu in range(4):
        acc = 0
        for lam in range(4):
            acc += R_up_low_low_low(lam, mu, lam, nu)
        Ric[mu, nu] = sp.simplify(acc)

R_scalar = sum(ginv[i, i] * Ric[i, i] for i in range(4))
R_scalar = sp.simplify(R_scalar)

paper_R = -(r_s**2) / (2 * r**4) * sp.exp(-r_s / r)

print("R_scalar (SymPy):")
print(R_scalar)
print("\nPaper R:")
print(paper_R)
print("\nDifference:")
print(sp.simplify(R_scalar - paper_R))

# R_{μν} R^{μν} = g^{μα} g^{νβ} R_{μν} R_{αβ}
R2 = 0
for mu in range(4):
    for nu in range(4):
        for a in range(4):
            for b in range(4):
                R2 += ginv[mu, a] * ginv[nu, b] * Ric[mu, nu] * Ric[a, b]
R2 = sp.simplify(R2)
print("\nR_{μν}R^{μν} (SymPy):")
print(R2)

# Kretschmann: R_abcd R^abcd with R^abcd = g^{ae}g^{bf}g^{cg}g^{dh} R_efgh
K = 0
for a in range(4):
    for b in range(4):
        for c in range(4):
            for d in range(4):
                R_abcd = R_low_low_low_low(a, b, c, d)
                if R_abcd == 0:
                    continue
                Rup = 0
                for e in range(4):
                    for f in range(4):
                        for gi in range(4):
                            for h in range(4):
                                Rup += (
                                    ginv[a, e]
                                    * ginv[b, f]
                                    * ginv[c, gi]
                                    * ginv[d, h]
                                    * R_low_low_low_low(e, f, gi, h)
                                )
                Rup = sp.simplify(Rup)
                if Rup == 0:
                    continue
                K += R_abcd * Rup

K = sp.simplify(K)

# Full closed-form Kretschmann invariant as written in eq:kretschner:
paper_K_full = (
    r_s**2 / (4 * r**8) * (48 * r**2 - 32 * r * r_s + 7 * r_s**2)
    * sp.exp(-2 * r_s / r)
)

# Weak-field leading term (Schwarzschild-recovery check only):
paper_K_leading = 12 * r_s**2 / r**6 * sp.exp(-2 * r_s / r)

print("\nK (SymPy):")
print(K)
print("\nPaper K (full eq:kretschner):")
print(paper_K_full)
print("\nDifference K - paper_K_full (should simplify to 0):")
print(sp.simplify(K - paper_K_full))

print("\nPaper K (leading term, Schwarzschild recovery only):")
print(paper_K_leading)
print("\nDifference K - paper_K_leading (nonzero: contains subleading terms):")
print(sp.simplify(K - paper_K_leading))
