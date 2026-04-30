"""Component-by-component check of the ISPG bi-conformal Einstein system
for both sign choices of the scalar kinetic term.

Tests the claim in ``MAIN/0. MAIN.tex`` Sec. 2.2 (around eq:action):
substituting the bi-conformal ansatz g_{\\mu\\nu}[\\varphi] into
G_{\\mu\\nu} = kappa * T^{(phi)}_{\\mu\\nu} (matter vacuum) selects
kappa = -1/2 uniquely: the three independent spherical-symmetric
components reduce to the SAME single ODE on phi(r) only for
kappa = -1/2, while kappa = +1/2 leaves an inconsistent (overdetermined)
system.

Static, spherically symmetric line element (c = 1):
    ds^2 = -exp(phi(r)) dt^2 + exp(-phi(r))(dr^2 + r^2 dOmega^2)

Scalar energy-momentum tensor (eq:Tphi in main text):
    T^(phi)_{mu nu} = d_mu phi d_nu phi - (1/2) g_{mu nu} (d phi)^2

Run:  python check_kappa_consistency.py
Exit: 0 on consistency for kappa=-1/2 AND inconsistency for kappa=+1/2.
"""
import sys
import sympy as sp

# coordinates: t, r, theta, phi
t, r, th, ph = sp.symbols("t r theta phi", real=True)
x = [t, r, th, ph]

# scalar field phi(r)
phi_r = sp.Function("phi")(r)

# bi-conformal metric (c=1)
g00 = -sp.exp(phi_r)
g11 = sp.exp(-phi_r)
g22 = sp.exp(-phi_r) * r**2
g33 = sp.exp(-phi_r) * r**2 * sp.sin(th) ** 2
g_mat = sp.diag(g00, g11, g22, g33)
ginv = sp.diag(1 / g00, 1 / g11, 1 / g22, 1 / g33)


def dg(a, b, c):
    return sp.diff(g_mat[a, b], x[c])


Gamma = {}
for k in range(4):
    for i in range(4):
        for j in range(4):
            s = 0
            for l in range(4):
                s += ginv[k, l] * (dg(j, l, i) + dg(i, l, j) - dg(i, j, l))
            Gamma[(k, i, j)] = sp.simplify(s / 2)


def dGamma(k, i, j, m):
    return sp.diff(Gamma[(k, i, j)], x[m])


def R_ulll(a, b, c, d):
    acc = dGamma(a, d, b, c) - dGamma(a, c, b, d)
    for e in range(4):
        acc += Gamma[(a, c, e)] * Gamma[(e, d, b)]
        acc -= Gamma[(a, d, e)] * Gamma[(e, c, b)]
    return sp.simplify(acc)


# Ricci tensor
Ric = sp.zeros(4, 4)
for mu in range(4):
    for nu in range(4):
        acc = 0
        for lam in range(4):
            acc += R_ulll(lam, mu, lam, nu)
        Ric[mu, nu] = sp.simplify(acc)

R_scalar = sp.simplify(sum(ginv[i, i] * Ric[i, i] for i in range(4)))

# Einstein tensor
G = sp.zeros(4, 4)
for mu in range(4):
    for nu in range(4):
        G[mu, nu] = sp.simplify(Ric[mu, nu] - sp.Rational(1, 2) * g_mat[mu, nu] * R_scalar)

# Scalar T^(phi)_{mu nu} = d_mu phi d_nu phi - (1/2) g_{mu nu} (d phi)^2
# With phi = phi(r) only, the only nonzero derivative is d_r phi.
dphi = [0, sp.diff(phi_r, r), 0, 0]
# (d phi)^2 = g^{mu nu} d_mu phi d_nu phi
dphi_sq = 0
for a in range(4):
    for b in range(4):
        dphi_sq += ginv[a, b] * dphi[a] * dphi[b]
dphi_sq = sp.simplify(dphi_sq)

Tphi = sp.zeros(4, 4)
for mu in range(4):
    for nu in range(4):
        Tphi[mu, nu] = sp.simplify(
            dphi[mu] * dphi[nu] - sp.Rational(1, 2) * g_mat[mu, nu] * dphi_sq
        )

# For a diagonal static spherical ansatz, the independent components are
# {tt, rr, theta theta} (phi phi is identical to theta theta up to sin^2).
# The Einstein system is G_{mu nu} = kappa * Tphi_{mu nu}.
# For consistency: (G - kappa Tphi)_{tt} = 0, _{rr} = 0, _{thth} = 0 must
# reduce to the SAME single ODE on phi(r).


def reduce_to_phi_eom(kappa, label):
    """Return the residual (G - kappa T^(phi)) for tt, rr, thth components,
    each written as an explicit expression in phi, phi', phi'', r.
    """
    pp = sp.Function("phi")(r)
    ppp = sp.diff(pp, r)
    pppp = sp.diff(pp, r, 2)
    subs_map = {sp.diff(pp, r, 2): pppp, sp.diff(pp, r): ppp}
    residuals = {}
    for name, (a, b) in [("tt", (0, 0)), ("rr", (1, 1)), ("thth", (2, 2))]:
        expr = sp.simplify(G[a, b] - kappa * Tphi[a, b])
        # divide out obvious common factor exp(+/- phi) * r^? * sin^?
        expr = sp.expand(sp.simplify(sp.powdenest(expr, force=True)))
        residuals[name] = sp.simplify(expr)
    return residuals


def test_kappa(kappa_val, label):
    print(f"\n=== Testing kappa = {kappa_val}  ({label}) ===")
    res = reduce_to_phi_eom(kappa_val, label)
    # Extract the ODE from each component by dividing out the common
    # metric factor so that each component becomes a scalar polynomial
    # in phi', phi'' with explicit 1/r, 1/r^2 coefficients.
    # Normalize each component by stripping exp() factors.
    normalized = {}
    pp = phi_r
    ppp = sp.diff(pp, r)
    pppp = sp.diff(pp, r, 2)
    for name, expr in res.items():
        e = sp.simplify(expr)
        # pull out exponentials of phi by factoring
        # write as polynomial in ppp, pppp with r-dependent coefficients
        e = sp.expand(e)
        # divide by exp(-phi) (common overall for this diagonal system up to sign conventions)
        factor = sp.exp(-pp)
        candidate = sp.simplify(e / factor)
        # if the result is rational in r and polynomial in phi', phi'',
        # we accept this normalization. Otherwise try exp(phi).
        if sp.simplify(candidate * factor - e) == 0:
            normalized[name] = sp.expand(candidate)
        else:
            normalized[name] = sp.expand(e)
    # Compute pairwise differences: if all three give the SAME ODE,
    # all differences simplify to zero (or to a common multiple factorable out).
    names = list(normalized.keys())
    diffs = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = sp.simplify(normalized[names[i]] - normalized[names[j]])
            diffs[(names[i], names[j])] = d
    consistent = all(d == 0 for d in diffs.values())
    print("  Residual (tt):", sp.simplify(normalized["tt"]))
    print("  Residual (rr):", sp.simplify(normalized["rr"]))
    print("  Residual (thth):", sp.simplify(normalized["thth"]))
    print()
    for pair, d in diffs.items():
        print(f"  diff[{pair[0]} - {pair[1]}] = {sp.simplify(d)}")
    # For a bi-conformal reduction, the three residuals are proportional
    # but not identical (each carries a different r-power from the metric
    # factor). The meaningful test is: do they share a common single ODE?
    # Equivalently: is each residual a polynomial multiple of a single
    # universal ODE expression in (phi', phi'').
    # We test by dividing (tt) by a shared structural form and checking
    # whether (rr), (thth) match the same form.
    return normalized, diffs, consistent


def extract_ode(normalized):
    """Attempt to extract a single ODE by finding a ratio between the three
    components that is a function of r only. If such a ratio exists for
    both (rr/tt) and (thth/tt), the system is reducible; otherwise it is
    overdetermined."""
    pp = phi_r
    ppp = sp.diff(pp, r)
    pppp = sp.diff(pp, r, 2)
    tt = normalized["tt"]
    rr = normalized["rr"]
    thth = normalized["thth"]
    # Try ratio rr/tt
    try:
        ratio_rt = sp.simplify(rr / tt)
    except Exception:
        ratio_rt = None
    try:
        ratio_tht = sp.simplify(thth / tt)
    except Exception:
        ratio_tht = None
    # Check that the ratios depend only on r (not on phi, phi', phi'')
    def depends_only_on_r(expr):
        if expr is None:
            return False
        s = sp.sympify(expr)
        free = s.free_symbols | {a for a in s.atoms(sp.Derivative)} | {a for a in s.atoms(sp.Function)}
        allowed = {r}
        bad = {a for a in free if a not in allowed and not (a.is_Symbol and a == r)}
        # Exclude Function(phi) and its derivatives from "r-only"
        return not any(
            (a == phi_r) or isinstance(a, sp.Derivative) for a in s.atoms(sp.Function, sp.Derivative)
        )
    ok_rt = depends_only_on_r(ratio_rt)
    ok_tht = depends_only_on_r(ratio_tht)
    return ratio_rt, ratio_tht, ok_rt, ok_tht


if __name__ == "__main__":
    print("ISPG bi-conformal Einstein-scalar consistency check")
    print("=" * 60)
    print(
        "Metric: ds^2 = -exp(phi(r)) dt^2 + exp(-phi(r))(dr^2 + r^2 dOmega^2)"
    )
    print("System: G_{mu nu} = kappa * T^(phi)_{mu nu}")
    print(
        "        T^(phi)_{mu nu} = d_mu phi d_nu phi - (1/2) g_{mu nu} (d phi)^2"
    )
    # Print Einstein tensor diagonals (already printed above is excluded;
    # we re-emit compactly).
    print("\nNonzero G components (diagonal):")
    print("  G_tt  =", sp.simplify(G[0, 0]))
    print("  G_rr  =", sp.simplify(G[1, 1]))
    print("  G_thth=", sp.simplify(G[2, 2]))
    print("\nNonzero T^(phi) components (diagonal):")
    print("  Tphi_tt  =", sp.simplify(Tphi[0, 0]))
    print("  Tphi_rr  =", sp.simplify(Tphi[1, 1]))
    print("  Tphi_thth=", sp.simplify(Tphi[2, 2]))

    # kappa = -1/2 should reduce to ONE ODE (consistent).
    norm_neg, diffs_neg, _ = test_kappa(sp.Rational(-1, 2), "ISPG phantom sign")
    rt_neg, tht_neg, ok_rt_neg, ok_tht_neg = extract_ode(norm_neg)
    print(f"\n  ratio rr/tt    = {sp.simplify(rt_neg)}    (r-only: {ok_rt_neg})")
    print(f"  ratio thth/tt = {sp.simplify(tht_neg)}    (r-only: {ok_tht_neg})")

    # kappa = +1/2 should leave an overdetermined system.
    norm_pos, diffs_pos, _ = test_kappa(sp.Rational(1, 2), "canonical sign")
    rt_pos, tht_pos, ok_rt_pos, ok_tht_pos = extract_ode(norm_pos)
    print(f"\n  ratio rr/tt    = {sp.simplify(rt_pos)}    (r-only: {ok_rt_pos})")
    print(f"  ratio thth/tt = {sp.simplify(tht_pos)}    (r-only: {ok_tht_pos})")

    # Reduction criterion: kappa=-1/2 should have BOTH ratios
    # r-only and finite; kappa=+1/2 should fail (ratios depend on phi
    # derivatives -> overdetermined).
    passed_neg = ok_rt_neg and ok_tht_neg
    passed_pos = not (ok_rt_pos and ok_tht_pos)
    print("\n" + "=" * 60)
    print(f"kappa=-1/2 consistent (reduces to ONE ODE):  {passed_neg}")
    print(f"kappa=+1/2 overdetermined (NOT a single ODE): {passed_pos}")
    if passed_neg and passed_pos:
        print("\nVERDICT: main-text claim verified. Exit 0.")
        sys.exit(0)
    else:
        print("\nVERDICT: claim NOT verified by this check. Exit 1.")
        sys.exit(1)
