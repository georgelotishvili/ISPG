"""
ISPG — პულსონ-სინქრონიზაციის ჰიპოთეზის ტესტი
==================================================
კითხვა:  შეიძლება თუ არა კიბის ნომრები N = (5, 72, 295) გამოვიდეს
         ცავიტის სიხშირის ω და პულსონის სიხშირის Ω_bg კომბინაციიდან?

ინტუიცია (მომხმარებლის):
    პულსონი ვაკუუმის "სუნთქვაა" — მეტრონომი, რომელიც ყველა ვიბრაციას
    ტიკტაკს უწესებს. N = რამდენი "შენი" ვიბრაცია შედის ერთ პულსონის ციკლში.
    ე.ი. N უნდა დამოკიდებული იყოს ω-ს და Ω_bg-ს თანაფარდობაზე.

რატომ ეს ახალი ტესტია:
    `n_formula_empirical_search.py` ტესტავდა მხოლოდ ω-ის ფუნქციებს.
    აქ Ω_bg-ს ცხადი სახით ვრთავთ.

დიაგნოსტიკური — .tex ფაილებს არ ცვლის.

ტესტდება 5 ფიზიკურად მოტივირებული ფორმულა:
    A. პირდაპირი სინქრო:      N ∝ ω/Ω_bg
    B. კვადრატული სინქრო:     N ∝ (ω/Ω_bg)²
    C. ენერგეტიკული უფსკრული: N ∝ √[(1−Ω²)/(1−ω²)]
    D. mass-gap (1.5-pow):    N ∝ [(1−Ω²)/(1−ω²)]^0.75
    E. სრული (n, ℓ, ω, Ω):    N ∝ (2n+ℓ+1) · ω/Ω_bg · (1−ω²)^(−1/2)

და დამატებით — Mathieu-ს ინვერტირება (F): რა ფორმით შეეძლო Mathieu-ს
ფუნქციას გამოყვანილი ყოფილიყო ცავიტის სიხშირიდან.
"""

import sys
import numpy as np
from scipy.integrate import solve_bvp
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.interpolate import interp1d
from scipy.special import mathieu_b

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")

ALPHA = 0.5

# სამიზნე კიბის ნომრები (MASS §3)
N_TARGET = {"tau": 295, "mu": 72, "e": 5}
# ცავიტის მოდის იდენტიფიკაცია (QUANTUM §7, Version A)
NL_TARGET = {"tau": (0, 0), "mu": (1, 0), "e": (0, 2)}


# ------------------------------------------------------------------
#  ოსცილონის აგება (test_e1d_derivation.py-იდან)
# ------------------------------------------------------------------

def nl_func(Phi):
    return Phi * (1.0 - np.exp(-ALPHA * Phi))


def dnl_func(Phi):
    return (1.0 - np.exp(-ALPHA * Phi)) + ALPHA * Phi * np.exp(-ALPHA * Phi)


def solve_osc(Phi0, r_max=60.0, r_prev=None, y_prev=None, p_prev=None):
    Om_guess = np.sqrt(max(0.01, 1.0 - min(Phi0 / 4.2, 0.95)))

    def ode(r, y, p):
        Om = p[0]
        Phi, dPhi = y
        r_safe = np.maximum(r, 1e-8)
        NL = nl_func(Phi)
        d2 = -(2.0 / r_safe) * dPhi - (Om**2 - 1) * Phi - NL
        d2_0 = -(Om**2 - 1) * Phi / 3.0 - NL / 3.0
        d2 = np.where(r < 1e-8, d2_0, d2)
        return np.vstack([dPhi, d2])

    def bc(ya, yb, p):
        return np.array([ya[0] - Phi0, ya[1], yb[0]])

    if r_prev is not None:
        N = max(500, len(r_prev))
        r = np.linspace(1e-6, r_max, N)
        f0 = interp1d(r_prev, y_prev[0], fill_value=0, bounds_error=False)
        f1 = interp1d(r_prev, y_prev[1], fill_value=0, bounds_error=False)
        sc = Phi0 / max(abs(y_prev[0][0]), 1e-30)
        y_init = np.vstack([f0(r) * sc, f1(r) * sc])
        Om_guess = p_prev[0]
    else:
        r = np.linspace(1e-6, r_max, 500)
        kg = np.sqrt(max(0.01, 1.0 - Om_guess**2))
        Phi_init = Phi0 / np.cosh(r * kg)**2
        y_init = np.vstack([Phi_init, np.gradient(Phi_init, r)])

    sol = solve_bvp(ode, bc, r, y_init, p=[Om_guess], tol=1e-6,
                    max_nodes=40000, verbose=0)
    if sol.success and 0.01 < sol.p[0] < 0.999:
        return sol.p[0], sol
    return None, None


def cavity_eigs(r_bg, Phi_bg, l_val, N=2500):
    r_max = r_bg[-1] * 0.9
    dr = r_max / (N + 1)
    r = np.linspace(dr, r_max - dr, N)
    f = interp1d(r_bg, Phi_bg, fill_value=0, bounds_error=False)
    Phi = f(r)
    c_lin = dnl_func(Phi)
    V = 1.0 - c_lin + l_val * (l_val + 1) / r**2
    H = diags(
        [-np.ones(N - 1) / dr**2, 2.0 / dr**2 + V, -np.ones(N - 1) / dr**2],
        [-1, 0, 1], format="csc",
    )
    evals, _ = eigsh(H, k=min(15, N - 2), which="SM")
    return np.sqrt(np.maximum(np.sort(evals[evals < 1.0]), 0))


# ------------------------------------------------------------------
#  1. ოსცილონის აგება, Ω_bg-ის და ω-ების მიღება
# ------------------------------------------------------------------

print("=" * 78)
print("  პულსონ-სინქრონიზაციის ჰიპოთეზის ტესტი")
print("=" * 78)

print("\n  ოსცილონის აგება (Phi_0 = 0.05 → 2.35, უწყვეტი კანი)...")
prev = None
for Phi0 in np.arange(0.05, 2.40, 0.05):
    if prev:
        Om, sol = solve_osc(Phi0, r_prev=prev.x, y_prev=prev.y, p_prev=prev.p)
    else:
        Om, sol = solve_osc(Phi0)
    if Om:
        prev = sol

Om_bg, sol_bg = solve_osc(2.35, r_prev=prev.x, y_prev=prev.y, p_prev=prev.p)
print(f"  ✓ პულსონის სიხშირე  Ω_bg = {Om_bg:.6f}")

om_vals = {}
for name, (n, l) in NL_TARGET.items():
    eigs = cavity_eigs(sol_bg.x, sol_bg.y[0], l, N=2500)
    if n >= len(eigs):
        print(f"  [!] ნაკლები მოდი ℓ={l}-ზე")
        continue
    om_vals[name] = eigs[n]

print(f"\n  ცავიტის სიხშირეები ({len(om_vals)} ლეპტონი):")
for name in ["tau", "mu", "e"]:
    if name in om_vals:
        n, l = NL_TARGET[name]
        print(f"    ω_{name:3} (n={n},ℓ={l}) = {om_vals[name]:.6f}    "
              f"→ სამიზნე N = {N_TARGET[name]}")


# ------------------------------------------------------------------
#  2. კანდიდატი ფორმულების ტესტი
# ------------------------------------------------------------------

def test_formula(label, desc, fn, scale_to="e"):
    """ფორმულას ვუყენებთ ყველა ლეპტონზე, ნორმირება 'e' ლეპტონზე
    (რომ N_e = 5 ზუსტი იყოს) და ვზომავთ ცდომილებას სხვებზე."""
    raw = {p: fn(om_vals[p], Om_bg, *NL_TARGET[p]) for p in om_vals}
    # ნორმირება: იმ ფაქტორით რომ N_e = 5 ზუსტი გახდეს
    scale = N_TARGET[scale_to] / raw[scale_to]
    pred = {p: raw[p] * scale for p in raw}
    print(f"\n  ── ფორმულა {label}: {desc}")
    print(f"     {'':>5} {'ω':>8} {'N_raw':>10} {'N_pred':>10} "
          f"{'N_target':>10} {'ცდომ.':>10}")
    total_err = 0.0
    for p in ["tau", "mu", "e"]:
        if p not in raw:
            continue
        err = 100.0 * (pred[p] - N_TARGET[p]) / N_TARGET[p]
        print(f"     {p:>5} {om_vals[p]:8.4f} {raw[p]:10.4f} {pred[p]:10.4f} "
              f"{N_TARGET[p]:10d} {err:+9.2f}%")
        total_err += err**2
    rmse = np.sqrt(total_err / 3)
    print(f"     RMS ცდომილება: {rmse:.2f}%")
    return rmse


print("\n" + "=" * 78)
print("  კანდიდატი ფორმულების სკანი")
print("  (ყველა ნორმირებულია e-ზე: N_e = 5 ზუსტი)")
print("=" * 78)

results = {}

# A
results["A"] = test_formula(
    "A", "N ∝ ω/Ω_bg  (უმარტივესი სინქრო)",
    lambda o, O, n, l: o / O,
)

# B
results["B"] = test_formula(
    "B", "N ∝ (ω/Ω_bg)²  (კვადრატული სინქრო)",
    lambda o, O, n, l: (o / O) ** 2,
)

# C
results["C"] = test_formula(
    "C", "N ∝ √[(1−Ω²)/(1−ω²)]  (ენერგიის უფსკრული)",
    lambda o, O, n, l: np.sqrt((1 - O**2) / (1 - o**2)),
)

# D (1.5-pow გამომდინარე m ∝ (1−ω²)^1.5 იდეიდან)
results["D"] = test_formula(
    "D", "N ∝ [(1−Ω²)/(1−ω²)]^0.75  (mass-gap 1.5-ხარ.)",
    lambda o, O, n, l: ((1 - O**2) / (1 - o**2)) ** 0.75,
)

# E (ყველა ცვლადი ერთად)
results["E"] = test_formula(
    "E", "N ∝ (2n+ℓ+1) · (ω/Ω_bg) · (1−ω²)^(−1/2)",
    lambda o, O, n, l: (2 * n + l + 1) * (o / O) / np.sqrt(1 - o**2),
)

# F — ახალი: N ∝ (Ω_bg)/(Ω_bg − ω)  (ძგერების სიხშირე)
results["F"] = test_formula(
    "F", "N ∝ Ω_bg / |Ω_bg − ω|  (ძგერების სიხშირე)",
    lambda o, O, n, l: O / max(abs(O - o), 1e-6),
)

# G — ჰიბრიდი: კუბური სინქრო
results["G"] = test_formula(
    "G", "N ∝ (ω/Ω_bg)³",
    lambda o, O, n, l: (o / O) ** 3,
)


# ------------------------------------------------------------------
#  3. Mathieu-ს პირდაპირი ტესტი
# ------------------------------------------------------------------

print("\n" + "=" * 78)
print("  Mathieu-ს შემოწმება: b_N(q) რომელი q-სთვის იძლევა ფარდობებს?")
print("=" * 78)

print(f"\n  დაკვირვებული (PDG): m_μ/m_e = 206.768,  m_τ/m_e = 3477.47")
print(f"\n  {'q':>8} {'b_5':>10} {'b_72':>10} {'b_295':>12} "
      f"{'m_μ/m_e':>10} {'m_τ/m_e':>10}")
print("  " + "-" * 70)
for q_test in [0.5, 1.0, 1.5, 1.853, 2.0, 3.0, Om_bg, Om_bg**2]:
    b5 = mathieu_b(5, q_test)
    b72 = mathieu_b(72, q_test)
    b295 = mathieu_b(295, q_test)
    print(f"  {q_test:8.4f} {b5:10.3f} {b72:10.3f} {b295:12.3f} "
          f"{b72/b5:10.3f} {b295/b5:10.3f}")

print("\n  შენიშვნა: MASS paper ფიტავს q = 1.853 (რიცხვით ზუსტია)")
print(f"  ჩვენი Ω_bg = {Om_bg:.4f}.  დამოკიდებულება q → Ω_bg ცხადი არ ჩანს.")


# ------------------------------------------------------------------
#  4. დასკვნა — საუკეთესო კანდიდატი
# ------------------------------------------------------------------

print("\n" + "=" * 78)
print("  რეზიუმე")
print("=" * 78)
best_label = min(results, key=lambda k: results[k])
print(f"\n  საუკეთესო ფორმულა: {best_label}   (RMS ცდომილება = {results[best_label]:.2f}%)")
print(f"  ყველა ფორმულის RMS ცდომილებები:")
for label in sorted(results, key=lambda k: results[k]):
    marker = "  ← საუკეთესო" if label == best_label else ""
    print(f"    {label}: {results[label]:10.2f}%{marker}")

print("\n  ინტერპრეტაცია:")
print("    RMS < 5%   →  ფორმულა მართალი შეიძლება იყოს")
print("    RMS < 20%  →  მნიშვნელოვანი ფიზიკის ხელწერა, ზუსტი ფორმა საძიებო")
print("    RMS > 50%  →  ფორმულა არ მუშაობს; სხვა სტრუქტურა საჭიროა")
