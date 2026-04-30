"""
ISPG — 5, 72, 295 რიცხვების კანონზომიერების სკანი
===================================================
არაფერს ცვლის .tex ფაილებში — მხოლოდ დიაგნოსტიკა.

კითხვა (IQ-ტესტის სტილში):
  ლეპტონის Mathieu ლადერის N = (5, 72, 295).
  რას წარმოადგენს ისინი?
    (a) Lie ჯგუფის ფესვების / განზომილებების რიცხვს?
    (b) მარტივ არითმეტიკულ კანონზომიერებას?
    (c) მხოლოდ ფიტის შედეგს (სტრუქტურის გარეშე)?

ამ სკრიპტი სამ ტიპის ფილტრს ატარებს:
  1. Lie ჯგუფების ფესვები / განზომილებები / რეპრ. ზომები
  2. მარტივი არითმეტიკული ფორმულები (ორი პარამეტრით)
  3. კომბინატორული / გრაფული რიცხვები
"""

import sys
from itertools import product

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")

TARGET = [5, 72, 295]

# =====================================================================
#  1. Lie ჯგუფების რიცხვობრივი მახასიათებლები
# =====================================================================

def lie_catalog():
    """Lie ალგებრების ფესვების რიცხვი (#R) და განზომილება (dim G)
    კლასიკური და განსაკუთრებული ოჯახებისთვის."""
    out = []
    # A_n = SU(n+1): dim = n(n+2), #roots = n(n+1)
    for n in range(1, 30):
        out.append((f"A_{n} = SU({n+1})",      n*(n+1),      n*(n+2)))
    # B_n = SO(2n+1): dim = n(2n+1), #roots = 2n²
    for n in range(2, 20):
        out.append((f"B_{n} = SO({2*n+1})",    2*n*n,        n*(2*n+1)))
    # C_n = Sp(2n): dim = n(2n+1), #roots = 2n²
    for n in range(2, 20):
        out.append((f"C_{n} = Sp({2*n})",      2*n*n,        n*(2*n+1)))
    # D_n = SO(2n): dim = n(2n-1), #roots = 2n(n-1)
    for n in range(3, 20):
        out.append((f"D_{n} = SO({2*n})",      2*n*(n-1),    n*(2*n-1)))
    # განსაკუთრებულები
    out.append(("G_2",  12,  14))
    out.append(("F_4",  48,  52))
    out.append(("E_6",  72,  78))
    out.append(("E_7", 126, 133))
    out.append(("E_8", 240, 248))
    return out


def lie_scan():
    print("=" * 72)
    print("  [ნაწ. 1]  Lie ალგებრების სტრუქტურული რიცხვები")
    print("=" * 72)
    cat = lie_catalog()
    for t in TARGET:
        hits = []
        for name, n_roots, dim in cat:
            if n_roots == t:
                hits.append((name, "#roots", n_roots))
            if dim == t:
                hits.append((name, "dim",    dim))
        print(f"\n  N = {t}:")
        if hits:
            for name, kind, val in hits:
                print(f"     ✓ {name}  {kind} = {val}")
        else:
            print("     (ზუსტი დამთხვევა არ არის)")

    # ახლოს დგომა 295-სთან
    print("\n  295-ის გარშემო (±5):")
    for name, n_roots, dim in cat:
        if abs(n_roots - 295) <= 5:
            print(f"     {name}: #roots = {n_roots}")
        if abs(dim - 295) <= 5:
            print(f"     {name}: dim    = {dim}")


# =====================================================================
#  2. მარტივი არითმეტიკული კანონზომიერებები
# =====================================================================

def arithmetic_scan():
    print("\n" + "=" * 72)
    print("  [ნაწ. 2]  მარტივი არითმეტიკული კანონზომიერება")
    print("=" * 72)

    e, mu, tau = TARGET

    # წყვილის თანაფარდობები
    print(f"\n  თანაფარდობები:")
    print(f"     μ/e   = {mu/e:.4f}")
    print(f"     τ/e   = {tau/e:.4f}")
    print(f"     τ/μ   = {tau/mu:.4f}")

    # კვადრატები (MASS paper-ში m ∝ N²)
    print(f"\n  N² მნიშვნელობები (მასის ანალოგი):")
    print(f"     5²   = {e*e}")
    print(f"     72²  = {mu*mu}")
    print(f"     295² = {tau*tau}")
    print(f"     72²/5²   = {mu*mu/(e*e):.4f}    (obs m_μ/m_e = 206.77)")
    print(f"     295²/5²  = {tau*tau/(e*e):.4f}   (obs m_τ/m_e = 3477.5)")

    # N² შორის სხვაობა
    d1 = mu*mu - e*e
    d2 = tau*tau - mu*mu
    print(f"\n  N² სხვაობები:")
    print(f"     72² − 5²    = {d1}")
    print(f"     295² − 72²  = {d2}")
    print(f"     ფარდობა: (295²−72²)/(72²−5²) = {d2/d1:.4f}")

    # ფაქტორიზაცია
    def factor(n):
        f = []
        d = 2
        x = n
        while d*d <= x:
            while x % d == 0:
                f.append(d); x //= d
            d += 1
        if x > 1:
            f.append(x)
        return f

    print(f"\n  ფაქტორიზაცია:")
    print(f"     5   = {factor(5)}")
    print(f"     72  = {factor(72)}")
    print(f"     295 = {factor(295)}")

    # არითმეტიკული / გეომეტრიული პროგრესიის ტესტი
    print(f"\n  პროგრესიის ტესტები:")
    print(f"     არით. პროგრ.: 72−5 = {mu-e}, 295−72 = {tau-mu}  (არა)")
    print(f"     გეომ. პროგრ.: 72/5 = {mu/e:.3f}, 295/72 = {tau/mu:.3f}  (არა)")

    # Fibonacci-ს მსგავსი: N_3 = a·N_2 + b·N_1?
    print(f"\n  წრფივი რეკურენტი N_3 = a·N_2 + b·N_1:")
    for a in range(-5, 10):
        for b in range(-20, 20):
            if a*mu + b*e == tau:
                print(f"     ✓ 295 = {a}·72 + {b}·5")

    # N = α·n² + β·n + γ  (n = 1, 2, 3)?
    print(f"\n  კვადრატული ფიტი N(n) = α·n² + β·n + γ, n ∈ {{1,2,3}}:")
    # 3 განტ. 3 უცნობით:  α + β + γ = 5
    #                    4α + 2β + γ = 72
    #                    9α + 3β + γ = 295
    # ამოხსნა
    # (4α+2β+γ) − (α+β+γ) = 3α + β = 67
    # (9α+3β+γ) − (4α+2β+γ) = 5α + β = 223
    # 2α = 156 → α = 78, β = 67 - 3·78 = 67 - 234 = -167, γ = 5 - 78 + 167 = 94
    alpha, beta, gamma = 78, -167, 94
    for n, t in zip([1,2,3], TARGET):
        v = alpha*n*n + beta*n + gamma
        print(f"     n={n}:  78·{n}² − 167·{n} + 94 = {v}   (target {t})")
    # კომენტარი: კოეფიციენტები "ნებისმიერია" — 3 განტოლებაზე 3 უცნობი ყოველთვის ამოიხსნება


# =====================================================================
#  3. კომბინატორული და გრაფული რიცხვები
# =====================================================================

def combinatorial_scan():
    print("\n" + "=" * 72)
    print("  [ნაწ. 3]  კომბინატორული რიცხვები")
    print("=" * 72)

    # სამკუთხაობა T_n = n(n+1)/2
    T = {n: n*(n+1)//2 for n in range(1, 30)}
    # კვადრატი
    S = {n: n*n for n in range(1, 30)}
    # ხუთკუთხა P_n = n(3n-1)/2
    P = {n: n*(3*n-1)//2 for n in range(1, 20)}
    # ტეტრაედრული Te_n = n(n+1)(n+2)/6
    Te = {n: n*(n+1)*(n+2)//6 for n in range(1, 15)}

    for t in TARGET:
        found = []
        for n, v in T.items():
            if v == t: found.append(f"T_{n} (სამკუთხა)")
        for n, v in S.items():
            if v == t: found.append(f"S_{n} (კვადრატი)")
        for n, v in P.items():
            if v == t: found.append(f"P_{n} (ხუთკუთხა)")
        for n, v in Te.items():
            if v == t: found.append(f"Te_{n} (ტეტრაედრული)")
        print(f"\n  N = {t}:  {', '.join(found) if found else '(არ არის ფიგურალური რიცხვი)'}")

    # Catalan: 1,1,2,5,14,42,132,429,...
    cat = [1,1,2,5,14,42,132,429,1430,4862]
    print(f"\n  Catalan C_n = 1,1,2,5,14,42,132,...:")
    for t in TARGET:
        if t in cat:
            print(f"     ✓ {t} = C_{cat.index(t)}")
        else:
            print(f"     {t}: არ არის Catalan")

    # სფერული ფუნქციების ჯამური რიცხვი რადიუსზე:
    # L_max-მდე: Σ(2ℓ+1) = (L+1)²
    # n_max-მდე: Σn² არის ტეტრაედრული
    print(f"\n  Spherical harmonics ჯამი: Σ_{{ℓ=0}}^{{L}}(2ℓ+1) = (L+1)²")
    for L in range(0, 20):
        v = (L+1)**2
        for t in TARGET:
            if v == t:
                print(f"     ✓ N = {t} = (L+1)² ამისთვის L = {L}")


# =====================================================================
#  4. სპეციალური კომბინაციები ფიზიკურიდან
# =====================================================================

def physics_scan():
    print("\n" + "=" * 72)
    print("  [ნაწ. 4]  ფიზიკური მიმართულება")
    print("=" * 72)

    # SU(5) GUT: 5+10+1 = 16 (ერთი ოჯახი), 5+10 = 15 (Higgs), ...
    # SO(10): 16 (spinor), 45 (adjoint), 10, 54, 126
    # E_6: 27 (fund), 78 (adj), 351, 650, 2925
    # ყველა SM ნაწილაკები ერთ ოჯახში: 15 (fermion DoF SU(5)-ში)

    print("\n  SU(5) რეპრეზენტაციები:  5, 10, 15, 24, 35, 40, 45, 50, 70, 75, 105, 126, ...")
    print("  SO(10):                  10, 16, 45, 54, 120, 126, 144, 210, 320, ...")
    print("  E_6:                     27, 78, 351, 650, 2925, ...")
    print("  E_7:                     56, 133, 912, 1539, 8645, ...")
    print("  E_8:                     248, 3875, 30380, 147250, ...")

    # ახლოს 295-ს — არცერთი არა.
    print(f"\n  ზუსტი დამთხვევა 295-სთან:  არც ერთი (ძალიან კონკრეტული)")
    print(f"  ახლოს:  E_6 350-ის წარმომადგ. (351)  —  სხვაობა 56")


def main():
    print("=" * 72)
    print("  ISPG — 5, 72, 295 კანონზომიერების სკანი")
    print("=" * 72)
    lie_scan()
    arithmetic_scan()
    combinatorial_scan()
    physics_scan()

    print("\n" + "=" * 72)
    print("  ბოლო დასკვნა: კოპიურდება ტერმინალის outputიდან")
    print("=" * 72)


if __name__ == "__main__":
    main()
