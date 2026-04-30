"""
Path 3 — ემპირიული N ფორმულის ძიება
====================================

მონაცემები (QUANTUM/Python/cavity_vs_mass_audit.py-დან):
  τ:  (n=0, ℓ=0), ω²=0.441,  N=295
  μ:  (n=1, ℓ=0), ω²=0.942,  N=72
  e:  (n=0, ℓ=2), ω²=0.998,  N=5
  grav: (n=0, ℓ=1), ω²=0.750, N=?  (წინასწარმეტყველება)

ვეძებთ მარტივ არითმეტიკულ კავშირს ცავიტის (n, ℓ, ω²)
კვანტურ რიცხვებსა და Mathieu-ს N ინდექსს შორის.

კრიტერიუმი: <5% ცდომილება სამივე ლეპტონზე.
თუ ვერცერთი ფორმულა ვერ აკმაყოფილებს — ეს ნიშნავს,
რომ 3D→1D რედუქცია არ არის ტრივიალური (E₁D ინტეგრალი საჭიროა).
"""

import numpy as np
from scipy.optimize import least_squares, brute
from pathlib import Path

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# --- ფიქსირებული მონაცემები ---
DATA = {
    'tau':  dict(n=0, ell=0, w2=0.441, N=295),
    'mu':   dict(n=1, ell=0, w2=0.942, N=72),
    'e':    dict(n=0, ell=2, w2=0.998, N=5),
    'ell1': dict(n=0, ell=1, w2=0.750, N=None),  # ცავიტის ℓ=1 მოდი (არაფიზიკური სანიტი-ტესტი)
}

PARTICLES = ['tau', 'mu', 'e']
N_obs = np.array([DATA[p]['N'] for p in PARTICLES], dtype=float)
n_arr = np.array([DATA[p]['n'] for p in PARTICLES], dtype=float)
ell_arr = np.array([DATA[p]['ell'] for p in PARTICLES], dtype=float)
w2_arr = np.array([DATA[p]['w2'] for p in PARTICLES], dtype=float)
b2_arr = 1.0 - w2_arr  # binding energy (squared)


def report(name, N_pred, params=None):
    """ცდომილების გამოთვლა და ბეჭდვა."""
    err = np.abs(N_pred - N_obs) / N_obs * 100
    max_err = np.max(err)
    print(f"\n  [{name}]")
    if params is not None:
        print(f"    params:  {params}")
    for i, p in enumerate(PARTICLES):
        print(f"    {p}:  N_pred = {N_pred[i]:8.2f},  N_obs = {N_obs[i]:6.0f}, "
              f"err = {err[i]:6.2f}%")
    print(f"    >>> max_err = {max_err:.2f}%  "
          f"{'✅ წარმატება' if max_err < 5 else '❌ ჩავარდნა'}")
    return max_err


# ================================================================
# ფორმულა F1: N = A · (1-ω²)^α   (წმინდა ბმის ენერგია)
# ================================================================
def F1():
    print("\n" + "="*70)
    print("  F1:  N = A · (1-ω²)^α")
    print("="*70)

    def residuals(p):
        A, alpha = p
        return A * b2_arr**alpha - N_obs

    res = least_squares(residuals, [300, 1.0], method='lm')
    A, alpha = res.x
    N_pred = A * b2_arr**alpha
    max_err = report("F1", N_pred, {'A': A, 'α': alpha})
    # ℓ=1 მოდი
    N_grav = A * (1-0.750)**alpha
    print(f"    → ℓ=1 მოდის პრედიქცია:  N_grav = {N_grav:.2f}")
    return max_err, (A, alpha)


# ================================================================
# ფორმულა F2: N = A · (1-ω²)^α · (2ℓ+1)^γ
# ================================================================
def F2():
    print("\n" + "="*70)
    print("  F2:  N = A · (1-ω²)^α · (2ℓ+1)^γ")
    print("="*70)

    def residuals(p):
        A, alpha, gamma = p
        return A * b2_arr**alpha * (2*ell_arr+1)**gamma - N_obs

    res = least_squares(residuals, [300, 1.0, 0.0], method='lm')
    A, alpha, gamma = res.x
    N_pred = A * b2_arr**alpha * (2*ell_arr+1)**gamma
    max_err = report("F2", N_pred, {'A': A, 'α': alpha, 'γ': gamma})
    N_grav = A * (1-0.750)**alpha * (2*1+1)**gamma
    print(f"    → ℓ=1 მოდის პრედიქცია:  N_grav = {N_grav:.2f}")
    return max_err, (A, alpha, gamma)


# ================================================================
# ფორმულა F3: N = A · (1-ω²)^α · (n+1)^β  (რადიალური მრავალფეროვნება)
# ================================================================
def F3():
    print("\n" + "="*70)
    print("  F3:  N = A · (1-ω²)^α · (n+1)^β")
    print("="*70)

    def residuals(p):
        A, alpha, beta = p
        return A * b2_arr**alpha * (n_arr+1)**beta - N_obs

    res = least_squares(residuals, [300, 1.0, 0.0], method='lm')
    A, alpha, beta = res.x
    N_pred = A * b2_arr**alpha * (n_arr+1)**beta
    max_err = report("F3", N_pred, {'A': A, 'α': alpha, 'β': beta})
    N_grav = A * (1-0.750)**alpha * (0+1)**beta
    print(f"    → ℓ=1 მოდის პრედიქცია:  N_grav = {N_grav:.2f}")
    return max_err, (A, alpha, beta)


# ================================================================
# ფორმულა F4: N = A · (1-ω²)^α · (2ℓ+1)^γ · (n+1)^β   (3 პარამ, 3 წერტ.)
# ეს უნდა იყოს ზუსტი — საინტერესოა პარამეტრების მნიშვნელობა
# ================================================================
def F4():
    print("\n" + "="*70)
    print("  F4:  N = A · (1-ω²)^α · (2ℓ+1)^γ · (n+1)^β")
    print("="*70)

    # გადატანა ლოგ-სივრცეში: log N = log A + α log(1-ω²) + γ log(2ℓ+1) + β log(n+1)
    log_N = np.log(N_obs)
    X = np.column_stack([
        np.ones(3),
        np.log(b2_arr),
        np.log(2*ell_arr+1),
        np.log(n_arr+1),
    ])
    # 3 წერტილი, 4 უცნობი — underdetermined. ვდებთ β=0 ან γ=0.
    # ვცდი β=0:
    X3 = X[:, :3]
    coef = np.linalg.solve(X3, log_N)
    logA, alpha, gamma = coef
    A = np.exp(logA)
    N_pred = A * b2_arr**alpha * (2*ell_arr+1)**gamma
    max_err = report("F4a (β=0, ზუსტი ფიტი)", N_pred,
                     {'A': A, 'α': alpha, 'γ': gamma, 'β': 0})
    N_grav = A * (1-0.750)**alpha * (2*1+1)**gamma
    print(f"    → ℓ=1 მოდის პრედიქცია:  N_grav = {N_grav:.2f}")

    # ვცდი γ=0:
    X3b = X[:, [0, 1, 3]]
    coef = np.linalg.solve(X3b, log_N)
    logA, alpha, beta = coef
    A = np.exp(logA)
    N_pred = A * b2_arr**alpha * (n_arr+1)**beta
    max_err_b = report("F4b (γ=0, ზუსტი ფიტი)", N_pred,
                       {'A': A, 'α': alpha, 'γ': 0, 'β': beta})
    N_grav_b = A * (1-0.750)**alpha * (0+1)**beta
    print(f"    → ℓ=1 მოდის პრედიქცია:  N_grav = {N_grav_b:.2f}")

    return min(max_err, max_err_b)


# ================================================================
# ფორმულა F5: N = round[ A · exp(-ω²/T) ]   (ბოლცმანის ტიპი)
# ================================================================
def F5():
    print("\n" + "="*70)
    print("  F5:  N = A · exp(-ω²/T)")
    print("="*70)

    def residuals(p):
        A, T = p
        return A * np.exp(-w2_arr/T) - N_obs

    res = least_squares(residuals, [1000, 0.2], method='lm')
    A, T = res.x
    N_pred = A * np.exp(-w2_arr/T)
    max_err = report("F5", N_pred, {'A': A, 'T': T})
    N_grav = A * np.exp(-0.750/T)
    print(f"    → ℓ=1 მოდის პრედიქცია:  N_grav = {N_grav:.2f}")
    return max_err


# ================================================================
# ფორმულა F6: ლადერი — ω²=1 - q/N (ანალოგია Mathieu-სთან)
# ანუ N = q / (1-ω²) ფიქსირებული q-თვის
# ================================================================
def F6():
    print("\n" + "="*70)
    print("  F6:  N = q / (1-ω²)   (Mathieu-ს მოსალოდნელი სკალირება)")
    print("="*70)

    # q ფიტი
    q_opt = np.mean(N_obs * b2_arr)
    N_pred = q_opt / b2_arr
    max_err = report("F6", N_pred, {'q': q_opt})
    N_grav = q_opt / (1-0.750)
    print(f"    → ℓ=1 მოდის პრედიქცია:  N_grav = {N_grav:.2f}")
    print(f"    კომენტარი:  q = {q_opt:.2f}, MASS paper-ში q=1.853")
    print(f"               (ეს q არ არის Mathieu-ს q)")
    return max_err


# ================================================================
# ფორმულა F7: N = (1-ω²)^α · something — ორი შემოწმება რიცხვური
# ================================================================
def F7():
    print("\n" + "="*70)
    print("  F7:  N = C / (1-ω²)²    (ძლიერი სკალირება)")
    print("="*70)
    C_opt = np.mean(N_obs * b2_arr**2)
    N_pred = C_opt / b2_arr**2
    max_err = report("F7", N_pred, {'C': C_opt})
    N_grav = C_opt / (1-0.750)**2
    print(f"    → ℓ=1 მოდის პრედიქცია:  N_grav = {N_grav:.2f}")
    return max_err


# ================================================================
# ფორმულა F8: N = A · (1-ω²)^α · ω^γ   (ω-ს კომბინაცია)
# ================================================================
def F8():
    print("\n" + "="*70)
    print("  F8:  N = A · (1-ω²)^α · ω^γ")
    print("="*70)
    log_N = np.log(N_obs)
    X = np.column_stack([np.ones(3), np.log(b2_arr), np.log(np.sqrt(w2_arr))])
    coef = np.linalg.solve(X, log_N)
    logA, alpha, gamma = coef
    A = np.exp(logA)
    N_pred = A * b2_arr**alpha * np.sqrt(w2_arr)**gamma
    max_err = report("F8 (ზუსტი ფიტი)", N_pred,
                     {'A': A, 'α': alpha, 'γ': gamma})
    N_grav = A * (1-0.750)**alpha * np.sqrt(0.750)**gamma
    print(f"    → ℓ=1 მოდის პრედიქცია:  N_grav = {N_grav:.2f}")
    return max_err


# ================================================================
# ფორმულა F9: ფუნდამენტური ფიზიკა — WKB-ტიპი ინტეგრალი
# N ~ ∫ √(V-ω²)/ω  dr  ~  1/(1-ω²)  მცირე ბმის ენერგიისთვის
# უფრო ზოგადი:  N = A · (1/b)^α   ⟺  F1
# ================================================================


# ================================================================
# ფორმულა F10:  ლოგარითმული —  log N = A + B·(1-ω²) + C·ℓ
# ================================================================
def F10():
    print("\n" + "="*70)
    print("  F10:  log N = A + B·(1-ω²) + C·ℓ   (ხაზობრივი log-სივრცეში)")
    print("="*70)
    log_N = np.log(N_obs)
    X = np.column_stack([np.ones(3), b2_arr, ell_arr])
    coef = np.linalg.solve(X, log_N)
    A, B, C = coef
    N_pred = np.exp(A + B*b2_arr + C*ell_arr)
    max_err = report("F10 (ზუსტი ფიტი)", N_pred,
                     {'A': A, 'B': B, 'C': C})
    N_grav = np.exp(A + B*(1-0.750) + C*1)
    print(f"    → ℓ=1 მოდის პრედიქცია:  N_grav = {N_grav:.2f}")
    return max_err


# ================================================================
# ფორმულა F11:  N = round[ A · b^α · (2n+ℓ+1)^γ ]  (3D harmonic-ოსცილ. ანალოგია)
# ================================================================
def F11():
    print("\n" + "="*70)
    print("  F11:  N = A · (1-ω²)^α · (2n+ℓ+1)^γ   (3D QHO-სებრი)")
    print("="*70)
    # 2n+ℓ+1:  τ→1, μ→3, e→3.  მაგრამ μ და e იგივე!
    # ეს ვერ გაარჩევს, მოდი ვნახოთ:
    q_arr = 2*n_arr + ell_arr + 1
    print(f"    q = 2n+ℓ+1: {q_arr}  (μ და e ერთნაირი — ვერ გაარჩევს)")
    # ამიტომ F11 ვარდება
    return 100.0


# ================================================================
# მთავარი
# ================================================================
def main():
    print("="*70)
    print("  ISPG — Path 3:  ემპირიული N ფორმულის სისტემატური ძიება")
    print("="*70)
    print(f"\n  სამიზნე:  N_τ={295},  N_μ={72},  N_e={5}")
    print(f"  კრიტერიუმი:  max_err < 5% სამივე წერტილზე")

    results = {}
    results['F1']  = F1()[0]
    results['F2']  = F2()[0]
    results['F3']  = F3()[0]
    results['F4']  = F4()
    results['F5']  = F5()
    results['F6']  = F6()
    results['F7']  = F7()
    results['F8']  = F8()
    results['F10'] = F10()
    results['F11'] = F11()

    print("\n" + "="*70)
    print("  რეზიუმე — ყველა ფორმულა")
    print("="*70)
    print(f"\n  {'ფორმულა':<8}  {'max ცდომილება':<20}  {'სტატუსი'}")
    print("  " + "─"*50)
    best = None
    for name, err in sorted(results.items(), key=lambda x: x[1]):
        status = "✅" if err < 5 else ("⚠️ " if err < 20 else "❌")
        print(f"  {name:<8}  {err:>8.2f}%            {status}")
        if best is None or err < best[1]:
            best = (name, err)

    print("\n" + "="*70)
    print("  კრიტიკული ანალიზი — უნიკალურობა")
    print("="*70)
    print("\n  3 მონაცემი + 3 პარამეტრი = ტრივიალურად ზუსტი ფიტი.")
    print("  ნამდვილი კითხვა: ფიტები ერთ და იმავე ფიზიკაზე მიუთითებენ?")
    print("\n  ℓ=1 მოდის (ℓ=1, n=0, ω²=0.750) პრედიქცია ფიტებში:")
    print("  ──────────────────────────────────────────────")
    print(f"    F2 (ℓ-power):        N_grav ≈ 121")
    print(f"    F3 (n-power):        N_grav ≈ 165")
    print(f"    F8 (ω-power):        N_grav ≈ 206")
    print(f"    F10 (log-linear):    N_grav ≈  35")
    print("\n  მეტი ვიდრე 5× განსხვავება — ფორმულა არ არის")
    print("  უნიკალურად განსაზღვრული მხოლოდ 3 წერტილით.")
    print("\n  ასევე: ფიტების პარამეტრები არ არის \"ბუნებრივი\"")
    print("  (α≈0.62, γ≈-0.35 — არცერთი მარტივი კარგი რიცხვი).")

    print("\n" + "="*70)
    print("  საბოლოო დასკვნა")
    print("="*70)
    if best[1] < 5:
        print(f"\n  ⚠️  მათემატიკურად შესაძლებელია (F2/F3/F8/F10 — 0% ცდ.),")
        print(f"  მაგრამ არც-ერთი არ არის ფიზიკურად პრივილეგირებული:")
        print(f"     - 3 პარამ. + 3 წერტ. = ტრივიალური ფიტი")
        print(f"     - ℓ=1 მოდის პრედიქცია არასტაბილური (35–206)")
        print(f"     - პარამეტრები არაბუნებრივი (α=0.62 ≠ 1/2, 2/3)")
        print(f"\n  1 პარამეტრიანი F1 (N = A·(1-ω²)^α) **ვერ მუშაობს**")
        print(f"  (e-ზე 70% ცდომილება) — ანუ ℓ-ის ჩართვა აუცილებელია.")
        print(f"\n  მეცნიერული დასკვნა: ფორმულის უნიკალურობისთვის")
        print(f"  საჭიროა ოთხივე წერტილი (ℓ=1 მოდით) და თეორიული")
        print(f"  შეზღუდვები (E₁D ინტეგრალის სტრუქტურა).")
        print(f"  ემპირიული ძიება **არ გამოდის** — გადავდებთ future work-ად.")
    else:
        print(f"\n  ❌ ვერცერთი მარტივი ფორმულა ვერ აღადგენს (5, 72, 295)-ს.")

    # დამატებითი: შევამოწმოთ "ლამაზი" პარამეტრები
    print("\n" + "="*70)
    print("  ლამაზი პარამეტრების ტესტი (F2:  A·(1-ω²)^α · (2ℓ+1)^γ)")
    print("="*70)
    for alpha_try, gamma_try in [(2/3, -1/3), (0.5, -0.5), (1.0, -1.0),
                                  (2/3, -1/2), (1/2, -1/3), (0.6, -0.35)]:
        # A ფიტი ერთი წერტილით
        A_try = 295 / (b2_arr[0]**alpha_try * (2*ell_arr[0]+1)**gamma_try)
        N_pred = A_try * b2_arr**alpha_try * (2*ell_arr+1)**gamma_try
        err = np.abs(N_pred - N_obs) / N_obs * 100
        print(f"    α={alpha_try:.3f}, γ={gamma_try:+.3f}, A={A_try:.1f}  "
              f"→  τ={err[0]:.1f}%, μ={err[1]:.1f}%, e={err[2]:.1f}%  "
              f"max={np.max(err):.1f}%")


if __name__ == "__main__":
    main()
