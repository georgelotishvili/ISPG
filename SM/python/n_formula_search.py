"""
ISPG — N-ის ფუნქციური ფორმის ძიება
======================================
კითხვა: არსებობს N = f(Q, T₃, Y, გენ., ფერი, სპინი)?

მეთოდი:
  1. ცხრილად SM კვანტური რიცხვები 12 ნაწილაკისთვის.
  2. ვცდი სხვადასხვა ანზაცს:
     A. N = a·Q² + b·T₃² + c·Y² + d·გენ² + e·სპინი + f·ფერი + const
     B. N = გენ² · (a + b·Q² + c·Y²)
     C. N = გენ · a^k (გეომეტრიული სიდიდე)
     D. log N = a·გენ + b·Q² + c·Y² + ...
     E. რიცხვთა თეორია — მთელი რიცხვების კომბინაციები
  3. ფიტის ხარისხი:  R² და max |N_pred − N_obs|/N_obs.

სამიზნე: R² > 0.99 → ფორმულა რეალურია.
         R² < 0.5  → ფორმულა არ არსებობს ამ დონეზე.
"""

import numpy as np
from numpy.linalg import lstsq
from itertools import combinations
from pathlib import Path

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
#  ნაწილაკების SM კვანტური რიცხვები
#     Q   — ელ. მუხტი
#     T3  — სუსტი ისოსპინი (ლეფტ-მხარი)
#     Y   — ჰიპერმუხტი (Y = 2(Q − T3))
#     gen — თაობა (1, 2, 3)
#     color — ფერის განზომილება (1 = ლეპტონი/ბოზ., 3 = ქვარკი)
#     spin — სპინი (0.5 ფერმ., 1 ვექტ. ბოზ., 0 Higgs)
#     N_obs — sm_particle_scan.py-ის შედეგი
# ----------------------------------------------------------------------
PARTICLES = [
    # name, Q, T3, Y, gen, color, spin, N_obs
    ("e",  -1,    -0.5,  -1.0,    1, 1, 0.5,    5),
    ("μ",  -1,    -0.5,  -1.0,    2, 1, 0.5,   72),
    ("τ",  -1,    -0.5,  -1.0,    3, 1, 0.5,  295),
    ("u",  +2/3, +0.5, +1/3,    1, 3, 0.5,   10),
    ("c",  +2/3, +0.5, +1/3,    2, 3, 0.5,  250),
    ("t",  +2/3, +0.5, +1/3,    3, 3, 0.5, 2910),
    ("d",  -1/3, -0.5, +1/3,    1, 3, 0.5,   15),
    ("s",  -1/3, -0.5, +1/3,    2, 3, 0.5,   68),
    ("b",  -1/3, -0.5, +1/3,    3, 3, 0.5,  453),
    ("W",  +1,    +1,    0,       0, 1, 1.0, 1986),
    ("Z",   0,     0,    0,       0, 1, 1.0, 2115),
    ("H",   0,     0,    1,       0, 1, 0,   2479),
]


def fit_poly(features, N_obs, feat_names):
    """ფიტი:  N = Σ c_i · feature_i   (ლინეარული ანზაცი)"""
    A = np.array(features).T  # M × K
    b = np.array(N_obs, dtype=float)
    coefs, residual, rank, sv = lstsq(A, b, rcond=None)
    N_pred = A @ coefs
    ss_res = np.sum((N_obs - N_pred) ** 2)
    ss_tot = np.sum((N_obs - np.mean(N_obs)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rel_err = np.abs((N_pred - N_obs) / N_obs)
    return coefs, N_pred, r2, rel_err


def print_fit(label, N_obs, N_pred, r2, rel_err, feat_names=None, coefs=None):
    print(f"\n  ── {label} ──")
    if coefs is not None and feat_names is not None:
        formula = " + ".join(f"{c:+.3f}·{n}" for c, n in zip(coefs, feat_names))
        print(f"    N = {formula}")
    print(f"    R² = {r2:.4f},  max |ცდ.| = {np.max(rel_err)*100:.1f}%,  "
          f"საშ. = {np.mean(rel_err)*100:.1f}%")
    worst = int(np.argmax(rel_err))
    best = int(np.argmin(rel_err))
    print(f"    საუკეთესო: N_pred({best}) = {N_pred[best]:.1f} (obs={N_obs[best]})")
    print(f"    უარესი:    N_pred({worst}) = {N_pred[worst]:.1f} (obs={N_obs[worst]})")


def main():
    print("=" * 76)
    print("  N = f(კვანტური რიცხვები)  — ფორმულის ძიება")
    print("=" * 76)

    names = [p[0] for p in PARTICLES]
    Q  = np.array([p[1] for p in PARTICLES])
    T3 = np.array([p[2] for p in PARTICLES])
    Y  = np.array([p[3] for p in PARTICLES])
    gen = np.array([p[4] for p in PARTICLES], dtype=float)
    col = np.array([p[5] for p in PARTICLES], dtype=float)
    sp  = np.array([p[6] for p in PARTICLES])
    N_obs = np.array([p[7] for p in PARTICLES], dtype=float)

    print(f"\n  ცხრილი ({len(PARTICLES)} ნაწილაკი):")
    print(f"  {'name':<4} {'Q':>6} {'T3':>6} {'Y':>6} {'gen':>4} "
          f"{'col':>4} {'spin':>5} {'N_obs':>6}")
    for p in PARTICLES:
        print(f"  {p[0]:<4} {p[1]:>6.3f} {p[2]:>6.2f} {p[3]:>6.3f} "
              f"{p[4]:>4d} {p[5]:>4d} {p[6]:>5.1f} {p[7]:>6d}")

    # ---- ანზაცი A: სრული ლინეარული ---
    ones = np.ones_like(N_obs)
    feats_A = [ones, Q**2, T3**2, Y**2, gen**2, sp, col]
    names_A = ["1", "Q²", "T₃²", "Y²", "gen²", "spin", "color"]
    coefs, N_pred, r2, err = fit_poly(feats_A, N_obs, names_A)
    print_fit("ანზაცი A: N = c₀ + c₁Q² + c₂T₃² + c₃Y² + c₄gen² + c₅spin + c₆color",
              N_obs, N_pred, r2, err, names_A, coefs)

    # ---- ანზაცი B: gen²-ის მასშტაბი ---
    feats_B = [gen**2, gen**2 * Q**2, gen**2 * Y**2, gen**2 * col, ones]
    names_B = ["gen²", "gen²·Q²", "gen²·Y²", "gen²·col", "1"]
    coefs, N_pred, r2, err = fit_poly(feats_B, N_obs, names_B)
    print_fit("ანზაცი B: N ~ gen²·(Q², Y², color)",
              N_obs, N_pred, r2, err, names_B, coefs)

    # ---- ანზაცი C: log N ფიტი ---
    valid = N_obs > 0
    logN = np.log(N_obs[valid])
    feats_C = [ones[valid], gen[valid], Q[valid]**2, Y[valid]**2, col[valid], sp[valid]]
    names_C = ["1", "gen", "Q²", "Y²", "color", "spin"]
    A = np.array(feats_C).T
    coefs, *_ = lstsq(A, logN, rcond=None)
    logN_pred = A @ coefs
    N_pred_C = np.exp(logN_pred)
    ss_res = np.sum((logN - logN_pred)**2)
    ss_tot = np.sum((logN - np.mean(logN))**2)
    r2_C = 1 - ss_res / ss_tot
    err_C = np.abs((N_pred_C - N_obs[valid]) / N_obs[valid])
    print_fit("ანზაცი C: log N = c₀ + c₁·gen + c₂·Q² + c₃·Y² + c₄·col + c₅·sp",
              N_obs[valid], N_pred_C, r2_C, err_C, names_C, coefs)

    # ---- ანზაცი D: √N ფიტი (რადგან m ~ N², √N ~ √m) ---
    sqrtN = np.sqrt(N_obs)
    feats_D = [ones, gen, gen**2, Q**2, Y**2, col]
    names_D = ["1", "gen", "gen²", "Q²", "Y²", "col"]
    A = np.array(feats_D).T
    coefs, *_ = lstsq(A, sqrtN, rcond=None)
    sqrtN_pred = A @ coefs
    N_pred_D = sqrtN_pred ** 2
    ss_res = np.sum((sqrtN - sqrtN_pred)**2)
    ss_tot = np.sum((sqrtN - np.mean(sqrtN))**2)
    r2_D = 1 - ss_res / ss_tot
    err_D = np.abs((N_pred_D - N_obs) / N_obs)
    print_fit("ანზაცი D: √N = c₀ + c₁gen + c₂gen² + c₃Q² + c₄Y² + c₅col",
              N_obs, N_pred_D, r2_D, err_D, names_D, coefs)

    # ---- მხოლოდ ფერმიონები (N=12 → 9) ---
    print("\n" + "=" * 76)
    print("  მხოლოდ ფერმიონები (9 ნაწილაკი, ბოზონების გამორიცხვა)")
    print("=" * 76)
    fmask = sp == 0.5
    N_f = N_obs[fmask]
    feats_F = [ones[fmask], gen[fmask], gen[fmask]**2,
               Q[fmask]**2, Y[fmask]**2, col[fmask]]
    names_F = ["1", "gen", "gen²", "Q²", "Y²", "col"]
    coefs, N_pred, r2, err = fit_poly(feats_F, N_f, names_F)
    print_fit("ფერმიონი: N = c₀ + c₁gen + c₂gen² + c₃Q² + c₄Y² + c₅col",
              N_f, N_pred, r2, err, names_F, coefs)

    sqrtN_f = np.sqrt(N_f)
    A = np.array(feats_F).T
    c2, *_ = lstsq(A, sqrtN_f, rcond=None)
    sqrtN_pred = A @ c2
    N_pred2 = sqrtN_pred**2
    ss_res = np.sum((sqrtN_f - sqrtN_pred)**2)
    ss_tot = np.sum((sqrtN_f - np.mean(sqrtN_f))**2)
    r2_2 = 1 - ss_res / ss_tot
    err_2 = np.abs((N_pred2 - N_f) / N_f)
    print_fit("ფერმიონი √N = c₀ + c₁gen + c₂gen² + c₃Q² + c₄Y² + c₅col",
              N_f, N_pred2, r2_2, err_2, names_F, c2)

    # ---- გენერაცია-1 ცალკე (4 ნაწილაკი: e, u, d) ---
    print("\n" + "=" * 76)
    print("  მხოლოდ გენერაცია-1 ფერმიონები (e, u, d)")
    print("=" * 76)
    g1mask = (sp == 0.5) & (gen == 1)
    N_g1 = N_obs[g1mask]
    names_g1 = [names[i] for i in range(len(names)) if g1mask[i]]
    print(f"\n  {dict(zip(names_g1, N_g1.astype(int)))}")
    print(f"  N-ის ფარდობა e-სთან:  "
          f"{[int(n)//int(N_g1[0]) for n in N_g1]}")
    print(f"  თუ N = k · (... ფერი, მუხტი), ვნახოთ კავშირი Q-სთან:")
    for nm, n, q_ in zip(names_g1, N_g1, Q[g1mask]):
        print(f"    {nm}: N={int(n)}, |3Q|={abs(3*q_):.1f}, "
              f"5·(1+ფერი·(1−|Q|))={5*(1 + 3*(1-abs(q_)))}")

    # ---- ფორმულის გაფართოება ---
    print("\n" + "=" * 76)
    print("  რიცხვთა თეორიის ცდა: მარტივი კომბინაციები")
    print("=" * 76)
    print("\n  ცნობილი დაკვირვებები:")
    print(f"    e(N=5):   5 = 5·1")
    print(f"    u(N=10):  10 = 5·2")
    print(f"    d(N=15):  15 = 5·3")
    print(f"    → გენ-1: N = 5·k,  k = (1, 2, 3)?  k ∝ ???")
    print(f"    μ(N=72):  72 = 8·9 = 2³·3²")
    print(f"    τ(N=295): 295 = 5·59  (5-ი ისევ)")
    print(f"    s(N=68):  68 = 4·17")
    print(f"    c(N=250): 250 = 2·5³")
    print(f"    b(N=453): 453 = 3·151")
    print(f"    t(N=2910):2910 = 2·3·5·97")

    # ---- ფოკუსი: გენ-1 ფორმულა ---
    print("\n" + "=" * 76)
    print("  ფოკუსი:  N_gen1 = 5 + 15·(1 − |Q|)")
    print("=" * 76)
    for nm, q_ in zip(names_g1, Q[g1mask]):
        pred = 5 + 15 * (1 - abs(q_))
        obs = dict(zip(names_g1, N_g1))[nm]
        print(f"    {nm}:  Q={q_:+.3f},  |Q|={abs(q_):.3f},  "
              f"N_pred = 5+15·(1−|Q|) = {pred:.1f},  N_obs = {int(obs)}  "
              f"{'✓ ზუსტი' if abs(pred-obs)<0.01 else '✗'}")

    # ---- გენ-2 და გენ-3 შემოწმება იგივე ფორმულით ---
    print("\n  იგივე ფორმულა გენ-2,3-ზე:")
    for p in PARTICLES:
        name, q_, _, _, g, _, sp_, N_ = p
        if sp_ == 0.5 and g > 1:
            pred = 5 + 15 * (1 - abs(q_))
            print(f"    {name} (gen={g}):  N_pred={pred:.1f},  N_obs={N_},  "
                  f"ცდ. = {abs(pred-N_)/N_*100:.1f}%  ✗")

    # ---- Koide N-სივრცეში ---
    print("\n" + "=" * 76)
    print("  Koide ურთიერთობა N-სივრცეში")
    print("=" * 76)
    Ne, Nmu, Ntau = 5, 72, 295
    lhs = Ne**2 + Nmu**2 + Ntau**2
    rhs = (Ne + Nmu + Ntau)**2
    Q_koide = lhs / rhs
    print(f"    N_e² + N_μ² + N_τ² = {lhs}")
    print(f"    (N_e + N_μ + N_τ)² = {rhs}")
    print(f"    Q = ΣN²/(ΣN)² = {Q_koide:.6f}   (2/3 = {2/3:.6f})")
    print(f"    ცდ. 2/3-თან: {abs(Q_koide - 2/3)*100/(2/3):.3f} %")

    print("\n" + "=" * 76)
    print("  მთავარი დასკვნა")
    print("=" * 76)
    print("""
    ნაპოვნია:
      1. გენ-1 ფერმიონი:  N = 5 + 15·(1 − |Q|)  —  ზუსტი (e, u, d)
      2. Koide 2/3 არის N-სივრცის ოქროს კუთხის პირობა: ΣN² = (2/3)(ΣN)²
      3. გენერაცია-გენერაცია კავშირი  ფორმულა არ გვაქვს

    სუსტი ფიტი (R² < 0.99) მთლიან სივრცეში ნიშნავს რომ:
      • გენ-1-ის სტრუქტურა (Q → N) რეალურია
      • მასების **ჰიერარქია** თაობებს შორის ISPG-ს ჯერ არ აქვს ახსნილი
    """)


if __name__ == "__main__":
    main()
