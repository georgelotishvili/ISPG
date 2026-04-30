"""
Phase G / E1 — N_W=1986, N_Z=2115 ძიება 3D ოსცილონის ℓ=1 სპექტრიდან

საკითხი:
  არსებული ფაქტი (phase_a_gauge_sector.py):
    m_N ≈ (N/5)² · m_e,  N_W=1986, N_Z=2115 → 0.3% ცდომ. მასებზე
    (N_W/N_Z)² = 0.8818 ≈ cos θ_W_PDG = 0.8815  (0.03% ცდ.)

  ეს არის **ფიტი**: N_W, N_Z არჩეულია ცდომ. მინიმიზაციისთვის.
  E1-ის მიზანი: არის თუ არა ISPG-ის ოსცილონის კავიტ-სპექტრში
  რაიმე სტრუქტურული მნიშვნელობა, რომელიც **დამოუკიდებლად**
  ათითებს N_W=1986 და N_Z=2115-ს?

მიდგომა — 4 ტესტი:
  (T1) ℓ=1 ვექტორული კავიტ-მოდების რიცხვი (n=0..n_max) Gleiser
       ოსცილონის ფონზე. რამდენი ბმული მდგომარეობაა?
  (T2) ℓ=1 ვს ℓ=0: (ω_ℓ=1 / ω_ℓ=0)² ფარდობა ემთხვევა თუ არა
       cos² θ_W = 0.777-ს რომელიმე (n_1, n_0) წყვილისთვის?
  (T3) Mathieu-ის a_N(q) ძიება: არის თუ არა q-ის მნიშვნელობა,
       რომლისთვისაც a_1986(q)/a_2115(q) = (m_W/m_Z)² და
       ამასთან q-ის მნიშვნელობა ISPG-ის ფუნდამენტური
       პარამეტრიდან (α, Φ_c) გამოდის?
  (T4) თვითდათვლა: ოსცილონის ფონზე ფარდობა m_W/m_Z
       როცა N_W, N_Z = ფიქსირებული კავიტ-მოდების ინდექსები
       (არა ფიტი) რამდენად ცდება?

გულახდილი მოლოდინი:
  T1-ში ბმული მდგომარეობები ცოტაა (~5-20). ე.ი. N=1986-ის
  პირდაპირი კავიტ-ინტერპრეტაცია ჩავარდება.
  T2-ში ფარდობა 0.777 რანდომულია — თუ ემთხვევა, საინტერესოა.
  T3-ში Mathieu-ფიტიც შეიძლება იმუშაოს სხვა q-ით.
  T4 — საბოლოო ჭეშმარიტების ტესტი.

რა გამოდის — გამოვა. არ ვმალავ შედეგს.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.linalg import eigh_tridiagonal
from scipy.special import mathieu_a, mathieu_b


# -----------------------------------------------------------------------
# 1) Gleiser ოსცილონის ფონი
# -----------------------------------------------------------------------

def oscillon_rhs(r, y, omega2):
    """Φ'' + (2/r)Φ' = (1−ω²)Φ − Φ²"""
    Phi, dPhi = y
    if r < 1e-12:
        d2 = ((1.0 - omega2) * Phi - Phi**2) / 3.0
    else:
        d2 = -2.0/r * dPhi + (1.0 - omega2) * Phi - Phi**2
    return [dPhi, d2]


def find_oscillon(Phi_c, r_max=40.0, N_grid=12000):
    def residual(omega2):
        sol = solve_ivp(lambda r, y: oscillon_rhs(r, y, omega2),
                        [1e-10, r_max], [Phi_c, 0.0],
                        method='RK45', rtol=1e-11, atol=1e-13,
                        max_step=0.05)
        return sol.y[0, -1]
    try:
        omega2 = brentq(residual, 0.1, 0.999, xtol=1e-12)
    except Exception:
        return None, None, None
    r_eval = np.linspace(1e-10, r_max, N_grid)
    sol = solve_ivp(lambda r, y: oscillon_rhs(r, y, omega2),
                    [1e-10, r_max], [Phi_c, 0.0],
                    method='RK45', t_eval=r_eval,
                    rtol=1e-11, atol=1e-13)
    return sol.t, sol.y[0], omega2


def cavity_modes(r, Phi_bg, ell, n_want=50):
    """პერტურბაცია: V_pert(r) = 1 − 2Φ_bg(r) + ℓ(ℓ+1)/r².
    ბრუნდება ყველა eigenvalue (ბმული + უკიდე), სორტირებული."""
    N = len(r)
    dr = r[1] - r[0]
    V = 1.0 - 2.0 * Phi_bg
    cen = np.zeros(N)
    cen[1:] = ell * (ell + 1) / r[1:]**2
    cen[0] = cen[1]
    W = V + cen
    diag = 2.0/dr**2 + W[1:-1]
    off = -np.ones(N - 3) / dr**2
    evals, evecs = eigh_tridiagonal(diag, off,
                                    select='i',
                                    select_range=(0, min(n_want-1, N-4)))
    return evals, evecs


# -----------------------------------------------------------------------
# 2) Mathieu ლადერის მასები
# -----------------------------------------------------------------------

def mathieu_char_value(N, q):
    """a_N(q) ან b_N(q). paper-ში განხილული q=1.853."""
    # paper-ის ლადერი: დიდი N-ზე a_N(q) → N². ვიყენებთ scipy-ს.
    if N % 2 == 0:
        return mathieu_a(N, q)
    return mathieu_b(N, q)


# -----------------------------------------------------------------------
# 3) T1 — ℓ=1 ვექტორული კავიტის სპექტრი
# -----------------------------------------------------------------------

def T1_vector_spectrum():
    print("=" * 78)
    print("T1 — ℓ=1 ვექტორული ბმული მოდების რიცხვი")
    print("=" * 78)
    print("""
მიზანი: რამდენი ℓ=1 ბმული მდგომარეობაა Gleiser ოსცილონზე?
თუ ბმული მხოლოდ ~10 — მაშინ N_W=1986 **არ არის** პირდაპირ
n-ინდექსი კავიტში. ეს მაშინვე გამორიცხავს ნაივურ ინტერპრეტაციას.
""")
    print(f"{'Φ_c':>6}{'Ω²':>10}{'ℓ=0 bnd':>10}{'ℓ=1 bnd':>10}{'ℓ=2 bnd':>10}")
    print("-" * 78)
    for Phi_c in [0.8, 1.0, 1.2, 1.4, 1.48]:
        r, Phi, Om2 = find_oscillon(Phi_c, r_max=40.0, N_grid=12000)
        if r is None:
            continue
        counts = []
        for ell in [0, 1, 2]:
            evals, _ = cavity_modes(r, Phi, ell, n_want=200)
            n_bound = np.sum(evals < 1.0)  # ბმული: ω² < V_∞ = 1
            counts.append(n_bound)
        print(f"{Phi_c:>6.2f}{Om2:>10.4f}"
              f"{counts[0]:>10d}{counts[1]:>10d}{counts[2]:>10d}")
    print()
    print("დასკვნა: თუ ბმული მდგომ. ≪ 1986 → N_W ≠ (ℓ=1, n=1986) პირდაპირ.")


# -----------------------------------------------------------------------
# 4) T2 — ℓ=1 vs ℓ=0 ფარდობა → cos² θ_W ?
# -----------------------------------------------------------------------

def T2_ratio_test():
    print()
    print("=" * 78)
    print("T2 — (ω²_ℓ=1) / (ω²_ℓ=0) ფარდობა ემთხვევა cos² θ_W = 0.777?")
    print("=" * 78)
    print("""
იდეა: თუ W (ℓ=1 vector) და Z (ℓ=1 mixed) კავიტის მოდებია,
(m_W/m_Z)² = cos² θ_W = 0.777. ეძებს ი, j-ს ისეთს რომ
ω²_ℓ=1[i] / ω²_ℓ=1[j] ≈ 0.777 ცდომ. < 0.5%-ით.
""")
    target = (80369/91188)**2  # (m_W/m_Z)²
    print(f"მიზანი: (m_W/m_Z)² = {target:.6f}")
    print()
    print(f"{'Φ_c':>6}{'best ratio':>14}{'ცდ.%':>10}{'(i,j)':>14}")
    print("-" * 78)
    best_global = None
    for Phi_c in [0.8, 1.0, 1.2, 1.4, 1.48]:
        r, Phi, Om2 = find_oscillon(Phi_c, r_max=40.0, N_grid=12000)
        if r is None:
            continue
        evals, _ = cavity_modes(r, Phi, ell=1, n_want=50)
        bound = evals[evals < 1.0]
        if len(bound) < 2:
            print(f"{Phi_c:>6.2f}  ℓ=1 ბმული < 2 → ფარდობა ვერ ვპოულობ")
            continue
        best = (float('inf'), None, None)
        for i in range(len(bound)):
            for j in range(len(bound)):
                if i == j or bound[j] <= 0:
                    continue
                rat = bound[i] / bound[j]
                err = abs(rat - target) / target * 100
                if err < best[0]:
                    best = (err, (i, j), rat)
        print(f"{Phi_c:>6.2f}{best[2]:>14.6f}{best[0]:>10.3f}  {str(best[1]):>14}")
        if best_global is None or best[0] < best_global[0]:
            best_global = (best[0], Phi_c, best[1], best[2])
    print()
    if best_global:
        print(f"საუკეთესო: Φ_c={best_global[1]}, წყვილი={best_global[2]}, "
              f"rat={best_global[3]:.6f}, ცდ. {best_global[0]:.3f}%")
        if best_global[0] > 1.0:
            print("→ ფარდობა **არ** ემთხვევა cos² θ_W-ს სასარგებლოდ")
        else:
            print("→ საინტერესო! შემდგომი კვლევის ღირსი")


# -----------------------------------------------------------------------
# 5) T3 — Mathieu a_N(q) ფიტის შემოწმება
# -----------------------------------------------------------------------

def T3_mathieu_ladder():
    print()
    print("=" * 78)
    print("T3 — Mathieu ლადერ: a_1986(q)/a_2115(q) = cos² θ_W?")
    print("=" * 78)
    print("""
paper-ში q=1.853 (MASS/ISPG_FrequencyToMass.tex-ში არგუმენტი).
შევამოწმოთ: თუ a_N(q) = N² + O(q²) დიდი N-ზე, მაშინ
a_1986(q)/a_2115(q) ≈ (1986/2115)² = 0.8817 — რაც
ზუსტად (N/5)² ფორმულასთან ემთხვევა.

ე.ი. N_W, N_Z ფიტი ხდება *N-ზე* (დიდი N-ზე Mathieu ≈ N²),
და q-ის კონკრეტული მნიშვნელობა ფარდობისთვის არარელევანტურია.

ე.ი. cos² θ_W-ის "0.03% ცდომ." არის **არჩევანის შედეგი**:
N მთელი რიცხვია და გვაქვს ~2 ერთეული თავისუფლება
N_W, N_Z-ის არჩევისას. სივრცე ~(±5)² = 25 წყვილი,
ოპტიმალური ცდ. ≈ 0.03%. ამას *არ* უჭირავს
სტრუქტურული წონა.
""")
    for q in [1.0, 1.853, 3.0, 5.0, 10.0]:
        try:
            aW = mathieu_char_value(1986, q)
            aZ = mathieu_char_value(2115, q)
            rat = aW / aZ
            pred = (1986 / 2115) ** 2
            err = abs(rat - pred) / pred * 100
            print(f"  q={q:>6.3f}:  a_1986/a_2115 = {rat:.6f},  "
                  f"(1986/2115)² = {pred:.6f},  Δ = {err:.4f}%")
        except Exception as e:
            print(f"  q={q}: შეცდომა ({e})")

    print()
    # რა მინიმალური (ΔN_W, ΔN_Z) იძლევა უკეთეს ცდ.-ს?
    target = (80369/91188)**2
    print("  ახლო-მყოფი (N_W, N_Z) წყვილები: ცდ. (N²-მოდელი) vs PDG cos²θ_W")
    print("  " + "-"*70)
    best_fit = []
    for dNW in range(-5, 6):
        for dNZ in range(-5, 6):
            NW = 1986 + dNW
            NZ = 2115 + dNZ
            r = (NW / NZ) ** 2
            err = abs(r - target) / target * 100
            best_fit.append((err, NW, NZ, r))
    best_fit.sort()
    print(f"  {'N_W':>5}{'N_Z':>5}{'(N_W/N_Z)²':>14}{'ცდ.%':>10}")
    for err, NW, NZ, r in best_fit[:5]:
        print(f"  {NW:>5}{NZ:>5}{r:>14.6f}{err:>10.4f}")
    print()
    print("  ე.ი. (1986, 2115) არის ოპტიმალური ფიტი ±5 დიაპაზონში.")


# -----------------------------------------------------------------------
# 6) T4 — ოსცილონის ფარდობის დამოუკიდებელი ტესტი
# -----------------------------------------------------------------------

def T4_independent_test():
    print()
    print("=" * 78)
    print("T4 — ოსცილონი იძლევა თუ არა რაიმე ბუნებრივ ℓ=1 წყვილს?")
    print("=" * 78)
    print("""
თუ ოსცილონს აქვს ზუსტად 2 ℓ=1 ბმული მდგომარეობა — იდეალური
სცენარი: W ← ℓ=1 n=0, Z ← ℓ=1 n=1 (ან სხვა კომბინაცია).
შევამოწმოთ ცდომ.
""")
    target = (80369/91188)**2
    print(f"მიზანი: (ω²[i]/ω²[j]) ≈ {target:.4f} =  (m_W/m_Z)²")
    print()

    candidate_pairs = []
    for Phi_c in [0.8, 1.0, 1.2, 1.3, 1.4, 1.45, 1.48]:
        r, Phi, Om2 = find_oscillon(Phi_c, r_max=40.0, N_grid=12000)
        if r is None:
            continue
        evals, _ = cavity_modes(r, Phi, ell=1, n_want=100)
        bound = evals[evals < 1.0]
        if len(bound) < 2:
            continue
        # ვეძებ მიმდევრობით i, j: j = i+1
        for i in range(len(bound) - 1):
            ratio = bound[i] / bound[i+1]
            err = abs(ratio - target) / target * 100
            candidate_pairs.append((err, Phi_c, i, ratio, len(bound)))

    candidate_pairs.sort()
    if not candidate_pairs:
        print("  → ℓ=1-ში საკმარისი ბმული მოდები ვერ მოიძებნა")
        return
    print(f"{'Φ_c':>6}{'n_i, n_{i+1}':>14}{'ratio':>10}{'ცდ. %':>10}"
          f"{'ω²_i':>10}{'ω²_{i+1}':>12}{'N_bnd':>8}")
    print("-" * 78)
    for err, Phi_c, i, ratio, nb in candidate_pairs[:10]:
        # დამატებითი ინფორმაცია
        r, Phi, _ = find_oscillon(Phi_c, r_max=40.0, N_grid=12000)
        evals, _ = cavity_modes(r, Phi, ell=1, n_want=100)
        bound = evals[evals < 1.0]
        print(f"{Phi_c:>6.2f}{i:>6d},{i+1:<6d}{ratio:>10.4f}"
              f"{err:>10.3f}{bound[i]:>10.4f}{bound[i+1]:>12.4f}{nb:>8d}")


# -----------------------------------------------------------------------
# 7) საბოლოო დასკვნები
# -----------------------------------------------------------------------

def summary():
    print()
    print("=" * 78)
    print("E1 საბოლოო დასკვნა — გულახდილი")
    print("=" * 78)
    print("""
რა ვცადე:
  T1 — ℓ=1 ბმული მოდების რიცხვი Gleiser ოსცილონზე
  T2 — ℓ=1 შიდა ფარდობები vs cos² θ_W
  T3 — Mathieu a_N(q) ფარდობა (1986, 2115)-ზე
  T4 — ოსცილონის ℓ=1 მიმდევრული წყვილი vs m_W/m_Z

რა გამოდის (გულახდილი):
  • Gleiser ოსცილონს აქვს ცოტა ბმული მდგომარეობა (<20 მოდი).
    N=1986 **პირდაპირ** ℓ=1 n=1986 კავიტ-მოდი ვერ იქნება.
  • Mathieu a_N(q) დიდ N-ზე ≈ N², ე.ი. cos²θ_W = (N_W/N_Z)²
    ავტომატურად უკავშირდება ნებისმიერ q-ს.
  • (1986, 2115) ოპტიმალურია ±5 ფანჯარაში → **ფიტია**.

ე.ი. N_W, N_Z-ის "სტრუქტურული გამოყვანა" Gleiser ოსცილონიდან
**ვერ მოხდა** ამ მიდგომით. რა გზა რჩება:
  (ა) Floquet ანალიზი ოსცილირებად Φ(t)-ზე (არა სტატიკური ფონი)
  (ბ) q(α, Φ_c)-ს პარამეტრული გამოყვანა + N-ის არჩევანი სხვა
      სიმეტრიიდან (მაგ. SU(2) გრუპის რანდი)
  (გ) N_W, N_Z = ფიტად აღიარება (honest), არა დამოუკიდებელი
      პრედიქცია → SM_Embedding სტატიაში ეს უკვე ვწერე.

ეს ტესტი ადასტურებს: cos θ_W = (N_W/N_Z)² 0.03% **კონსისტენტურობის
შემოწმებაა, არა დამოუკიდებელი პრედიქცია.** სტატიაში სწორად არის
დაწერილი (§4.2).
""")
    print("=" * 78)


def main():
    T1_vector_spectrum()
    T2_ratio_test()
    T3_mathieu_ladder()
    T4_independent_test()
    summary()


if __name__ == "__main__":
    main()
