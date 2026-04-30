"""
ISPG — Phase 12: პულსონის ჰარმონიკები — 59 მათ შორისაა?
================================================================

მომხმარებლის იდეა: იქნებ 59 პულსონის ჰარმონიკაა?

ტესტი:
   პულსონი: Φ(t) = Φ_c · cos(ωt)  (ცენტრში)
   არაწრფივი კუპლინგი: F(Φ) = Φ(1 − e^{−αΦ})

   F(Φ_c cos ωt) გადაიწერება Fourier-ის მწკრივში:
      F = Σ c_k cos(kωt)

   ჰარმონიკები k = 1, 3, 5, 7, ... (კენტები)

საძიებო საკითხი:
   A. რამდენი მნიშვნელოვანი ჰარმონიკაა? (სპექტრის სიგანე)
   B. არის c_59 განსაკუთრებულად დიდი/მცირე?
   C. არის ფარდობა c_k/c_1, სადაც k=59 ცალკე დგას?
   D. ალტერნატივა: რადიალური Fourier — Φ(r) = Σ a_k sin(kr/R)
      რამდენი სპექტრული კომპონენტია? N=59 ბანდი?
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.fft import fft, fftfreq

ALPHA_NL = 0.5
PHI_C = 2.35

def oscillon_rhs(r, y, Omega, alpha=ALPHA_NL):
    Phi, dPhi = y
    if r < 1e-12:
        d2Phi = (Phi * np.exp(-alpha * Phi) - Omega**2 * Phi) / 3.0
    else:
        d2Phi = -2.0 / r * dPhi - Omega**2 * Phi + Phi * np.exp(-alpha * Phi)
    return [dPhi, d2Phi]

def find_oscillon(Phi_c=PHI_C, alpha=ALPHA_NL, r_max=40.0):
    def residual(Omega):
        sol = solve_ivp(lambda r, y: oscillon_rhs(r, y, Omega, alpha),
                        [1e-10, r_max], [Phi_c, 0.0],
                        method='RK45', rtol=1e-12, atol=1e-14, max_step=0.05)
        return sol.y[0, -1]
    Omega = brentq(residual, 0.30, 0.999, xtol=1e-12)
    r_eval = np.linspace(1e-10, r_max, 6000)
    sol = solve_ivp(lambda r, y: oscillon_rhs(r, y, Omega, alpha),
                    [1e-10, r_max], [Phi_c, 0.0],
                    method='RK45', t_eval=r_eval, rtol=1e-12, atol=1e-14)
    return sol.t, sol.y[0], Omega


print("=" * 72)
print(" Phase 12 — პულსონის ჰარმონიკები — 59 განსაკუთრებულია?")
print("=" * 72)

r, Phi, Omega = find_oscillon()
alpha_phi_c = ALPHA_NL * PHI_C
print(f"\n   Φ_c = {PHI_C},  α = {ALPHA_NL},  αΦ_c = {alpha_phi_c}")
print(f"   Ω   = {Omega:.6f}")

# ===================================================================
# A. დრო-ჰარმონიკები: F(Φ_c cos ωt) Fourier
# ===================================================================

print("\n" + "-" * 72)
print(" A. დრო-ჰარმონიკები: F(Φ_c · cos ωt) Fourier")
print("-" * 72)

# გამოთვალოთ რამდენიმე პერიოდი
N_t = 2**14  # 16384
t = np.linspace(0, 2*np.pi, N_t, endpoint=False)
Phi_t = PHI_C * np.cos(t)
F_t = Phi_t * (1.0 - np.exp(-ALPHA_NL * Phi_t))

# FFT
F_k = np.fft.rfft(F_t) / N_t * 2   # ნორმალიზ. amplitudes
freqs = np.fft.rfftfreq(N_t, d=1.0/N_t)   # integer harmonics

# მხოლოდ ამპლიტუდები
amps = np.abs(F_k)

print(f"\n   პირველი 20 ჰარმონიკის ამპლიტუდა:")
print(f"   {'k':>4} {'|c_k|':>14} {'|c_k/c_1|':>14}")
c1 = amps[1]  # k=1 ფუნდამენტური
for k in range(0, 21):
    ratio = amps[k]/c1 if c1 > 0 else 0
    print(f"   {k:>4d} {amps[k]:>14.6e} {ratio:>14.6e}")

# ===================================================================
# B. მაღალი ჰარმონიკები — 50-70 დიაპაზონი
# ===================================================================

print("\n" + "-" * 72)
print(" B. 50-70 დიაპაზონი (59-ის გარშემო)")
print("-" * 72)

print(f"\n   {'k':>4} {'|c_k|':>14} {'|c_k/c_1|':>14}")
for k in range(50, 71):
    ratio = amps[k]/c1 if c1 > 0 else 0
    marker = "  ← 59!" if k == 59 else ""
    print(f"   {k:>4d} {amps[k]:>14.6e} {ratio:>14.6e}{marker}")

# ===================================================================
# C. რამდენი ჰარმონიკაა "მნიშვნელოვანი"?
# ===================================================================

print("\n" + "-" * 72)
print(" C. სპექტრის სიგანე — რამდენი ჰარმონიკა მნიშვნელოვანია?")
print("-" * 72)

thresholds = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10]
for th in thresholds:
    count = np.sum(amps/c1 > th)
    print(f"   |c_k/c_1| > {th:.0e}:  {count} ჰარმონიკა")

# ბოლო მნიშვნელოვანი ჰარმონიკის ინდექსი
last_k_1em6 = 0
for k in range(len(amps)):
    if amps[k]/c1 > 1e-6:
        last_k_1em6 = k
print(f"\n   ბოლო ჰარმონიკა |c_k/c_1| > 10⁻⁶:  k = {last_k_1em6}")

# ===================================================================
# D. ანალიტიკური: Bessel I_k სტრუქტურა
# ===================================================================

print("\n" + "-" * 72)
print(" D. ანალიტიკური: Bessel I_k(αΦ_c)")
print("-" * 72)

# exp(-αΦ_c cos t) = I_0(αΦ_c) + 2·Σ (-1)^k I_k(αΦ_c) cos(kt)
# ე.ი. F ~ Φ_c cos t · [1 - exp(-αΦ_c cos t)]
#    = Φ_c cos t - Φ_c cos t · [I_0 + 2Σ(-1)^k I_k cos(kt)]

from scipy.special import iv as besseli

print(f"\n   Bessel I_k(αΦ_c = {alpha_phi_c}):")
print(f"   {'k':>4} {'I_k':>14} {'I_k/I_0':>14}")
I_0 = besseli(0, alpha_phi_c)
for k in [0, 1, 2, 3, 5, 10, 20, 50, 59, 60, 70, 100]:
    I_k = besseli(k, alpha_phi_c)
    print(f"   {k:>4d} {I_k:>14.6e} {I_k/I_0:>14.6e}")

print(f"""
   შეინიშნე: I_k(x) ∝ (x/2)^k / k! დიდი k-სთვის.
   თუ x = {alpha_phi_c}, I_59(x) ≈ {besseli(59, alpha_phi_c):.3e} — ძალიან მცირე.
   ე.ი. 59-ე ჰარმონიკა **არ** არის განსაკუთრებული — ის აზუსტდება ნულამდე.
""")

# ===================================================================
# E. Φ(r) რადიალური სპექტრი — Fourier-Bessel?
# ===================================================================

print("-" * 72)
print(" E. Φ(r) რადიალური სპექტრი")
print("-" * 72)

# რადიალური FFT (გაფართოებული 2R-ზე სიმეტრიულად)
r_ext = np.concatenate([-r[::-1][:-1], r])
Phi_ext = np.concatenate([Phi[::-1][:-1], Phi])

N_r = len(r_ext)
dr = r_ext[1] - r_ext[0]
Phi_k = np.fft.rfft(Phi_ext) / N_r
k_r = np.fft.rfftfreq(N_r, d=dr) * 2 * np.pi

amps_r = np.abs(Phi_k)

print(f"\n   რადიალური Fourier სპექტრი (|Φ̂(k)|):")
print(f"   {'k_index':>8} {'k_r':>10} {'|Φ̂|':>14}")
for i in [0, 1, 5, 10, 50, 59, 100, 295, 500]:
    if i < len(amps_r):
        print(f"   {i:>8d} {k_r[i]:>10.4f} {amps_r[i]:>14.6e}")

# ბოლო მნიშვნელოვანი
max_amp = amps_r.max()
last_k_r = 0
for i in range(len(amps_r)):
    if amps_r[i]/max_amp > 1e-6:
        last_k_r = i
print(f"\n   სპექტრის ეფექტური სიგანე (|Φ̂|/max > 10⁻⁶): k_index = {last_k_r}")

# ===================================================================
# F. დასკვნა
# ===================================================================

print("\n" + "=" * 72)
print(" F. დასკვნა")
print("=" * 72)

c59_ratio = amps[59]/c1 if c1 > 0 else 0
c60_ratio = amps[60]/c1 if c1 > 0 else 0

print(f"""
   რა ვიპოვეთ:
   ──────────

   A. დრო-ჰარმონიკა k=59:
      |c_59/c_1| = {c59_ratio:.4e}
      → არასტრუქტურული, ექსპონენციურად ქრება.

   B. Bessel I_59(αΦ_c) = {besseli(59, alpha_phi_c):.3e}
      → არასტრუქტურული.

   C. ბოლო მნიშვნელოვანი ჰარმონიკა (|c_k/c_1| > 10⁻⁶): k = {last_k_1em6}
      → {'59-თან ახლოს!' if abs(last_k_1em6 - 59) < 5 else 'არა 59'}

   დასკვნა:
   ────────
   59 **არ** არის პულსონის ბუნებრივი ჰარმონიკა.
   პულსონის სპექტრი ექსპონენციურად ქრება — ნებისმიერი მაღალი k
   იმდენად პატარაა, რომ სტრუქტურული არ არის.

   ე.ი. თუ 59 არის "რეალური", ის სხვა ზედაპირიდან გამოდის:
     — ან პულსონების რეზონანსი ერთმანეთს შორის (ჯგუფური)
     — ან ფუნდამენტური შესვლელია
""")

print("=" * 72)
print(" Phase 12 — დასრულდა")
print("=" * 72)
