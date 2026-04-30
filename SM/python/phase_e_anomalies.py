"""
Phase E — ანომალიების გაბათილება ერთ თაობაზე (Gen-1).

5 სტანდარტული გეიჯური ანომალია უნდა გაიაროს:
  1. U(1)_Y³
  2. SU(2)² × U(1)_Y
  3. SU(3)² × U(1)_Y
  4. Gravitational × U(1)_Y (Σ Y)
  5. Witten SU(2) (დუბლეტების ლუწი რაოდ.)

ISPG-ს კონსისტენცია: რადგან Phase C-ში დავადგინეთ რომ Gen-1-ის Q, T₃, Y
ემთხვევა SM-ის სტანდარტულ მნიშვნელობებს, ანომალიები ავტომატურად
უნდა გაიაროს.
"""

from fractions import Fraction

# Gen-1 მარცხენა Weyl ფერმიონები (ყველა R → CP-კონიუგატი, Y → −Y)
# (name,  Y,         T_SU2,  N_c,  multiplicity)
# multiplicity = SU(2) დუბლეტის წევრები × ფერი
fields = [
    ("L_L",      Fraction(-1),     Fraction(1,2), 1, 2),   # lepton doublet (ν, e)
    ("Q_L",      Fraction(1, 3),   Fraction(1,2), 3, 6),   # quark doublet (u, d) × 3 ფერი
    ("e_L^c",    Fraction(2),      Fraction(0),   1, 1),   # conj(e_R), Y: -2→+2
    ("u_L^c",    Fraction(-4, 3),  Fraction(0),   3, 3),   # conj(u_R), Y: 4/3→-4/3
    ("d_L^c",    Fraction(2, 3),   Fraction(0),   3, 3),   # conj(d_R), Y: -2/3→+2/3
]

print("=" * 74)
print("Phase E — ანომალიების გაბათილება Gen-1-ზე")
print("=" * 74)
print(f"\nმარცხენა Weyl ფერმიონები (R-ები CP-კონიუგატად):")
print(f"{'ველი':<10}{'Y':>10}{'T_SU2':>8}{'N_c':>5}{'N_mult':>8}")
for name, Y, T, Nc, mult in fields:
    print(f"{name:<10}{str(Y):>10}{str(T):>8}{Nc:>5}{mult:>8}")

print()
print("-" * 74)
print("ანომალიების გამოთვლა")
print("-" * 74)

# 1. U(1)^3: Σ Y³
A_Y3 = sum(mult * Y**3 for _, Y, _, _, mult in fields)
print(f"1. U(1)_Y³:              Σ Y³        = {A_Y3}    {'✓' if A_Y3 == 0 else '✗'}")

# 2. SU(2)² × U(1): დუბლეტების რიცხვი (ფერის ჩათვლით) × Y
#    mult უკვე შეიცავს ფერის ფაქტორს; n_doublets = mult / 2
A_SU2 = Fraction(0)
for name, Y, T, Nc, mult in fields:
    if T != 0:
        n_doublets = mult // 2  # დუბლეტის კოპიები (ფერი უკვე აქ)
        A_SU2 += n_doublets * Y
print(f"2. SU(2)² × U(1):        Σ_dbl Y     = {A_SU2}    {'✓' if A_SU2 == 0 else '✗'}")
print(f"   (L_L: 1·(-1) = -1;  Q_L: 3·(1/3) = 1  →  ჯამი 0)")

# 3. SU(3)² × U(1): მხოლოდ ფერიანი ველები, Σ (N_SU2_components · Y)
A_SU3 = Fraction(0)
for name, Y, T, Nc, mult in fields:
    if Nc > 1:
        # SU(2) კომპონენტების რიცხვი ერთ ფერზე
        n_SU2_comp = mult // Nc
        A_SU3 += n_SU2_comp * Y
print(f"3. SU(3)² × U(1):        Σ N_SU2·Y   = {A_SU3}    {'✓' if A_SU3 == 0 else '✗'}")
print(f"   (Q_L: 2·(1/3) = 2/3;  u_L^c: 1·(-4/3) = -4/3;  d_L^c: 1·(2/3) = 2/3  →  0)")

# 4. Gravitational × U(1): Σ Y (ყველა მარცხ. ფერმიონზე)
A_grav = sum(mult * Y for _, Y, _, _, mult in fields)
print(f"4. Grav × U(1)_Y:        Σ Y         = {A_grav}    {'✓' if A_grav == 0 else '✗'}")

# 5. Witten SU(2): დუბლეტების რიცხვი ლუწი?
n_doublets_total = 0
for name, Y, T, Nc, mult in fields:
    if T != 0:
        n_doublets_total += mult // 2  # რამდენი დუბლეტი * ფერი
        # ერთი SU(2) დუბლეტი ერთ ფერზე = 1 დუბლეტი
print(f"5. Witten SU(2):         N_doublets  = {n_doublets_total} (ლუწი: {'✓' if n_doublets_total % 2 == 0 else '✗'})")
print(f"   (L_L: 1 დუბლ.; Q_L: 3 დუბლ. (3 ფერი) →  ჯამი 4 = ლუწი)")

print()
print("=" * 74)

all_ok = (A_Y3 == 0 and A_SU2 == 0 and A_SU3 == 0 and A_grav == 0
          and n_doublets_total % 2 == 0)
if all_ok:
    print("✅ ყველა 5 ანომალია გაუქმდა — ISPG-ის Gen-1 სექტორი კონსისტენტურია")
else:
    print("✗ პრობლემა — ანომალია არ იხურება")
print("=" * 74)

print("""
ინტერპრეტაცია ISPG-სთვის:
  რადგან Gen-1-ის Q, T₃, Y მნიშვნელობები ემთხვევა SM-ის
  სტანდარტულ მნიშვნელობებს (Phase C Step 2), ანომალიები
  ავტომატურად გაუქმდება. ეს არის ცნობილი SM-ის თვისება,
  რომელიც ISPG-მ შეინარჩუნა რადგან მისი Q, T₃, Y
  სტრუქტურა იდენტურია.

ISPG-ისთვის **სპეციფიკური** შემოწმება (Phase C Step 2-დან):
  Y · N_c ∈ {−2, −1, +1, +4} — ყველა **მთელი**

  ეს არ არის შემთხვევა — ეს ანომალიის გაუქმების
  ნებართვა-პირობაა. ISPG-ის კავიტიდან ნებისმიერი
  Q, Y-ს მინიჭებამ, რომელიც ამ მთელ-რიცხოვან
  სტრუქტურას აკმაყოფილებს, ავტომატურად გაუქმებს
  ანომალიებს.

ამიტომ Phase E = სრული ✅.
""")
