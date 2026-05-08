"""
Phase A — SU(3) × SU(2) × U(1) გეიჯური სექტორი ISPG-ში

მდგომარეობა:
  🟡 SU(3): octet-count clue (D₃-ს ტენზორული დეფორმაცია, gluon_eight.py);
     full Lie algebra / vertices / couplings open (Q.3 closure)
     - N ∈ {8, 10} ბინარული არჩევა
     - V = Δ (რეზონანსული დეფორმაცია), g₀ ფართო დიაპაზონი
     - არა fine-tuning
  ⏳ SU(2): W±, Z — ცნობილია N-ზე Mathieu ლადერზე
  ⏳ U(1): γ მასალო; B-გეიჯური ველი ISPG-ში არ ჩანს

მიზანი: სტრუქტურული შემოწმებები რიცხვითი ცდომილებით.
"""

import numpy as np

# SM EW სექტორის ნაწილაკები
# (name, m_MeV, N_Mathieu, charge, ℓ_assumed?)
ew_bosons = [
    ("γ",         0,    None, 0, 1),   # ფოტონი: masslsss
    ("W+",  80369,    1986, +1, 1),
    ("W-",  80369,    1986, -1, 1),   # igივე N, საპირისპირო Q
    ("Z",   91188,    2115,  0, 1),
    ("H",  125250,    2479,  0, 0),   # სკალარი (ℓ=0)
]

m_e = 0.511  # MeV

print("=" * 78)
print("Phase A — EW ბოზონების სპექტრი Mathieu ლადერზე")
print("=" * 78)
print(f"{'ნაწ.':<5}{'m (GeV)':>10}{'N':>6}{'Q':>5}{'ℓ?':>4}{'m_pred (N²/25)·m_e':>20}{'ცდ.':>8}")
print("-" * 78)
for name, m_MeV, N, Q, l in ew_bosons:
    if N is None:
        print(f"{name:<5}{'0 (ფოტ.)':>10}{'—':>6}{Q:>5}{l:>4}{'—':>20}{'—':>8}")
    else:
        m_pred = (N/5)**2 * m_e / 1000  # GeV-ში
        err = 100 * abs(m_pred*1000 - m_MeV) / m_MeV
        print(f"{name:<5}{m_MeV/1000:>10.3f}{N:>6}{Q:>5}{l:>4}{m_pred:>17.2f} GeV{err:>7.2f}%")

print()
print("=" * 78)
print("ბინარული გადამოწმება: cos θ_W^os (on-shell mass-ratio convention)")
print("=" * 78)
N_W, N_Z = 1986, 2115
cos_th_W_pred = (N_W / N_Z) ** 2
cos_th_W_obs = 80369 / 91188  # m_W/m_Z
cos_th_W_os = 0.8815  # on-shell mass-ratio convention
print(f"(N_W/N_Z)²      = ({N_W}/{N_Z})² = {cos_th_W_pred:.6f}")
print(f"m_W/m_Z (გაზომ.) = {cos_th_W_obs:.6f}")
print(f"cos θ_W^os       = {cos_th_W_os:.4f}")
err = 100 * abs(cos_th_W_pred - cos_th_W_os) / cos_th_W_os
print(f"ცდომილება (N² vs on-shell): {err:.3f}%")

print()
print("=" * 78)
print("SU(3) გლუონების სტრუქტურა (octet-count clue; full derivation open)")
print("=" * 78)
print("""
სკრიპტი: SM/python/gluon_eight_robustness.py

მიდგომა:
  3D ოსცილონი + D₃-სიმეტრიული დეფორმაცია (3 კვარკი ტრიანგულზე)
  ტენზორული მოდები → 8 confined + 2 deconfined = 10 სულ

ბინარული შედეგი:
  N_confined ∈ {8, 10}; 5, 6, 7, 9 — არასოდეს
  N = 8 რეგიონი: g₀ ∈ [0.1, 8.0], V ∈ [3.0, 9.5] (V=Δ≈6.26 ცენტრი)

ფიზიკური სტატუსი:
  V = Δ — რეზონანსული დეფორმაციის სამიზნეა toy model-ში.
  ეს არის octet-count compatibility clue, არა completed derivation:
  - g₀ ფართო დიაპაზონში მუშაობს ამ toy scan-ში
  - D₃ სიმეტრია იძლევა ბინარულ 8/10 count-ს
  - V=Δ, leak threshold, algebra/vertices და G₂(X) დინამიკა ღიაა
""")

print("=" * 78)
print("SU(2) × U(1) — რა გვაქვს და რა აკლია")
print("=" * 78)
print("""
🟢 ცნობილი ფაქტები:
  1. m_W = 80.4 GeV, N=1986 (Mathieu ფიტი)
  2. m_Z = 91.2 GeV, N=2115 (Mathieu ფიტი)
  3. γ მასალოა → Mathieu-ზე პირდაპირი ადგილი არ აქვს
  4. H = 125 GeV, N=2479 (სკალარი, Phase D)
  5. **m_W/m_Z = cos θ_W 0.027% ცდომ.-ით**
     (compatibility / consistency diagnostic)

⏳ დარჩენილი კითხვები:
  (a) რატომ ზუსტად N=1986 W-სთვის? რა კავიტ-მოდი?
  (b) რატომ ზუსტად N=2115 Z-სთვის? რა გაყოფის მექანიზმი?
  (c) γ-ფოტონის გეომეტრიული წარმოშობა (massless → რა ℓ?)
  (d) SU(2) გრუპის გამომავალი სიმეტრია (3 გენერატორი)

💡 შესაძლო ჰიპოთეზები (შესამოწმებელი):
  1. W±, W³ = ℓ=1 ტრიპლეტი (3 m-მნიშვნელობა) კავიტში
     სხვადასხვა რადიალური n-ით
  2. B (hypercharge-ბოზონი) = ℓ=0 რადიალური აღგზნება
  3. Z, γ = W³ და B-ს ორთოგონალური კომბინაცია
     (EW შერევა; Weinberg კუთხის დამოუკიდებელი derivation
      მოითხოვს N_W, N_Z-ის კავიტიდან გამოყვანას)

   თუ N_W და N_Z **გეომეტრიულად** დაფიქსირებულია კავიტ-სპექტრით,
   მაშინ cos θ_W = (N_W/N_Z)² არის **პრედიქცია** (არა ფიტი).
   მიმდინარე ცდომ. რიცხვით თავსებადობას აჩვენებს, მაგრამ derivation
   არ არის, სანამ N_W და N_Z დამოუკიდებლად არ გამოვა.
""")

print("=" * 78)
print("Phase A — დასკვნები")
print("=" * 78)
print("""
🟡 SU(3): octet-count partial robustness (gluon_eight.py + robustness);
          algebra/vertices and G₂(X) parameters remain open
🟡 SU(2): N_W, N_Z ცნობილია Mathieu-ზე; cos θ_W-ის რიცხვითი
          თავსებადობა (0.027%) არის consistency diagnostic;
          ფუნდამენტური "რატომ" — ღია
⏳ U(1): γ მასალო → საჭიროა ცალკე მიდგომა (არა Mathieu)

**Phase A-ს სტატუსი:**
  SU(3) მყარია count-compatibility დონეზე, მაგრამ full
  SU(3)c derivation საჭიროებს algebra/vertices-ს. SU(2)×U(1)
  რიცხვითად **კონსისტენტურია**, მაგრამ derived სტატუსი არ აქვს:
  "რატომ N_W=1986" საჭიროებს
  3D კავიტის სრული ვექტორული სპექტრის გაანგარიშებას — ეს არის
  მომდევნო რიცხვითი ამოცანა.

**წუხანდელი უარი:** არც SU(2), არც U(1)-ს კავიტ-წარმოშობას
ამჟამად **არ** ვაძლევთ პრეცედენტს ISPG-ის ფუნდამენტური
ლაგრანჟიანიდან. ეს რჩება პრიორიტეტულ ღია ფრონტად.
""")
