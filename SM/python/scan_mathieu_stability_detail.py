"""
ISPG — Mathieu-ს სტაბილურობის დეტალური სკანი N ∈ [1, 300]-ზე
==============================================================
კითხვა:  რომელი N-ები გამოიყოფა „განსაკუთრებით სტაბილური"-დ?
         ემთხვევა თუ არა ეს {5, 72, 295}-ს?

მიდგომა:
  ყოველი N-ისთვის გამოვთვლი:
    a_N(q=1.853) — Mathieu-ს მახასიათებელი (mass²-ის ანალოგი)
    b_N(q=1.853) — Mathieu-ს მახასიათებელი მეორე ოჯახიდან
    gap_N = b_N − a_N — არასტაბილურობის ზოლის სიგანე
  სტაბილურობის მაჩვენებელი: log₁₀(1/|gap|).
  რაც პატარაა |gap|, მით უფრო სტაბილურია ზონა.
"""
import numpy as np
from scipy.special import mathieu_a, mathieu_b

q = 1.853
N_vals = np.arange(1, 301)

results = []
for N in N_vals:
    a = mathieu_a(N, q)
    b = mathieu_b(N, q) if N >= 1 else a
    gap = abs(b - a)
    results.append((N, a, b, gap))

# გავარჩიოთ სტაბილური N-ები: gap < threshold
# ძალიან პატარა gap → სტაბილური ზონა
print("=" * 70)
print("  Mathieu სტაბილურობის ზონები  q = 1.853")
print("=" * 70)
print(f"  {'N':>4} {'a_N':>12} {'b_N':>12} {'|gap|':>14}")
print("  " + "-" * 50)
# ვაჩვენოთ N=1..10 + 5, 72, 295 + ყველა „ულტრა-სტაბილური"
for N, a, b, gap in results:
    if N <= 10 or N in (5, 72, 295) or gap < 1e-15:
        flag = ""
        if N in (5, 72, 295): flag = "  ← ლეპტონი"
        print(f"  {N:>4} {a:>12.6f} {b:>12.6f} {gap:>14.3e}{flag}")

# გამოვიყვანოთ 30 ყველაზე სტაბილური N
print("\n  [TOP 30]  ყველაზე მცირე gap (ყველაზე სტაბილური ზონები):")
results_sorted = sorted(results, key=lambda x: x[3])
for i, (N, a, b, gap) in enumerate(results_sorted[:30]):
    flag = "  ← ლეპტონი" if N in (5, 72, 295) else ""
    print(f"    {i+1:>3}.  N = {N:>4}   gap = {gap:.3e}{flag}")

# ვნახოთ რომელი ადგილებზე არის {5, 72, 295}
print("\n  ლეპტონების ადგილი რეიტინგში (gap-ის მიხედვით):")
for N_target in (5, 72, 295):
    for rank, (N, a, b, gap) in enumerate(results_sorted):
        if N == N_target:
            print(f"    N = {N_target:>4}:  ადგილი #{rank+1} / 300,  gap = {gap:.3e}")
            break

# ვნახოთ ტოპ 3 სტაბილური მიუელი-კიბიდან
print("\n  თუ ბუნება ირჩევდა 3 **ყველაზე სტაბილურს**:")
top3 = sorted(results_sorted[:3], key=lambda x: x[0])
for N, a, b, gap in top3:
    mark = " ✓" if N in (5, 72, 295) else " ✗"
    print(f"    N = {N:>4}  ({mark} ემთხვევა თუ არა ლეპტონს)")
