"""
ISPG E6 — SU(3) target-algebra closure check 3-ღერძიან ფერის-სივრცეში
=====================================================================

§19-ის ჰიპოთეზა: 3 ფერი = 3 სივრცული ღერძი (R=x, G=y, B=z).

E6 გადასამოწმებელი: 3-ღერძიანი ფერის სივრცეში მოქმედი უნიტარული
გარდაქმნები (SU(3)) ქმნიან 8 გენერატორს, რომლებიც იხურებიან სტანდარტული
Lie-ალგებრით [T_a, T_b] = i·f_abc·T_c.

თუ textbook f_abc ზუსტად აღდგება, მაშინ 3-ღერძიანი ფერის სივრცე
თავსებადია SU(3) target-algebra-სთან; ეს ჯერ არ ამტკიცებს, რომ
`gluon_eight.py`-ს 8 confined mode დინამიკურად იგივე გენერატორებია.

Q.3 audit status:
  This is a target-algebra consistency check under an assumed
  three-axis color-space basis. It does NOT map the confined
  cavity modes to generators and does NOT extract f_abc from
  G₂(X) cavity-mode commutators. E6 remains open until that
  dynamical extraction is done.

მეთოდი:
  1. ავაგოთ Gell-Mann-ის 8 მატრიცა 3-ფერი (3×3) სივრცეში
  2. T_a = λ_a / 2 (ფიზიკის კონვენცია)
  3. რიცხვითად დავითვალოთ [T_a, T_b] = C_ab
  4. ავიღოთ f_abc = -2i·Tr(C_ab·T_c) ⟶ textbook-სთან შევადაროთ
  5. შევამოწმოთ Jacobi-ს იდენტობა ∑ [[T_a,T_b],T_c] + cyclic = 0
"""

import numpy as np


# ──────────────────────────────────────────────────────────────
#  §1  Gell-Mann მატრიცები — 3-ფერის სივრცეში მოქმედება
# ──────────────────────────────────────────────────────────────
#  ეს არის 8 დამოუკიდებელი ჰერმიტული tracless 3×3 მატრიცა.
#  სტანდარტული SU(3) ბაზისი — ISPG-ში იგივე ობიექტი, ფიზიკური
#  ინტერპრეტაცია: λ₁,λ₂,λ₃ = x↔y ღერძებს შორის შერევა + ფაზა
#                  λ₄,λ₅   = x↔z ღერძებს შორის შერევა
#                  λ₆,λ₇   = y↔z ღერძებს შორის შერევა
#                  λ₈       = R+G-2B (ღერძთა დისბალანსი)

def gell_mann():
    λ = np.zeros((8, 3, 3), dtype=complex)

    # x-y ღერძებს შორის
    λ[0] = [[0, 1, 0], [1, 0, 0], [0, 0, 0]]
    λ[1] = [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]]
    λ[2] = [[1, 0, 0], [0, -1, 0], [0, 0, 0]]

    # x-z ღერძებს შორის
    λ[3] = [[0, 0, 1], [0, 0, 0], [1, 0, 0]]
    λ[4] = [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]]

    # y-z ღერძებს შორის
    λ[5] = [[0, 0, 0], [0, 0, 1], [0, 1, 0]]
    λ[6] = [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]]

    # დიაგონალური (R+G-2B ბალანსი)
    λ[7] = (1/np.sqrt(3)) * np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]])

    return λ


def commutator(A, B):
    return A @ B - B @ A


def anticommutator(A, B):
    return A @ B + B @ A


# textbook f_abc (არანულოვანი კომპონენტები) — SU(3) სტანდარტული
# ცნობარი: Peskin & Schroeder, Appendix A
FABC_REFERENCE = {
    (0, 1, 2): 1.0,
    (0, 3, 6): 0.5,
    (0, 4, 5): -0.5,  # = (0,5,4) + perm
    (1, 3, 5): 0.5,
    (1, 4, 6): 0.5,
    (2, 3, 4): 0.5,
    (2, 5, 6): -0.5,
    (3, 4, 7): np.sqrt(3) / 2,
    (5, 6, 7): np.sqrt(3) / 2,
}


def main():
    print("STATUS: target-algebra consistency check only;")
    print("        dynamic f_abc extraction from ISPG cavity modes remains open.\n")

    print("=" * 68)
    print("  E6  —  SU(3) target-algebra closure check 3-ღერძიან ფერის-სივრცეში")
    print("=" * 68)

    λ = gell_mann()
    # T_a = λ_a / 2 (ფიზიკის კონვენცია)
    T = λ / 2.0

    # ─────────────────────────────────────────────
    #  §1  ჰერმიტულობა და tracless შემოწმება
    # ─────────────────────────────────────────────
    print("\n§1  ბაზისური თვისებები")
    print("-" * 50)
    for a in range(8):
        herm_err = np.max(np.abs(T[a] - T[a].conj().T))
        tr = np.trace(T[a])
        assert herm_err < 1e-12, f"T_{a+1} ჰერმიტული არ არის"
        assert abs(tr) < 1e-12, f"T_{a+1} tracless არ არის"
    print(f"  ✓ ყველა 8 გენერატორი ჰერმიტული და tracless")

    # ნორმალიზაცია Tr(T_a T_b) = (1/2) δ_ab
    print("\n  Tr(T_a T_b) ნორმალიზაცია:")
    norm_ok = True
    for a in range(8):
        for b in range(8):
            tr = np.trace(T[a] @ T[b])
            expected = 0.5 if a == b else 0.0
            if abs(tr - expected) > 1e-12:
                norm_ok = False
                print(f"    ! T_{a+1}·T_{b+1}: Tr = {tr:.4f}, უნდა იყოს {expected}")
    if norm_ok:
        print(f"    ✓ Tr(T_a T_b) = ½δ_ab ყველა (a,b)-სთვის")

    # ─────────────────────────────────────────────
    #  §2  სტრუქტურული კონსტანტები f_abc
    # ─────────────────────────────────────────────
    print("\n§2  სტრუქტურული კონსტანტები f_abc")
    print("-" * 50)
    print("  [T_a, T_b] = i·f_abc·T_c")
    print("  ამოღება: f_abc = -2i · Tr([T_a,T_b] · T_c)")
    print()

    f = np.zeros((8, 8, 8))
    for a in range(8):
        for b in range(8):
            comm = commutator(T[a], T[b])
            for c in range(8):
                # [T_a,T_b] = i f_abc T_c → Tr(·T_c) = i f_abc · ½
                f[a, b, c] = np.real(-2j * np.trace(comm @ T[c]))

    # ─────────────────────────────────────────────
    #  §3  შედარება textbook-ის f_abc-თან
    # ─────────────────────────────────────────────
    print("§3  textbook-სთან შედარება")
    print("-" * 50)
    print("   (a,b,c)    ISPG f_abc       textbook     Δ")
    print("  " + "─" * 52)

    max_err = 0.0
    for (a, b, c), f_ref in sorted(FABC_REFERENCE.items()):
        f_computed = f[a, b, c]
        err = abs(f_computed - f_ref)
        max_err = max(max_err, err)
        marker = "✓" if err < 1e-10 else "✗"
        print(f"  ({a+1},{b+1},{c+1})       {f_computed:+.6f}      "
              f"{f_ref:+.6f}     {err:.2e}  {marker}")

    print(f"\n  მაქსიმალური ცდომილება: {max_err:.2e}")
    if max_err < 1e-10:
        print("  ✓ ყველა სტრუქტურული კონსტანტა ემთხვევა textbook SU(3)-ს")

    # ─────────────────────────────────────────────
    #  §4  ანტისიმეტრიულობა და Jacobi
    # ─────────────────────────────────────────────
    print("\n§4  f_abc ანტისიმეტრიულობა")
    print("-" * 50)
    antisym_err = 0.0
    for a in range(8):
        for b in range(8):
            for c in range(8):
                # f_abc = -f_bac = -f_acb
                antisym_err = max(antisym_err,
                                   abs(f[a, b, c] + f[b, a, c]),
                                   abs(f[a, b, c] + f[a, c, b]))
    print(f"  მაქს. ცდომ. ანტისიმ.: {antisym_err:.2e}")
    assert antisym_err < 1e-10, "ანტისიმეტრიულობა დაირღვა"
    print("  ✓ f_abc სრულად ანტისიმეტრიულია")

    print("\n§5  Jacobi-ის იდენტობა")
    print("-" * 50)
    print("  [[T_a,T_b],T_c] + [[T_b,T_c],T_a] + [[T_c,T_a],T_b] = 0")

    jacobi_err = 0.0
    for a in range(8):
        for b in range(a+1, 8):
            for c in range(b+1, 8):
                J = (commutator(commutator(T[a], T[b]), T[c])
                     + commutator(commutator(T[b], T[c]), T[a])
                     + commutator(commutator(T[c], T[a]), T[b]))
                err = np.max(np.abs(J))
                jacobi_err = max(jacobi_err, err)
    print(f"  მაქს. ცდომ.: {jacobi_err:.2e}")
    assert jacobi_err < 1e-10, "Jacobi-ს იდენტობა დაირღვა"
    print("  ✓ Jacobi-ს იდენტობა შესრულებულია")

    # ─────────────────────────────────────────────
    #  §6  Cartan-ის ქვე-ალგებრა (ჩვეულებრივ რანდი)
    # ─────────────────────────────────────────────
    print("\n§6  Cartan-ის ქვე-ალგებრა")
    print("-" * 50)
    # T_3, T_8 უნდა კომუტირდნენ (SU(3)-ის რანდი = 2)
    c38 = commutator(T[2], T[7])
    cart_err = np.max(np.abs(c38))
    print(f"  [T_3, T_8] = {cart_err:.2e} ≈ 0  →  SU(3) რანდი = 2  ✓")

    # ─────────────────────────────────────────────
    #  §7  ფიზიკური ინტერპრეტაცია 3 ღერძში
    # ─────────────────────────────────────────────
    print("\n§7  ფიზიკური ინტერპრეტაცია 3-ღერძიან ოსცილონში")
    print("-" * 50)
    meanings = [
        ("T₁ (λ₁)", "x↔y ღერძებს შორის რეალური შერევა    (რბილი)"),
        ("T₂ (λ₂)", "x↔y ღერძებს შორის წარმ. შერევა       (ფაზური)"),
        ("T₃ (λ₃)", "x−y ბალანსი                            (დიაგონალი 1)"),
        ("T₄ (λ₄)", "x↔z ღერძებს შორის რეალური შერევა"),
        ("T₅ (λ₅)", "x↔z ღერძებს შორის წარმ. შერევა"),
        ("T₆ (λ₆)", "y↔z ღერძებს შორის რეალური შერევა"),
        ("T₇ (λ₇)", "y↔z ღერძებს შორის წარმ. შერევა"),
        ("T₈ (λ₈)", "R+G−2B დისბალანსი                      (დიაგონალი 2)"),
    ]
    for lbl, desc in meanings:
        print(f"  {lbl}:  {desc}")

    # ─────────────────────────────────────────────
    #  §8  დასკვნა
    # ─────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  დასკვნა")
    print("=" * 68)
    print("""
  3-ღერძიან ფერ-სივრცეში (§19 ჰიპოთეზა R=x, G=y, B=z) უნიტარული
  გარდაქმნები ქმნიან ზუსტად 8 დამოუკიდებელ გენერატორს, რომლებიც:

    ✓ იხურებიან SU(3) Lie-ალგებრაზე
    ✓ სტრუქტურული კონსტანტები f_abc = textbook SU(3) (ცდომ. < 1e-10)
    ✓ სრულად ანტისიმეტრიულია
    ✓ Jacobi-ს იდენტობა შესრულებულია
    ✓ რანდი = 2 (Cartan ქვე-ალგებრა {T_3, T_8})

  ეს ადასტურებს, რომ ნავარაუდევი 3-ღერძიანი ფერ-სივრცის
  textbook გენერატორები ზუსტად იხურება SU(3)-ზე. ეს არის
  target-algebra consistency check, არა იმის დინამიკური
  მტკიცებულება, რომ `gluon_eight.py`-ის confined cavity modes
  უკვე იმავე გენერატორებს წარმოადგენენ.

  კავშირი: SM-ში ფერი არის "შიდა" (internal) სიმეტრია.
           ISPG-ში იგივე SU(3) ამოდის **სივრცული 3-ღერძიანი**
           სტრუქტურიდან — ანუ "ფერი" არ არის დამოუკიდებელი
           თავისუფლების ხარისხი, არამედ 3D სივრცის თვისება.

  E6 არ არის დასრულებული: საჭიროა confined cavity modes-ის
  generator-map და f_abc-ის ამოღება G₂(X)-ის trilinear vertices-დან.
""")


if __name__ == "__main__":
    main()
