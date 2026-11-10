"""Deney 02: sqrt(d_k) ölçeklemesi ve softmax doygunluğu.

Soru: skorlar sqrt(d_k) ile bölünmezse, d_k büyüdükçe softmax ne olur?
Teori: Var(q . k) = d_k  ->  skorlar büyür  ->  softmax one-hot'a gider
       ->  J = diag(p) - p p^T sıfıra gider  ->  gradyan akmaz.
Ölçülenler (her d_k için, ölçekli ve ölçeksiz):
    skor standart sapması, satır maksimumu, entropi H, Jacobian normu ||J||_F
    ve doygun satır oranı (||J||_F < 1e-3, yani o satırdan gradyan geçmiyor)
T = 6 token'lık tek bir dizi gürültülüdür (yalnızca 6 satır), bu yüzden her d_k
için TEKRAR farklı rastgele dizinin ortalaması alınır. Isı haritası ilk dizidir.
Çalıştırma: python experiments/02_scaling.py
"""

import os
import sys

# Proje kök dizinini import yoluna ekle (src klasörünü bulabilmek için)
KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.attention import scaled_dot_product_attention  # noqa: E402
from src.softmax_analysis import satir_entropisi, softmax_jacobian  # noqa: E402
from src.visualization import isi_haritasi, sekil_kaydet  # noqa: E402

TOHUM = 42
T = 6
D_K_DEGERLERI = [4, 16, 64, 256, 1024]
TEKRAR = 200           # her d_k için rastgele dizi sayısı
DOYGUN_ESIK = 1e-3     # ||J||_F bu değerin altındaysa satır doygun sayılır

# Q ve K bileşenleri N(0, 1): teorideki "ortalama 0, varyans 1" varsayımının birebir karşılığı
rng = np.random.default_rng(TOHUM)
sonuclar = {"olcekli": [], "olceksiz": []}
haritalar = {}

for d_k in D_K_DEGERLERI:
    # Satır bazlı ölçüler, TEKRAR * T satır üzerinden toplanır
    biriken = {ad: {"std": [], "max": [], "H": [], "J": []} for ad in sonuclar}
    for tekrar in range(TEKRAR):
        Q = rng.normal(size=(T, d_k))
        K = rng.normal(size=(T, d_k))
        V = rng.normal(size=(T, d_k))
        ham_skor = Q @ K.T                               # (T, T), Var ~ d_k

        for ad, olcek in [("olcekli", True), ("olceksiz", False)]:
            _, A = scaled_dot_product_attention(Q, K, V, scale=olcek)
            b = biriken[ad]
            b["std"].append((ham_skor / np.sqrt(d_k) if olcek else ham_skor).std())
            b["max"].extend(A.max(axis=1))
            b["H"].extend(satir_entropisi(A))
            b["J"].extend(np.linalg.norm(softmax_jacobian(a)) for a in A)
            if tekrar == 0:
                haritalar[(ad, d_k)] = A

    for ad, b in biriken.items():
        J = np.array(b["J"])
        sonuclar[ad].append({
            "std": float(np.mean(b["std"])),                 # ortalama skor standart sapması
            "max": float(np.mean(b["max"])),                 # ortalama satır maksimumu
            "H": float(np.mean(b["H"])),                     # ortalama entropi
            "J": float(J.mean()),                            # ortalama ||J||_F
            "doygun": float(np.mean(J < DOYGUN_ESIK)),       # gradyan geçmeyen satır oranı
        })

print(f"T = {T}, her d_k icin {TEKRAR} dizi, log T = {np.log(T):.3f} (maksimum entropi)")
print("Her hucre: olcekli / olceksiz\n")
print(f"{'d_k':>5} | {'std(S)':>13} | {'satir max':>13} | {'entropi H':>13} | "
      f"{'||J||_F':>15} | {'doygun satir':>13}")
print("-" * 90)
for i, d_k in enumerate(D_K_DEGERLERI):
    o, s = sonuclar["olcekli"][i], sonuclar["olceksiz"][i]
    print(f"{d_k:>5} | {o['std']:>5.2f} / {s['std']:<5.2f} | {o['max']:>5.3f} / {s['max']:<5.3f} | "
          f"{o['H']:>5.3f} / {s['H']:<5.3f} | {o['J']:>6.4f} / {s['J']:<6.4f} | "
          f"{o['doygun']:>5.1%} / {s['doygun']:<5.1%}")

# --- Şekil 1: dört ölçünün d_k ile değişimi ---
fig, eksenler = plt.subplots(1, 4, figsize=(15.0, 3.6), constrained_layout=True)
cizimler = [
    ("max", "Ortalama satır maksimumu", "max$_j$ $A_{ij}$"),
    ("H", "Ortalama satır entropisi", "$H$ (nat)"),
    ("J", "Softmax Jacobian normu", r"ortalama $\|J\|_F$"),
    ("doygun", "Doygun satır oranı", r"$\|J\|_F < 10^{-3}$ olan satırlar"),
]
for ax, (anahtar, baslik, etiket) in zip(eksenler, cizimler):
    for ad, renk, isaret, gosterim in [("olcekli", "#2563eb", "o", r"ölçekli ($/\sqrt{d_k}$)"),
                                       ("olceksiz", "#dc2626", "s", "ölçeksiz")]:
        ax.plot(D_K_DEGERLERI, [r[anahtar] for r in sonuclar[ad]],
                marker=isaret, color=renk, label=gosterim)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("$d_k$ (log ölçek)")
    ax.set_ylabel(etiket)
    ax.set_title(baslik)
    ax.grid(alpha=0.3)
eksenler[1].axhline(np.log(T), color="gray", linestyle="--", linewidth=1, label=r"$\log T$")
eksenler[0].legend()
eksenler[1].legend()
sekil_kaydet(fig, "olcekleme_doygunluk.pdf")

# --- Şekil 2: en büyük d_k için ölçekli ve ölçeksiz dikkat matrisleri (ilk dizi) ---
d_k = D_K_DEGERLERI[-1]
fig, eksenler = plt.subplots(1, 2, figsize=(9.0, 4.2), constrained_layout=True)
isi_haritasi(eksenler[0], haritalar[("olcekli", d_k)], rf"Ölçekli, $d_k = {d_k}$")
im = isi_haritasi(eksenler[1], haritalar[("olceksiz", d_k)], rf"Ölçeksiz, $d_k = {d_k}$")
eksenler[0].set_ylabel("Sorgu (Query) token'ı")
fig.colorbar(im, ax=eksenler, label="Dikkat ağırlığı")
sekil_kaydet(fig, "olcekleme_isi_haritalari.pdf")
