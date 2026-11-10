"""Deney 04: Öz-dikkatin zaman ve bellek karmaşıklığı, O(T^2).

Teori: S = Q K^T matrisi (T, T) boyutludur; her elemanı d_k uzunluğunda bir
nokta çarpımıdır. Zaman O(T^2 d_k), dikkat matrisi A için bellek O(T^2).
T iki katına çıkınca ikisi de yaklaşık dört katına çıkmalıdır; log-log
grafikte eğim 2 olmalıdır.

Not: Mutlak süreler bilgisayara bağlıdır ve her çalıştırmada biraz değişir;
yorumlanan şey eğimdir. Bellek değerleri ise kesindir (A.nbytes = T^2 * 8 bayt).
Çalıştırma: python experiments/04_complexity.py
"""

import os
import sys
import time

# Proje kök dizinini import yoluna ekle (src klasörünü bulabilmek için)
KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.attention import scaled_dot_product_attention  # noqa: E402
from src.visualization import sekil_kaydet  # noqa: E402

TOHUM = 42
D_K = 64
T_DEGERLERI = [64, 128, 256, 512, 1024, 2048, 4096]
TEKRAR = 5             # her T için tekrar sayısı; en kısa süre alınır (gürültüyü azaltır)
EGIM_ESIGI = 512       # küçük T'de sabit maliyetler baskın, eğim bu değerden itibaren hesaplanır

rng = np.random.default_rng(TOHUM)
sureler, bellekler = [], []

print(f"d_k = {D_K}, float64, her T icin {TEKRAR} tekrarin en kisasi\n")
print(f"{'T':>6} | {'sure (ms)':>10} | {'A bellek (MB)':>13} | {'sure orani':>10}")
print("-" * 50)

for T in T_DEGERLERI:
    Q = rng.normal(size=(T, D_K))
    K = rng.normal(size=(T, D_K))
    V = rng.normal(size=(T, D_K))

    en_kisa = float("inf")
    for _ in range(TEKRAR):
        baslangic = time.perf_counter()              # time.time()'dan daha hassas
        _, A = scaled_dot_product_attention(Q, K, V)
        en_kisa = min(en_kisa, time.perf_counter() - baslangic)

    sureler.append(en_kisa * 1000.0)                 # ms
    bellekler.append(A.nbytes / 1e6)                 # MB, T^2 * 8 bayt
    oran = "" if len(sureler) == 1 else f"x{sureler[-1] / sureler[-2]:.1f}"
    print(f"{T:>6} | {sureler[-1]:>10.2f} | {bellekler[-1]:>13.1f} | {oran:>10}")

# --- Eğim: log(süre) = a log(T) + b; teori a = 2 ---
T_dizi = np.array(T_DEGERLERI, dtype=float)
secim = T_dizi >= EGIM_ESIGI
egim_sure = np.polyfit(np.log(T_dizi[secim]), np.log(np.array(sureler)[secim]), 1)[0]
egim_bellek = np.polyfit(np.log(T_dizi), np.log(bellekler), 1)[0]
print(f"\nlog-log egim (sure, T >= {EGIM_ESIGI}): {egim_sure:.2f}   (teori: 2)")
print(f"log-log egim (bellek):          {egim_bellek:.2f}   (teori: 2)")

# Donanım etkisi: uzun bağlamda tek bir dikkat matrisinin boyutu (float16 = 2 bayt)
for T_uzun in [8192, 32768, 131072]:
    print(f"T = {T_uzun:>6}: tek dikkat matrisi (float16) = {T_uzun ** 2 * 2 / 1e9:.1f} GB")

# --- Şekil: log-log süre ve bellek, eğimi 2 olan referans çizgisiyle ---
fig, eksenler = plt.subplots(1, 2, figsize=(10.0, 3.8), constrained_layout=True)
for ax, degerler, etiket, baslik in [
    (eksenler[0], sureler, "Süre (ms)", "Zaman"),
    (eksenler[1], bellekler, "$A$ belleği (MB)", "Bellek"),
]:
    ax.loglog(T_dizi, degerler, marker="o", color="#2563eb", label="ölçülen")
    # Referans: son noktadan geçen T^2 eğrisi
    referans = degerler[-1] * (T_dizi / T_dizi[-1]) ** 2
    ax.loglog(T_dizi, referans, linestyle="--", color="gray", label=r"$\propto T^2$ (eğim 2)")
    ax.set_xlabel("Dizi uzunluğu $T$ (log ölçek)")
    ax.set_ylabel(etiket + " (log ölçek)")
    ax.set_title(baslik)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
eksenler[0].set_title(f"Zaman (eğim, $T \\geq {EGIM_ESIGI}$: {egim_sure:.2f})")
sekil_kaydet(fig, "karmasiklik.pdf")
