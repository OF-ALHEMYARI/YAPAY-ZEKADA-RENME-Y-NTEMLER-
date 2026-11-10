"""Şema 00: Öz-dikkat (self-attention) hesaplama akışı.

Rapor için vektörel (PDF) blok şeması çizer:
    X -> Q, K, V -> S = Q K^T -> S / sqrt(d_k) -> softmax -> A -> O = A V
Çalıştırma: python experiments/00_self_attention_semasi.py
"""

import os

import matplotlib

matplotlib.use("Agg")   # pencere açmadan dosyaya çiz
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RENK_VERI = "#dbeafe"      # veri (matris) kutuları
RENK_ISLEM = "#fef3c7"     # işlem kutuları
RENK_KENAR = "#1f2937"


def kutu(ax, x, y, metin, renk, genislik=2.6, yukseklik=0.9):
    """(x, y) merkezli, yuvarlak köşeli bir kutu ve içine metin çizer."""
    ax.add_patch(FancyBboxPatch(
        (x - genislik / 2, y - yukseklik / 2), genislik, yukseklik,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=renk, edgecolor=RENK_KENAR, linewidth=1.2))
    ax.text(x, y, metin, ha="center", va="center", fontsize=10)


def ok(ax, baslangic, bitis, etiket=None, etiket_kaydir=(0.0, 0.0)):
    """baslangic -> bitis yönünde bir ok ve isteğe bağlı etiket çizer."""
    ax.annotate("", xy=bitis, xytext=baslangic,
                arrowprops=dict(arrowstyle="-|>", color=RENK_KENAR, linewidth=1.2))
    if etiket:
        orta_x = (baslangic[0] + bitis[0]) / 2 + etiket_kaydir[0]
        orta_y = (baslangic[1] + bitis[1]) / 2 + etiket_kaydir[1]
        ax.text(orta_x, orta_y, etiket, ha="center", va="center", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))


fig, ax = plt.subplots(figsize=(7.0, 8.0))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12.2)
ax.axis("off")

# --- Kutular (yukarıdan aşağıya akış) ---
kutu(ax, 5.0, 11.4, r"Girdi  $X \in \mathbb{R}^{T \times d_{model}}$", RENK_VERI, genislik=4.2)

kutu(ax, 2.0, 9.2, r"$Q$  (Sorgu)" + "\n" + r"$T \times d_k$", RENK_VERI)
kutu(ax, 5.0, 9.2, r"$K$  (Anahtar)" + "\n" + r"$T \times d_k$", RENK_VERI)
kutu(ax, 8.0, 9.2, r"$V$  (Değer)" + "\n" + r"$T \times d_v$", RENK_VERI)

kutu(ax, 3.5, 7.3, r"Skorlar  $S = Q K^{\top}$" + "\n" + r"$T \times T$", RENK_ISLEM, genislik=3.4)
kutu(ax, 3.5, 5.7, r"Ölçekleme  $S / \sqrt{d_k}$", RENK_ISLEM, genislik=3.4)
kutu(ax, 3.5, 4.1, "Softmax (satır bazlı)", RENK_ISLEM, genislik=3.4)
kutu(ax, 3.5, 2.5, r"Dikkat ağırlıkları  $A$" + "\n" + r"$T \times T$, satır toplamı 1",
     RENK_VERI, genislik=3.8)
kutu(ax, 5.5, 0.8, r"Çıktı  $O = A V \in \mathbb{R}^{T \times d_v}$", RENK_VERI, genislik=4.4)

# --- Oklar ---
ok(ax, (3.6, 10.95), (2.0, 9.65), r"$X W_Q$", etiket_kaydir=(-0.35, 0.0))
ok(ax, (5.0, 10.95), (5.0, 9.65), r"$X W_K$")
ok(ax, (6.4, 10.95), (8.0, 9.65), r"$X W_V$", etiket_kaydir=(0.35, 0.0))

ok(ax, (2.0, 8.75), (3.0, 7.75))
ok(ax, (5.0, 8.75), (4.0, 7.75))
ok(ax, (3.5, 6.85), (3.5, 6.15))
ok(ax, (3.5, 5.25), (3.5, 4.55))
ok(ax, (3.5, 3.65), (3.5, 2.95))
ok(ax, (3.5, 2.05), (4.8, 1.25))
ok(ax, (8.0, 8.75), (6.4, 1.25), r"$V$", etiket_kaydir=(0.3, 0.0))

ax.set_title("Ölçekli nokta çarpımı öz-dikkat: hesaplama akışı", fontsize=12)
fig.tight_layout()

klasor = os.path.join(KOK, "results", "figures")
os.makedirs(klasor, exist_ok=True)
# CreationDate=None: PDF her çalıştırmada bayt bayt aynı olur (tekrarlanabilirlik)
fig.savefig(os.path.join(klasor, "self_attention_semasi.pdf"), metadata={"CreationDate": None})   # vektörel PDF
print("Sekil kaydedildi: results/figures/self_attention_semasi.pdf")
