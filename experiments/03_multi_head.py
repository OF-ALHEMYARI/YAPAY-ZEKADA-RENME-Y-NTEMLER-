"""Deney 03: Çok başlı dikkat (Multi-Head Attention) ve kafa ısı haritaları.

Her kafa d_k = d_model / kafa_sayisi boyutlu ayrı bir alt uzayda dikkat hesaplar.
Çalıştırma: python experiments/03_multi_head.py

Not: Ağırlıklar rastgeledir (eğitim yok). Kafalar arasındaki fark yalnızca
farklı rastgele izdüşümlerden kaynaklanır; öğrenilmiş dilsel ilişki göstermez.
"""

import os
import sys

# Proje kök dizinini import yoluna ekle (src klasörünü bulabilmek için)
KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)

import matplotlib

matplotlib.use("Agg")   # pencere açmadan dosyaya çiz
import matplotlib.pyplot as plt
import numpy as np

from src.multi_head_attention import multi_head_attention

TOHUM = 42
T, D_MODEL, KAFA_SAYISI = 6, 8, 2

rng = np.random.default_rng(TOHUM)
tokenler = ["Bu", "bir", "dikkat", "mekanizması", "örneği", "dir"]
X = rng.normal(size=(T, D_MODEL))            # (6, 8)
Wq = rng.normal(size=(D_MODEL, D_MODEL))     # (8, 8)
Wk = rng.normal(size=(D_MODEL, D_MODEL))     # (8, 8)
Wv = rng.normal(size=(D_MODEL, D_MODEL))     # (8, 8)
Wo = rng.normal(size=(D_MODEL, D_MODEL))     # (8, 8)

# MultiHead(X) = Concat(kafa_1, ..., kafa_h) W_O
cikti, agirliklar = multi_head_attention(X, Wq, Wk, Wv, Wo, KAFA_SAYISI)
print("Cikti sekli:", cikti.shape)           # (6, 8)
for h, A_h in enumerate(agirliklar):
    print(f"Kafa {h + 1} dikkat sekli:", A_h.shape, "| satir toplamlari = 1:",
          np.allclose(A_h.sum(axis=1), 1.0))

# --- Her kafa için bir ısı haritası (yan yana, ortak renk ölçeği) ---
fig, eksenler = plt.subplots(1, KAFA_SAYISI, figsize=(5.0 * KAFA_SAYISI, 4.5),
                             constrained_layout=True)
for h, (ax, A_h) in enumerate(zip(eksenler, agirliklar)):
    im = ax.imshow(A_h, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(T))
    ax.set_xticklabels(tokenler, rotation=45, ha="right")
    ax.set_yticks(range(T))
    ax.set_yticklabels(tokenler)
    ax.set_xlabel("Anahtar (Key) token'ı")
    ax.set_title(f"Kafa {h + 1}")
eksenler[0].set_ylabel("Sorgu (Query) token'ı")
fig.colorbar(im, ax=eksenler, label="Dikkat ağırlığı")

klasor = os.path.join(KOK, "results", "figures")
os.makedirs(klasor, exist_ok=True)
# CreationDate=None: PDF her çalıştırmada bayt bayt aynı olur (tekrarlanabilirlik)
fig.savefig(os.path.join(klasor, "cok_basli_dikkat.pdf"), metadata={"CreationDate": None})   # vektörel PDF
print("Sekil kaydedildi: results/figures/cok_basli_dikkat.pdf")
