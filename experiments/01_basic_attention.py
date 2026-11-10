"""Deney 01: Tek başlı öz-dikkat (self-attention) ve ısı haritası.

Adımlar: X -> Q, K, V -> A = softmax(Q K^T / sqrt(d_k)) -> O = A V
Çalıştırma: python experiments/01_basic_attention.py
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

from src.attention import scaled_dot_product_attention

TOHUM = 42          # Rastgelelik tohumu (tekrarlanabilirlik için sabit)
T = 6               # Token sayısı
D_MODEL = 8         # Gömme (embedding) boyutu
D_K = 4             # Sorgu/anahtar boyutu

# --- 1) Girdi: her satır bir token'ın gömme vektörü ---
rng = np.random.default_rng(TOHUM)
tokenler = ["Bu", "bir", "dikkat", "mekanizması", "örneği", "dir"]
X = rng.normal(size=(T, D_MODEL))        # (6, 8)
print("X sekli:", X.shape)

# --- 2) Q, K, V izdüşümleri: Q = X Wq, K = X Wk, V = X Wv ---
Wq = rng.normal(size=(D_MODEL, D_K))     # (8, 4)
Wk = rng.normal(size=(D_MODEL, D_K))     # (8, 4)
Wv = rng.normal(size=(D_MODEL, D_K))     # (8, 4)

Q = X @ Wq   # (6, 8) @ (8, 4) -> (6, 4)
K = X @ Wk   # (6, 4)
V = X @ Wv   # (6, 4)
print("Q sekli:", Q.shape, "| K sekli:", K.shape, "| V sekli:", V.shape)

# --- 3) Dikkat: A = softmax(Q K^T / sqrt(d_k)),  O = A V ---
cikti, A = scaled_dot_product_attention(Q, K, V)
print("A sekli:", A.shape)                       # (6, 6)
print("cikti sekli:", cikti.shape)               # (6, 4)
print("A satir toplamlari:", A.sum(axis=1))      # her satır ~1
print("A (3 ondalik):")
print(np.round(A, 3))

# --- 4) Isı haritası: A[i, j] = i. token'ın j. token'a verdiği dikkat ---
fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(A, cmap="viridis", vmin=0.0, vmax=1.0)
ax.set_xticks(range(T))
ax.set_xticklabels(tokenler, rotation=45, ha="right")
ax.set_yticks(range(T))
ax.set_yticklabels(tokenler)
ax.set_xlabel("Anahtar (Key) token'ı")
ax.set_ylabel("Sorgu (Query) token'ı")
ax.set_title("Dikkat ağırlıkları A (6x6)")
fig.colorbar(im, ax=ax, label="Dikkat ağırlığı")
fig.tight_layout()

klasor = os.path.join(KOK, "results", "figures")
os.makedirs(klasor, exist_ok=True)
# CreationDate=None: PDF her çalıştırmada bayt bayt aynı olur (tekrarlanabilirlik)
fig.savefig(os.path.join(klasor, "dikkat_isi_haritasi.pdf"), metadata={"CreationDate": None})   # vektörel PDF
print("Sekil kaydedildi: results/figures/dikkat_isi_haritasi.pdf")
