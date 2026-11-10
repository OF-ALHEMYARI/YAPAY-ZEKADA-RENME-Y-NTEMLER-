"""Ölçekli nokta çarpımı dikkat (Scaled Dot-Product Attention).

Yalnızca NumPy kullanılır; hazır bir dikkat/softmax kütüphanesi yoktur.
Kaynak: Vaswani et al., "Attention Is All You Need" (2017), Denklem (1).
"""

import numpy as np


def stable_softmax(skorlar):
    """Satır bazlı, sayısal olarak kararlı softmax.

    Denklem: p_i = exp(z_i - m) / sum_j exp(z_j - m),   m = max(z)
    m çıkarmak sonucu değiştirmez (pay ve paydadaki exp(-m) sadeleşir),
    ama exp(büyük sayı) = inf taşmasını önler.

    skorlar: (..., T)  ->  döndürür: aynı şekil, her satırın toplamı 1
    """
    maks = np.max(skorlar, axis=-1, keepdims=True)    # (..., 1): her satırın en büyüğü
    ustel = np.exp(skorlar - maks)                    # en büyük üs 0 olur -> exp(0) = 1
    return ustel / np.sum(ustel, axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V, scale=True):
    """Ölçekli nokta çarpımı dikkat.

    Denklem: A = softmax(Q K^T / sqrt(d_k)),   O = A V
    Q, K: (T, d_k)   V: (T, d_v)
    scale=False ise sqrt(d_k) ile bölme yapılmaz (ölçekleme deneyi için).
    Döndürür: (cikti (T, d_v), dikkat_agirliklari (T, T))
    """
    d_k = Q.shape[-1]
    skorlar = Q @ K.T                               # (T, d_k) @ (d_k, T) -> (T, T)

    if scale:
        skorlar = skorlar / np.sqrt(d_k)            # skor varyansını d_k'dan 1'e indirir

    dikkat_agirliklari = stable_softmax(skorlar)    # (T, T), her satırın toplamı 1
    cikti = dikkat_agirliklari @ V                  # (T, T) @ (T, d_v) -> (T, d_v)

    return cikti, dikkat_agirliklari
