"""Çok başlı dikkat (Multi-Head Attention).

Kaynak: Vaswani et al. (2017), Denklem (4)-(5):
    MultiHead(X) = Concat(kafa_1, ..., kafa_h) W_O
    kafa_i       = Attention(Q_i, K_i, V_i)
"""

import numpy as np

from src.attention import scaled_dot_product_attention


def multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads):
    """Çok başlı dikkat.

    X: (T, d_model);  Wq, Wk, Wv, Wo: (d_model, d_model)
    Her kafa d_k = d_model / num_heads boyutlu bir alt uzayda çalışır.
    Döndürür: (cikti (T, d_model), kafa_agirliklari: liste, her biri (T, T))
    """
    T, d_model = X.shape
    assert d_model % num_heads == 0, "d_model, num_heads'e tam bölünmeli"
    d_k = d_model // num_heads

    Q = X @ Wq   # (T, d_model) @ (d_model, d_model) -> (T, d_model)
    K = X @ Wk   # (T, d_model)
    V = X @ Wv   # (T, d_model)

    kafa_ciktilari = []
    kafa_agirliklari = []
    for h in range(num_heads):
        s = slice(h * d_k, (h + 1) * d_k)           # bu kafanın sütunları (alt uzay)
        cikti_h, A_h = scaled_dot_product_attention(Q[:, s], K[:, s], V[:, s])
        kafa_ciktilari.append(cikti_h)              # (T, d_k)
        kafa_agirliklari.append(A_h)                # (T, T)

    birlesik = np.concatenate(kafa_ciktilari, axis=-1)   # Concat: (T, h * d_k) = (T, d_model)
    cikti = birlesik @ Wo                                # W_O kafaları karıştırır: (T, d_model)
    return cikti, kafa_agirliklari
