"""Softmax doygunluk analizi: entropi ve Jacobian.

Dikkat matrisinin her satırı bir olasılık dağılımıdır. Bu modül, bu dağılımın
ne kadar keskin olduğunu (entropi) ve softmax'ın türevinin ne kadar küçüldüğünü
(Jacobian) ölçer. Ölçekleme deneyi (deney 02) bu fonksiyonları kullanır.
"""

import numpy as np


def satir_entropisi(A):
    """Her satırın Shannon entropisi.

    Denklem: H_i = - sum_j A_ij log A_ij,   0 <= H_i <= log T
    H = log T: düzgün dağılım (her token'a eşit dikkat)
    H = 0    : one-hot dağılım (tek token'a dikkat), yani doygunluk

    A: (T, T), her satırın toplamı 1  ->  döndürür: (T,)
    """
    # 0 * log 0 = 0 kabul edilir (limit); log(0) = -inf hesaplanmasın diye maskelenir
    log_A = np.log(np.where(A > 0, A, 1.0))
    return -np.sum(A * log_A, axis=-1)


def softmax_jacobian(p):
    """Tek bir softmax çıktısının Jacobian matrisi.

    Denklem: dp_i / dz_j = p_i (delta_ij - p_j)
    Matris biçimi: J = diag(p) - p p^T
    Köşegen p_i (1 - p_i), köşegen dışı -p_i p_j'dir. p one-hot'a yaklaşınca
    tüm terimler sıfıra gider: gradyan doygunluğu.

    p: (T,), toplamı 1  ->  döndürür: (T, T), simetrik, her satır toplamı 0
    """
    return np.diag(p) - np.outer(p, p)


def jacobian_normu(A):
    """Dikkat matrisinin satırları için ortalama Frobenius Jacobian normu.

    Denklem: (1/T) sum_i || diag(a_i) - a_i a_i^T ||_F
    Gradyanın softmax'tan ne kadar geçebildiğinin tek sayılık ölçüsüdür.

    A: (T, T)  ->  döndürür: skaler
    """
    return float(np.mean([np.linalg.norm(softmax_jacobian(a)) for a in A]))
