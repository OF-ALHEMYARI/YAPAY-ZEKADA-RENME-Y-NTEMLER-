import numpy as np

from src.attention import stable_softmax
from src.softmax_analysis import jacobian_normu, satir_entropisi, softmax_jacobian


def _sayisal_jacobian(z, eps=1e-6):
    """Merkezi sonlu farklarla softmax Jacobian'ı: (p(z + eps e_j) - p(z - eps e_j)) / 2 eps."""
    T = z.shape[0]
    J = np.zeros((T, T))
    for j in range(T):
        e_j = np.zeros(T)
        e_j[j] = eps
        J[:, j] = (stable_softmax(z + e_j) - stable_softmax(z - e_j)) / (2 * eps)
    return J


def test_jacobian_sayisal_turevle_ayni():
    # Gradyan kontrolü: analitik formül, sayısal türevle eşleşmeli
    z = np.random.default_rng(42).normal(size=6)
    J = softmax_jacobian(stable_softmax(z))
    assert np.allclose(J, _sayisal_jacobian(z), atol=1e-8)


def test_jacobian_simetrik_ve_satir_toplami_sifir():
    # sum_i p_i = 1 sabit olduğundan her satırın (ve sütunun) toplamı 0'dır
    p = stable_softmax(np.random.default_rng(42).normal(size=6))
    J = softmax_jacobian(p)
    assert np.allclose(J, J.T)
    assert np.allclose(J.sum(axis=1), 0.0)


def test_one_hot_dagilimda_jacobian_sifir():
    # Doygunluk: one-hot dağılımda türev tamamen kaybolur
    p = np.array([0.0, 0.0, 1.0, 0.0])
    assert np.allclose(softmax_jacobian(p), 0.0)


def test_entropi_sinirlari():
    T = 6
    duzgun = np.full((1, T), 1.0 / T)
    one_hot = np.eye(T)[:1]
    assert np.allclose(satir_entropisi(duzgun), np.log(T))
    assert np.allclose(satir_entropisi(one_hot), 0.0)


def test_jacobian_normu_doygunlukta_kuculur():
    duzgun = np.full((6, 6), 1.0 / 6)
    neredeyse_one_hot = stable_softmax(np.eye(6) * 50.0)
    assert jacobian_normu(neredeyse_one_hot) < 1e-6 < jacobian_normu(duzgun)
