import numpy as np

from src.attention import stable_softmax, scaled_dot_product_attention


def _ornek(T=6, d_k=4, tohum=42):
    """Test için tekrarlanabilir rastgele Q, K, V üretir."""
    rng = np.random.default_rng(tohum)
    Q = rng.normal(size=(T, d_k))
    K = rng.normal(size=(T, d_k))
    V = rng.normal(size=(T, d_k))
    return Q, K, V


def test_sekiller():
    Q, K, V = _ornek()
    cikti, A = scaled_dot_product_attention(Q, K, V)
    assert cikti.shape == (6, 4)
    assert A.shape == (6, 6)


def test_satir_toplami_bir():
    Q, K, V = _ornek()
    _, A = scaled_dot_product_attention(Q, K, V)
    assert np.allclose(A.sum(axis=1), 1.0)


def test_buyuk_degerlerde_nan_yok():
    Q, K, V = _ornek()
    cikti, A = scaled_dot_product_attention(Q * 1000, K * 1000, V)
    assert np.all(np.isfinite(cikti))
    assert np.all(np.isfinite(A))


def test_olcekleme_kapatilabilir():
    # scale=False: skorlar sqrt(d_k)'ya bölünmez, dağılım daha sivri olur
    Q, K, V = _ornek()
    _, A_olcekli = scaled_dot_product_attention(Q, K, V, scale=True)
    _, A_olceksiz = scaled_dot_product_attention(Q, K, V, scale=False)
    assert np.allclose(A_olceksiz.sum(axis=1), 1.0)
    assert A_olceksiz.max() > A_olcekli.max()


def test_softmax_buyuk_girdi():
    z = np.array([[1000.0, 1001.0, 1002.0]])
    p = stable_softmax(z)
    assert np.allclose(p.sum(), 1.0)
    assert np.all(p >= 0)


def test_softmax_kaydirma_degismezligi():
    # softmax(z) == softmax(z + c)  (c sabit)
    z = np.array([[0.1, 0.5, -0.3]])
    assert np.allclose(stable_softmax(z), stable_softmax(z + 100.0))
