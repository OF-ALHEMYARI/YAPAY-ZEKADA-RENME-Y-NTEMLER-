import numpy as np

from src.attention import scaled_dot_product_attention
from src.multi_head_attention import multi_head_attention


def _veri(T=6, d_model=8, tohum=42):
    """Test için tekrarlanabilir rastgele X ve ağırlık matrisleri üretir."""
    rng = np.random.default_rng(tohum)
    X = rng.normal(size=(T, d_model))
    Wq = rng.normal(size=(d_model, d_model))
    Wk = rng.normal(size=(d_model, d_model))
    Wv = rng.normal(size=(d_model, d_model))
    Wo = rng.normal(size=(d_model, d_model))
    return X, Wq, Wk, Wv, Wo


def test_cikti_sekli_ve_kafa_sayisi():
    X, Wq, Wk, Wv, Wo = _veri()
    cikti, agirliklar = multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads=2)
    assert cikti.shape == (6, 8)
    assert len(agirliklar) == 2
    assert agirliklar[0].shape == (6, 6)


def test_her_kafanin_satir_toplami_bir():
    X, Wq, Wk, Wv, Wo = _veri()
    _, agirliklar = multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads=4)
    for A_h in agirliklar:
        assert np.allclose(A_h.sum(axis=1), 1.0)


def test_tek_kafa_ozdes_matris_ile_tek_dikkate_esit():
    X, Wq, Wk, Wv, _ = _veri()
    Wo = np.eye(8)   # birim matris: çıkış projeksiyonu hiçbir şeyi değiştirmez
    cikti, _ = multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads=1)
    beklenen, _ = scaled_dot_product_attention(X @ Wq, X @ Wk, X @ Wv)
    assert np.allclose(cikti, beklenen)
