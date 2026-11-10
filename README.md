# Self-Attention ve Transformer: Sıfırdan NumPy Uygulaması

ISE 521 (Yapay Zekâda Öğrenme Yöntemleri) dönem projesi, Konu #17.

Öz-dikkat (self-attention) mekanizması, hazır bir derin öğrenme kütüphanesi
kullanılmadan yalnızca NumPy ile uygulanmıştır:

$$A = \mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right), \qquad O = AV$$

## Kurulum

Python 3.11 ile test edilmiştir.

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Çalıştırma

Tüm deneyler tek komutla çalışır:

```
python run_all.py
```

Birim testleri:

```
python -m pytest tests -v
```

Rastgelelik tohumu sabittir (`TOHUM = 42`), bu yüzden sonuçlar her çalıştırmada
birebir aynıdır. Şekiller `results/figures/` klasörüne vektörel PDF olarak yazılır.

## Proje yapısı

```
src/
  attention.py              stable_softmax, scaled_dot_product_attention
  multi_head_attention.py   multi_head_attention
tests/
  test_attention.py         softmax ve dikkat testleri
  test_multi_head.py        çok başlı dikkat testleri
experiments/
  00_self_attention_semasi.py   hesaplama akışı şeması
  01_basic_attention.py         tek başlı dikkat ve ısı haritası
  03_multi_head.py              çok başlı dikkat ve kafa ısı haritaları
results/figures/            üretilen şekiller (PDF)
docs/                       teori notları ve çalışma raporu
run_all.py                  tüm deneyleri çalıştırır
```

## Ortak arayüz

```python
stable_softmax(skorlar)                              # (..., T) -> aynı şekil
scaled_dot_product_attention(Q, K, V, scale=True)    # -> (cikti (T, d_v), A (T, T))
multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads)   # -> (cikti (T, d_model), [A_1, ..., A_h])
```

`scale=False` verildiğinde skorlar $\sqrt{d_k}$ ile bölünmez; bu seçenek
ölçekleme deneyi içindir.

## Örnek boyutlar

| Matris | Şekil | Açıklama |
|---|---|---|
| $X$ | (6, 8) | 6 token, gömme boyutu 8 |
| $W_Q, W_K, W_V$ | (8, 4) | izdüşüm ağırlıkları |
| $Q, K, V$ | (6, 4) | sorgu, anahtar, değer |
| $S = QK^{\top}/\sqrt{d_k}$ | (6, 6) | skorlar |
| $A$ | (6, 6) | dikkat ağırlıkları, her satırın toplamı 1 |
| $O = AV$ | (6, 4) | çıktı |

## Görev dağılımı

| | Kişi 1 | Kişi 2 |
|---|---|---|
| Kod | Q/K/V, ölçekli nokta çarpımı dikkat, kararlı softmax, çok başlı dikkat | ölçekleme deneyi, entropi ve Jacobian, karmaşıklık analizi |
| Dosyalar | `src/attention.py`, `src/multi_head_attention.py`, `tests/`, deney 00, 01, 03 | `src/softmax_analysis.py`, `src/visualization.py`, deney 02, 04 |

## Sınırlılıklar

Ağırlık matrisleri rastgeledir, eğitim yapılmamıştır. Isı haritalarındaki
örüntüler ve kafalar arasındaki farklar yalnızca rastgele izdüşümlerden
kaynaklanır; öğrenilmiş dilsel ilişkileri göstermez.

## Kaynak

Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.
