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
  softmax_analysis.py       satir_entropisi, softmax_jacobian, jacobian_normu
  visualization.py          ortak çizim yardımcıları (PDF kaydı, ısı haritası)
tests/
  test_attention.py         softmax ve dikkat testleri
  test_multi_head.py        çok başlı dikkat testleri
  test_softmax_analysis.py  Jacobian (sayısal türevle) ve entropi testleri
experiments/
  00_self_attention_semasi.py   hesaplama akışı şeması
  01_basic_attention.py         tek başlı dikkat ve ısı haritası
  02_scaling.py                 sqrt(d_k) ölçeklemesi ve softmax doygunluğu
  03_multi_head.py              çok başlı dikkat ve kafa ısı haritaları
  04_complexity.py              O(T^2) zaman ve bellek ölçümü
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

```python
satir_entropisi(A)        # (T, T) -> (T,),  H_i = -sum_j A_ij log A_ij
softmax_jacobian(p)       # (T,) -> (T, T),  J = diag(p) - p p^T
jacobian_normu(A)         # (T, T) -> skaler, satırların ortalama ||J||_F
```

## Temel bulgular

| Deney | Bulgu |
|---|---|
| 02, ölçeklemesiz | $d_k = 1024$'te skor standart sapması 31.26 ($\sqrt{d_k} = 32$), satırların %73.5'i doygun ($\|J\|_F < 10^{-3}$), ortalama entropi 0.067 |
| 02, ölçekli | tüm $d_k$ değerlerinde skor standart sapması ≈ 1, doygun satır %0, entropi ≈ 1.47 ($\log 6 = 1.79$) |
| 04 | süre ve bellek için log-log eğim ≈ 2, yani $\mathcal{O}(T^2)$ |

Deney 04'teki mutlak süreler bilgisayara göre değişir; eğim ve bellek değerleri
değişmez. Diğer tüm sayılar ve şekiller her çalıştırmada birebir aynıdır.

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
| Dosyalar | `src/attention.py`, `src/multi_head_attention.py`, `tests/`, deney 00, 01, 03 | `src/softmax_analysis.py`, `src/visualization.py`, `tests/test_softmax_analysis.py`, deney 02, 04 |
| Teori | `docs/teori_kisi1.md` | `docs/teori_kisi2.md` |

## Sınırlılıklar

Ağırlık matrisleri rastgeledir, eğitim yapılmamıştır. Isı haritalarındaki
örüntüler ve kafalar arasındaki farklar yalnızca rastgele izdüşümlerden
kaynaklanır; öğrenilmiş dilsel ilişkileri göstermez.

## Kaynak

Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.
