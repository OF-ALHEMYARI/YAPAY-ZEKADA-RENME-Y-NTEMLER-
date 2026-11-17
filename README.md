# Self-Attention ve Transformer: Sıfırdan NumPy Uygulaması

ISE 521 (Yapay Zekâda Öğrenme Yöntemleri) dönem projesi, Konu #17.

Öz-dikkat (self-attention) mekanizması, hazır bir dikkat katmanı veya derin
öğrenme kütüphanesi kullanılmadan yalnızca NumPy ile uygulanmıştır:

$$A = \mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right), \qquad O = AV$$

Tüm uygulama, doğrulama kontrolleri ve deneyler tek bir notebook'tadır:
**`ISE521_Konu17_SelfAttention.ipynb`**. Notebook kendi kendine yeterlidir;
harici veri, internet bağlantısı veya ek modül gerektirmez.

## Çalıştırma

**Google Colab:** notebook'u yükleyin ve *Runtime → Run all* seçin.

**Yerel (Python 3.11 ile test edilmiştir):**

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Tüm hesaplamalar ve şekiller tek komutla yeniden üretilir:

```
jupyter nbconvert --to notebook --execute --inplace ISE521_Konu17_SelfAttention.ipynb
```

Rastgelelik tohumları sabittir; sayılar ve şekiller her çalıştırmada birebir
aynıdır. Şekiller `figures/` klasörüne 300 DPI PNG olarak yazılır.

## Notebook içeriği

| Bölüm | İçerik |
|---|---|
| 01 | Neden self-attention: RNN/LSTM'den Transformer'a |
| 02 | Ortam ve boyutlar (`T = 6`, `d_model = 8`, `d_k = d_v = 4`) |
| 03 | Q, K, V izdüşümleri ve bir elemanın elle doğrulanması |
| 04 | Dikkat skorları `QKᵀ` ve transpozun gerekçesi |
| 05 | `√d_k` ile ölçekleme |
| 06 | Kararlı softmax, satır satır |
| 07 | Dikkat çıktısı `O = AV` |
| 08 | Tek başlı dikkat ısı haritası |
| 09 | `√d_k` gerekçesinin istatistiksel testi |
| 10 | Softmax Jacobian'ı, doygunluk ve sonlu farklarla doğrulama |
| 11 | Ölçekli / ölçeksiz gradyan doygunluğu deneyi ve ısı haritaları |
| 12 | Çok başlı dikkat: iki head, Concat, `W_O` ve alt uzay matematiği |
| 13 | `O(T²)` zaman ve bellek karmaşıklığı |
| 14 | Otomatik kontroller (19 kontrol) |
| 15 | Savunma soruları, sınırlılıklar ve kaynaklar |

## Örnek boyutlar

| Matris | Şekil | Açıklama |
|---|---|---|
| $X$ | (6, 8) | 6 token, gömme boyutu 8 |
| $W_Q, W_K, W_V$ | (8, 4) | izdüşüm ağırlıkları |
| $Q, K, V$ | (6, 4) | sorgu, anahtar, değer |
| $S = QK^{\top}/\sqrt{d_k}$ | (6, 6) | skorlar |
| $A$ | (6, 6) | dikkat ağırlıkları, her satırın toplamı 1 |
| $O = AV$ | (6, 4) | çıktı |

## Temel bulgular

| Deney | Bulgu |
|---|---|
| `√d_k` gerekçesi | Ölçeksiz skorların standart sapması $\sqrt{d_k}$'yı izler (1.97, 4.07, 8.03, 16.26; teori 2, 4, 8, 16); ölçekli skorlarınki ≈ 1 |
| Gradyan doygunluğu | Ortalama softmax Jacobian izi, ölçeksiz durumda $d_k$ = 4 → 256 için 0.548 → 0.092'ye düşer; ölçekli durumda ≈ 0.71'de sabit kalır |
| Jacobian doğrulaması | Analitik $J = \mathrm{diag}(p) - pp^{\top}$ ile sonlu fark türevi arasındaki en büyük fark ≈ $4 \times 10^{-11}$ |
| Çok başlı dikkat | $Q_{wide}K_{wide}^{\top} = Q_1K_1^{\top} + Q_2K_2^{\top}$; ancak iki ayrı softmax, tek geniş head ile aynı dağılımı vermez |
| Karmaşıklık | Skor sayısı $T^2$ ile büyür; token sayısı iki katına çıkınca skor sayısı dört katına çıkar |

## Çalışma süreci

Proje önce ayrı Python modülleri, birim testleri ve deney betikleri
(`src/`, `tests/`, `experiments/`) olarak geliştirildi. Ardından her adımın tek
bir ortamda okunabilmesi, açıklanabilmesi ve yeniden üretilebilmesi için tüm
mekanizma, testler ve deneyler bu notebook'ta birleştirildi. Önceki modüler
sürüm git geçmişinde korunmaktadır (commit `e8ffea1`).

| | Kişi 1 | Kişi 2 |
|---|---|---|
| Bölümler | Q/K/V, skorlar, kararlı softmax, dikkat çıktısı, ısı haritası, çok başlı dikkat | `√d_k` gerekçesi, Jacobian ve doygunluk deneyi, karmaşıklık analizi |

## Sınırlılıklar

Girdi gömmeleri ve ağırlık matrisleri rastgeledir, eğitim yapılmamıştır. Isı
haritalarındaki örüntüler ve başlıklar arasındaki farklar yalnızca rastgele
izdüşümlerden kaynaklanır; öğrenilmiş dilsel ilişkileri göstermez. Konumsal
kodlama, maske, LayerNorm ve ileri beslemeli alt katman yoktur; çalışma tam bir
Transformer değil, dikkat mekanizmasının kendisini inceler.

## Kaynaklar

- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.
- Bahdanau, D., Cho, K., Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate*. ICLR.
- Hochreiter, S., Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
- Dao, T. et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS.
