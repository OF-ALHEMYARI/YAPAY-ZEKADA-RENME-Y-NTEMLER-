# Teori Notları (Kişi 2): Rapor için taslak metin

Bu dosya, raporda Kişi 2'ye düşen kısımların taslağıdır: tarihsel arka plan
(ortak bölümün taslağı), $\sqrt{d_k}$ ölçeklemesi ve softmax doygunluğu, geri
yayılım kuralları ve $\mathcal{O}(T^2)$ karmaşıklığı. Denklemler LaTeX biçimindedir.
Deney sayıları `python run_all.py` çıktısından alınmıştır (tohum 42).

## 1. Tarihsel arka plan: RNN'in bilgi darboğazı

Yinelemeli ağlar (RNN) diziyi sırayla işler ve tüm geçmişi sabit boyutlu tek bir
gizli durumda taşır:

$$h_t = \tanh(W_h h_{t-1} + W_x x_t)$$

Bu tasarımın üç sınırı vardır:

1. **Bilgi darboğazı.** Seq2Seq çeviri modellerinde [Sutskever 2014, Cho 2014]
   kodlayıcı bütün cümleyi son gizli durum $h_T$'ye sıkıştırır; kod çözücü yalnızca
   bu vektörü görür. Cümle uzadıkça sabit boyutlu vektöre sığmayan bilgi kaybolur.
2. **Kaybolan gradyan.** $x_1$'in $h_T$ üzerindeki etkisi
   $\partial h_T / \partial h_1 = \prod_{t=2}^{T} \partial h_t / \partial h_{t-1}$
   çarpımından geçer; çarpanların normu 1'den küçükse etki üstel olarak söner
   [Bengio 1994]. LSTM [Hochreiter 1997] kapılar ve hücre durumuyla bunu azaltır,
   ama yol uzunluğu yine $\mathcal{O}(T)$'dir.
3. **Sıralı hesap.** $h_t$, $h_{t-1}$ olmadan hesaplanamaz; $T$ adım
   paralelleştirilemez ve GPU verimsiz kullanılır.

**Kronoloji.**

| Yıl | Çalışma | Katkı |
|---|---|---|
| 1997 | Hochreiter & Schmidhuber | LSTM: kapılarla uzun bağımlılık |
| 2014 | Sutskever et al.; Cho et al. | Seq2Seq kodlayıcı-kod çözücü, sabit vektör darboğazı |
| 2014 | Bahdanau et al. | Toplamsal dikkat: kod çözücü her adımda tüm kodlayıcı durumlarına bakar |
| 2015 | Luong et al. | Çarpımsal (nokta çarpımı) dikkat |
| 2017 | Vaswani et al. | Transformer: RNN tamamen kaldırılır, yalnızca dikkat |
| 2019 | Devlin et al. (BERT) | Çift yönlü ön eğitimli kodlayıcı |
| 2021 | Dosovitskiy et al. (ViT) | Görüntü yamalarına öz-dikkat |
| 2022 | Dao et al. (FlashAttention) | $T^2$ belleği yazmadan tam dikkat |

Öz-dikkatte her token tüm token'lara tek adımda bakar; Vaswani et al. (2017,
Tablo 1) karşılaştırması:

| Katman | Katman başına karmaşıklık | Sıralı işlem | En uzun yol |
|---|---|---|---|
| RNN | $\mathcal{O}(T d^2)$ | $\mathcal{O}(T)$ | $\mathcal{O}(T)$ |
| Öz-dikkat | $\mathcal{O}(T^2 d)$ | $\mathcal{O}(1)$ | $\mathcal{O}(1)$ |

Bedel, $T^2$ büyüklüğündeki dikkat matrisidir (Bölüm 5).

## 2. $\sqrt{d_k}$ ölçeklemesi

$q, k \in \mathbb{R}^{d_k}$ bileşenleri bağımsız, $\mathbb{E}[q_m] = \mathbb{E}[k_m] = 0$
ve $\mathrm{Var}(q_m) = \mathrm{Var}(k_m) = 1$ olsun. Bağımsızlıktan
$\mathbb{E}[q_m k_m] = 0$ ve

$$\mathrm{Var}(q_m k_m) = \mathbb{E}[q_m^2]\,\mathbb{E}[k_m^2] - 0 = 1 .$$

Terimler bağımsız olduğu için varyanslar toplanır:

$$\mathrm{Var}(q \cdot k) = \sum_{m=1}^{d_k} \mathrm{Var}(q_m k_m) = d_k,
\qquad \mathrm{Var}\!\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = 1 .$$

$d_k$ ile bölmek varyansı $1/d_k$'ya indirirdi; skorlar sıfıra yaklaşır ve softmax
düzgün dağılıma döner. $\sqrt{d_k}$, varyansı tam olarak 1'de tutan çarpandır.

**Deney 02 doğrulaması** (her $d_k$ için 200 rastgele dizi, ortalama skor standart sapması):

| $d_k$ | 4 | 16 | 64 | 256 | 1024 |
|---|---|---|---|---|---|
| ölçeksiz $\mathrm{std}(S)$ | 1.90 | 3.93 | 7.84 | 15.43 | 31.26 |
| $\sqrt{d_k}$ | 2 | 4 | 8 | 16 | 32 |
| ölçekli $\mathrm{std}(S)$ | 0.95 | 0.98 | 0.98 | 0.96 | 0.98 |

## 3. Softmax Jacobian'ı ve doygunluk

$p_i = e^{z_i} / \Sigma$, $\Sigma = \sum_k e^{z_k}$ olsun. Bölüm kuralıyla:

$$i = j: \quad \frac{\partial p_i}{\partial z_i} = \frac{e^{z_i}\Sigma - e^{z_i}e^{z_i}}{\Sigma^2} = p_i(1 - p_i)$$

$$i \neq j: \quad \frac{\partial p_i}{\partial z_j} = \frac{-e^{z_i}e^{z_j}}{\Sigma^2} = -p_i p_j$$

$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j), \qquad J = \mathrm{diag}(p) - pp^{\top} \tag{6}$$

**Özellikler.** $J$ simetriktir; $\sum_i p_i = 1$ sabit olduğundan her satırının
toplamı 0'dır. Köşegen terimi $p_i(1-p_i)$ en fazla $1/4$'tür ($p_i = 1/2$).

**Doygunluk.** $p \to e_k$ (one-hot) iken $p_k(1-p_k) \to 0$, $p_i \to 0$
($i \neq k$) ve tüm $p_ip_j \to 0$; dolayısıyla $J \to 0$. Kayıp $L$ için zincir
kuralı $\partial L / \partial z = J^{\top}\, \partial L / \partial p$ olduğundan,
doygun bir satırdan skorlara hiçbir gradyan geçmez.

**Entropi.** Dağılımın keskinliği $H(p) = -\sum_i p_i \log p_i$ ile ölçülür;
$0 \le H \le \log T$. $\log 6 = 1.792$ düzgün dağılım, $0$ one-hot demektir.

Koddaki karşılıkları: `src/softmax_analysis.py` (`softmax_jacobian`,
`satir_entropisi`, `jacobian_normu`). Analitik Jacobian, merkezi sonlu farklarla
hesaplanan sayısal Jacobian'la karşılaştırılarak sınanır
(`tests/test_softmax_analysis.py`, fark $< 10^{-8}$).

**Deney 02 sonuçları** ($T = 6$, her $d_k$ için 200 dizi; hücreler ölçekli / ölçeksiz):

| $d_k$ | satır maksimumu | entropi $H$ | $\|J\|_F$ | doygun satır ($\|J\|_F < 10^{-3}$) |
|---|---|---|---|---|
| 4 | 0.395 / 0.568 | 1.501 / 1.111 | 0.383 / 0.348 | %0 / %0.2 |
| 16 | 0.420 / 0.783 | 1.468 / 0.567 | 0.384 / 0.231 | %0 / %2.4 |
| 64 | 0.424 / 0.891 | 1.462 / 0.273 | 0.384 / 0.136 | %0 / %24.2 |
| 256 | 0.416 / 0.951 | 1.478 / 0.124 | 0.386 / 0.068 | %0 / %51.5 |
| 1024 | 0.417 / 0.972 | 1.471 / 0.067 | 0.387 / 0.040 | %0 / %73.5 |

**Yorum.** Ölçekli dikkatte dört ölçü de $d_k$'dan bağımsızdır. Ölçeksiz dikkatte
$d_k = 1024$ için satırların %73.5'i doygundur: bu satırlardaki sorgu token'ları
için $W_Q$ ve $W_K$'ya hiç gradyan ulaşmaz. Ortalama $\|J\|_F$ sıfıra üstel değil,
yaklaşık $1/\sqrt{d_k}$ hızında iner; bunun nedeni, en büyük iki skorun tesadüfen
birbirine yakın kaldığı az sayıdaki satırın ortalamayı taşımasıdır. Medyan satır
ise tamamen doygundur. Isı haritası (`olcekleme_isi_haritalari.pdf`) aynı durumu
gösterir: ölçeksiz $A$'nın her satırı neredeyse one-hot'tur.

## 4. Geri yayılım (gradyan kuralları)

Dikkat bir öğrenme algoritması değil, bir katmandır; kendi amaç fonksiyonu yoktur.
Parametreleri $W_Q, W_K, W_V, W_O$, görev kaybıyla öğrenilir; örneğin bir sonraki
token tahmini için çapraz entropi:

$$\mathcal{L} = -\sum_{t} \log p_\theta(y_t \mid y_{<t}) .$$

$S = c\,QK^{\top}$ ($c = 1/\sqrt{d_k}$ ya da ölçeksizde $c = 1$),
$A = \mathrm{softmax}(S)$, $O = AV$ için zincir kuralı:

| Gradyan | İfade | Boyut |
|---|---|---|
| $\partial L/\partial V$ | $A^{\top}\, \partial L/\partial O$ | $(T, d_v)$ |
| $\partial L/\partial A$ | $(\partial L/\partial O)\, V^{\top}$ | $(T, T)$ |
| $\partial L/\partial S_{i,:}$ | $J_i\, \partial L/\partial A_{i,:} = a_i \odot (g_i - a_i^{\top} g_i)$, $g_i = \partial L/\partial A_{i,:}$ | $(T,)$, her satır |
| $\partial L/\partial Q$ | $c\,(\partial L/\partial S)\, K$ | $(T, d_k)$ |
| $\partial L/\partial K$ | $c\,(\partial L/\partial S)^{\top} Q$ | $(T, d_k)$ |
| $\partial L/\partial W_Q$ | $X^{\top}\, \partial L/\partial Q$ | $(d_{model}, d_k)$ |
| $\partial L/\partial W_K$ | $X^{\top}\, \partial L/\partial K$ | $(d_{model}, d_k)$ |

$W_Q$ ve $W_K$'nın gradyanları $J$'den geçer, $W_V$'ninki geçmez. Doygunluk bu
yüzden özellikle sorgu ve anahtar izdüşümlerinin öğrenilmesini durdurur.

**Not.** Ölçeksiz durumda $c = 1$ olduğu için $\partial L/\partial Q$, ölçekli
duruma göre $\sqrt{d_k}$ kat büyük bir çarpan taşır; doymamış az sayıdaki satırda
bu, etkin öğrenme oranının büyümesi ve kararsızlık demektir. Yani ölçeksiz dikkat
iki yönden zararlıdır: satırların çoğunda gradyan sıfırdır, kalanlarda ise ölçek
kontrolsüzdür.

## 5. $\mathcal{O}(T^2)$ karmaşıklık ve donanım etkileri

| İşlem | Çarpma sayısı | Bellek |
|---|---|---|
| $Q, K, V = XW$ | $3\,T\, d_{model}\, d_k$ | $\mathcal{O}(T d)$ |
| $S = QK^{\top}$ | $T^2 d_k$ | $T^2$ |
| $O = AV$ | $T^2 d_v$ | $\mathcal{O}(T d)$ |

Toplam zaman $\mathcal{O}(T^2 d + T d^2)$, dikkat matrisi için bellek $\mathcal{O}(T^2)$'dir.

**Deney 04 doğrulaması** ($d_k = 64$, float64, 5 tekrarın en kısası):

| $T$ | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| süre (ms) | 3.65 | 16.31 | 63.90 | 249.37 |
| $A$ belleği (MB) | 2.1 | 8.4 | 33.6 | 134.2 |

log-log eğim: süre için $\approx 2.0$ ($T \geq 512$), bellek için tam $2.00$.
Mutlak süreler makineye ve çalıştırmaya göre değişir; eğim değişmez. Küçük $T$'de
sabit maliyetler (fonksiyon çağrısı, bellek ayırma) baskın olduğu için eğim
$T \geq 512$ üzerinden hesaplanır.

**Donanım etkileri.**

1. **Bellek kapasitesi.** $T = 131072$ token'lık bir bağlamda tek bir dikkat matrisi
   float16 ile $T^2 \cdot 2 \approx 34.4$ GB tutar; bu tek bir kafa ve tek bir
   katman içindir ve çoğu GPU'nun belleğini aşar.
2. **Bellek bant genişliği.** GPU'da matris çarpımı hızlıdır; asıl darboğaz $T \times T$
   matrisinin yavaş HBM belleğe yazılıp tekrar okunmasıdır. FlashAttention [Dao 2022]
   hesabı hızlı SRAM'de bloklar hâlinde yapar ve softmax'ı çevrimiçi (online)
   normalleştirir; $A$ hiç tam olarak yazılmaz. Zaman yine $\mathcal{O}(T^2)$'dir ve
   sonuç tamdır, ama ek bellek $\mathcal{O}(T)$'ye iner.
3. **Yaklaşık yöntemler.** Seyrek dikkat [Child 2019] her token'ın yalnızca bir alt
   kümeye bakmasını sağlar; doğrusal dikkat [Katharopoulos 2020] softmax yerine
   çekirdek ayrışımıyla $\mathcal{O}(T)$ zamana iner, ama tam softmax dikkatinin
   yaklaşığıdır.
4. **Çıkarım (inference).** Otoregresif üretimde önceki $K, V$ saklanır (KV önbelleği);
   önbellek $T$ ile doğrusal büyür ve her yeni token $\mathcal{O}(T)$ işlem gerektirir.

## 6. Sınırlılıklar

- Ağırlıklar rastgeledir ve eğitim yoktur. Doygunluk softmax'ın matematiksel bir
  özelliğidir ve yalnızca skorların büyüklüğüne bağlıdır; bu yüzden eğitim
  olmadan gösterilebilir ve eğitimin başında, ağırlıklar rastgeleyken en önemlidir.
- Deney 02'de $Q, K$ doğrudan $\mathcal{N}(0, 1)$'den çekilir; bu, Bölüm 2'deki
  varsayımın birebir karşılığıdır. Gerçek modellerde LayerNorm ve uygun başlatma bu
  varsayımı yaklaşık olarak sağlar.
- Zaman ölçümleri CPU üzerinde NumPy (çok iş parçacıklı BLAS) ile yapılmıştır;
  GPU davranışı (bant genişliği, SRAM) yalnızca tartışılmış, ölçülmemiştir.

## Şekiller

| Dosya | İçerik |
|---|---|
| `results/figures/olcekleme_doygunluk.pdf` | satır maksimumu, entropi, $\|J\|_F$ ve doygun satır oranının $d_k$ ile değişimi |
| `results/figures/olcekleme_isi_haritalari.pdf` | $d_k = 1024$ için ölçekli ve ölçeksiz $A$ |
| `results/figures/karmasiklik.pdf` | süre ve belleğin $T$ ile log-log değişimi, eğim 2 referansı |
