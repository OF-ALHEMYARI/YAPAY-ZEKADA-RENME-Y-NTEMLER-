# Teori Notları (Kişi 1): Rapor için taslak metin

Bu dosya, raporun *Mathematical Formulation* bölümünde Kişi 1'e düşen kısmın
taslağıdır. Denklemler LaTeX biçimindedir ve rapor şablonuna doğrudan taşınabilir.
Denklem numaraları Vaswani et al. (2017) ile uyumludur.

## 1. Girdi ve Q, K, V izdüşümleri

Bir dizi $T$ token içerir. Her token $d_{model}$ boyutlu bir gömme vektörüyle
temsil edilir ve bu vektörler bir matrisin satırlarını oluşturur:

$$X \in \mathbb{R}^{T \times d_{model}}$$

Öz-dikkatte aynı $X$, üç ayrı ağırlık matrisiyle üç farklı role izdüşürülür:

$$Q = X W_Q, \qquad K = X W_K, \qquad V = X W_V$$

$$W_Q, W_K \in \mathbb{R}^{d_{model} \times d_k}, \qquad W_V \in \mathbb{R}^{d_{model} \times d_v}$$

- **Sorgu ($Q$):** token'ın "ne aradığını" temsil eder.
- **Anahtar ($K$):** token'ın "ne sunduğunu" temsil eder.
- **Değer ($V$):** token seçildiğinde aktarılacak içeriği temsil eder.

Üç matris ayrıdır, çünkü üç rol farklıdır. $W_Q = W_K$ olsaydı skor matrisi
$S = QQ^{\top}$ simetrik olurdu ($S_{ij} = S_{ji}$) ve her token en çok kendisine
benzerdi; oysa "A, B'ye dikkat eder" ilişkisi "B, A'ya dikkat eder" ilişkisiyle
aynı olmak zorunda değildir.

## 2. Boyutların türetilmesi

Matris çarpımı $(m \times n)(n \times p) \to (m \times p)$ kuralına uyar: ortadaki
iki boyut eşit olmalıdır ve sonuçta kaybolur. Projedeki örnek değerler
$T = 6$, $d_{model} = 8$, $d_k = d_v = 4$ için:

| Adım | İşlem | Boyutlar | Sonuç |
|---|---|---|---|
| Girdi | $X$ | | $(6, 8)$ |
| İzdüşüm | $Q = X W_Q$ | $(6, 8)(8, 4)$ | $(6, 4)$ |
| İzdüşüm | $K = X W_K$ | $(6, 8)(8, 4)$ | $(6, 4)$ |
| İzdüşüm | $V = X W_V$ | $(6, 8)(8, 4)$ | $(6, 4)$ |
| Skorlar | $S = Q K^{\top} / \sqrt{d_k}$ | $(6, 4)(4, 6)$ | $(6, 6)$ |
| Ağırlıklar | $A = \mathrm{softmax}(S)$ | satır bazlı | $(6, 6)$ |
| Çıktı | $O = A V$ | $(6, 6)(6, 4)$ | $(6, 4)$ |

$K^{\top}$ zorunludur: $Q K$ çarpımı $(6, 4)(6, 4)$ olurdu ve ortadaki boyutlar
($4 \neq 6$) uyuşmazdı. Devrik alındığında $S_{ij} = q_i \cdot k_j$ olur; yani
$S_{ij}$, $i$. token'ın sorgusu ile $j$. token'ın anahtarının nokta çarpımıdır ve
"$i$. token $j$. token'a ne kadar dikkat etmeli" sorusunun ham skorudur.

Çıktının $(T, T)$ değil $(T, d_v)$ olmasının nedeni son çarpımdır: $O$'nun her
satırı, $V$'nin satırlarının $A$'daki ağırlıklarla alınmış ağırlıklı ortalamasıdır,

$$o_i = \sum_{j=1}^{T} A_{ij}\, v_j .$$

## 3. Ölçekli nokta çarpımı dikkat

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V \tag{1}$$

Koddaki karşılığı `src/attention.py` içindeki `scaled_dot_product_attention`
fonksiyonudur.

### 3.1 Neden $\sqrt{d_k}$ ile bölünür?

$q$ ve $k$ vektörlerinin bileşenleri bağımsız, ortalaması 0 ve varyansı 1 ise

$$q \cdot k = \sum_{m=1}^{d_k} q_m k_m, \qquad \mathrm{Var}(q \cdot k) = \sum_{m=1}^{d_k} \mathrm{Var}(q_m k_m) = d_k .$$

Yani skorların standart sapması $\sqrt{d_k}$ ile büyür. Büyük skorlar softmax'ı
doygunluğa iter: bir bileşen 1'e, diğerleri 0'a yaklaşır ve türevler sıfıra
yakınlaşır. $\sqrt{d_k}$ ile bölmek varyansı yeniden 1 yapar:

$$\mathrm{Var}\!\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1 .$$

(Sayısal deney Kişi 2'nin bölümündedir; `scale=False` parametresi bunun içindir.)

## 4. Softmax ve sayısal kararlılık

Softmax, bir skor satırını olasılık dağılımına çevirir:

$$p_i = \frac{e^{z_i}}{\sum_{j=1}^{T} e^{z_j}}, \qquad p_i > 0, \qquad \sum_{i=1}^{T} p_i = 1 .$$

$A$'nın her satırının toplamının 1 olması buradan gelir: payların toplamı paydaya
eşittir.

**Sorun.** Kayan noktalı sayılarda $e^{z}$, $z \approx 710$ üzerinde taşar
(`inf`), ve `inf / inf = nan` olur.

**Çözüm.** Her satırdan o satırın en büyük değeri $m = \max_j z_j$ çıkarılır:

$$p_i = \frac{e^{z_i - m}}{\sum_{j} e^{z_j - m}} .$$

**Sonucun değişmediğinin ispatı.** $e^{z_i - m} = e^{z_i} e^{-m}$ olduğundan

$$\frac{e^{z_i - m}}{\sum_j e^{z_j - m}} = \frac{e^{-m}\, e^{z_i}}{e^{-m} \sum_j e^{z_j}} = \frac{e^{z_i}}{\sum_j e^{z_j}} .$$

$e^{-m}$ çarpanı pay ve paydada sadeleşir. Genel olarak softmax, girdiye sabit
eklemeye karşı değişmezdir: $\mathrm{softmax}(z + c) = \mathrm{softmax}(z)$.

**Neden kararlı?** Çıkarmadan sonra en büyük üs $0$'dır, yani en büyük terim
$e^0 = 1$ olur; diğer tüm üsler negatiftir ve $e^{(\cdot)} \in (0, 1]$ aralığında
kalır. Taşma olamaz ve payda en az 1 olduğu için sıfıra bölme de olamaz.

Kodda `axis=-1` işlemi satır bazlı yapar; `keepdims=True` ise en büyük değerlerin
şeklini $(T, 1)$ olarak korur, böylece çıkarma ve bölme her satıra kendi değeriyle
uygulanır. Bu iki özellik `tests/test_attention.py` içinde sınanır
(`test_softmax_buyuk_girdi`, `test_softmax_kaydirma_degismezligi`).

## 5. Çok başlı dikkat (Multi-Head Attention)

Tek bir dikkat matrisi, her token için tek bir ağırlık dağılımı üretir. Çok başlı
dikkat, $h$ ayrı dikkat hesabını paralel yürütür; her biri farklı bir alt uzayda
çalışır ve farklı bir ilişki türüne odaklanabilir:

$$\mathrm{MultiHead}(X) = \mathrm{Concat}(\mathrm{kafa}_1, \ldots, \mathrm{kafa}_h)\, W_O \tag{4}$$

$$\mathrm{kafa}_i = \mathrm{Attention}(Q_i, K_i, V_i), \qquad d_k = d_v = \frac{d_{model}}{h} \tag{5}$$

Uygulamada $Q = X W_Q$, $K = X W_K$, $V = X W_V$ matrisleri $(T, d_{model})$
boyutunda hesaplanır ve sütunları $h$ eşit parçaya bölünür; $i$. kafa
$[(i-1) d_k,\; i\, d_k)$ sütunlarını kullanır. Bu, her kafaya ayrı
$(d_{model} \times d_k)$ izdüşüm matrisi vermekle eşdeğerdir.

Örnek değerler $T = 6$, $d_{model} = 8$, $h = 2$ için:

| Adım | Şekil |
|---|---|
| $Q, K, V$ (tümü) | $(6, 8)$ |
| Her kafanın $Q_i, K_i, V_i$ parçası | $(6, 4)$ |
| Her kafanın dikkat matrisi $A_i$ | $(6, 6)$ |
| Her kafanın çıktısı | $(6, 4)$ |
| Birleştirme (Concat) | $(6, 8)$ |
| $W_O$ ile çarpım | $(6, 8)(8, 8) \to (6, 8)$ |

**Neden alt uzaylar?** $d_k = d_{model}/h$ seçildiği için toplam hesap maliyeti
tek başlı dikkatle yaklaşık aynıdır, ama model tek bir ortalama dağılım yerine
$h$ farklı dağılım öğrenebilir.

**$W_O$'nun rolü.** Birleştirme yalnızca kafa çıktılarını yan yana koyar; kafalar
birbirinden habersizdir. $W_O \in \mathbb{R}^{d_{model} \times d_{model}}$ bu
parçaları doğrusal olarak karıştırır ve çıktıyı yeniden $d_{model}$ boyutlu ortak
uzaya taşır, böylece sonraki katman tüm kafaların bilgisini birlikte kullanabilir.

**Doğrulama.** $h = 1$ ve $W_O = I$ alındığında çok başlı dikkat, tek başlı
dikkate indirgenir. Bu özellik `tests/test_multi_head.py` içinde sınanır
(`test_tek_kafa_ozdes_matris_ile_tek_dikkate_esit`).

## 6. Deneylerin gösterdiği ve göstermediği

Deneyler, uygulamanın matematiksel olarak doğru olduğunu gösterir: boyutlar
beklendiği gibidir, $A$'nın her satırının toplamı 1'dir ve çok büyük girdilerde
`nan` veya `inf` oluşmaz.

**Sınırlılık.** Ağırlık matrisleri rastgeledir ve eğitim yapılmamıştır. Isı
haritalarındaki örüntüler ile kafalar arasındaki farklar yalnızca farklı rastgele
izdüşümlerden kaynaklanır; öğrenilmiş dilsel ilişkileri göstermez.

## Şekiller

| Dosya | İçerik |
|---|---|
| `results/figures/self_attention_semasi.pdf` | hesaplama akışı şeması |
| `results/figures/dikkat_isi_haritasi.pdf` | tek başlı dikkat matrisi $A$ |
| `results/figures/cok_basli_dikkat.pdf` | iki kafanın dikkat matrisleri |
