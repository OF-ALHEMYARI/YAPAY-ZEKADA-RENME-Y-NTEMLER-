# تقرير عمل الشخص الأول ودليل بداية الشخص الثاني

المقرر: ISE 521، الموضوع #17 (Self-Attention & Transformers).
الفرع: `kisi1-attention`.

---

## الجزء الأول: ما أنجزه الشخص الأول

### 1. الكود

| الملف | ما فيه |
|---|---|
| `src/attention.py` | `stable_softmax`: Softmax مستقر عدديًا. `scaled_dot_product_attention`: الانتباه `A = softmax(QKᵀ/√d_k)` و`O = AV` |
| `src/multi_head_attention.py` | `multi_head_attention`: تقسيم إلى رؤوس، انتباه في كل فضاء جزئي، دمج، ثم ضرب في `W_O` |
| `tests/test_attention.py` | 6 اختبارات: الأشكال، مجموع الصفوف = 1، لا `nan` مع قيم ضخمة، المعامل `scale`، Softmax مع قيم كبيرة، ثبات Softmax عند إضافة ثابت |
| `tests/test_multi_head.py` | 3 اختبارات: الأشكال وعدد الرؤوس، مجموع صفوف كل رأس = 1، رأس واحد مع `W_O = I` يساوي الانتباه العادي |
| `experiments/00_self_attention_semasi.py` | يرسم مخطط تدفق الحساب |
| `experiments/01_basic_attention.py` | `X` ثم Q/K/V ثم `A` ثم خريطة الحرارة |
| `experiments/03_multi_head.py` | رأسان وخريطة حرارة لكل رأس |
| `run_all.py` | يشغّل كل ملفات `experiments/` المرقّمة بالترتيب |

### 2. النتائج (تتكرر حرفيًا، البذرة 42)

| الشيء | النتيجة |
|---|---|
| الاختبارات | 9 من 9 ناجحة |
| `X` | `(6, 8)` |
| `Q`، `K`، `V` | `(6, 4)` |
| `A` | `(6, 6)`، مجموع كل صف = 1 |
| ناتج الرأس الواحد | `(6, 4)` |
| ناتج الرأسين | `(6, 8)` |

### 3. الرسوم (PDF متجهي، في `results/figures/`)

| الملف | محتواه |
|---|---|
| `self_attention_semasi.pdf` | مخطط Self-Attention |
| `dikkat_isi_haritasi.pdf` | خريطة حرارة الرأس الواحد |
| `cok_basli_dikkat.pdf` | خريطتا حرارة الرأسين |

### 4. النظرية

الملف `docs/teori_kisi1.md` (بالتركية، والمعادلات بـ LaTeX) فيه مسودة نصيب
الشخص الأول من قسم Mathematical Formulation: أبعاد Q/K/V واشتقاقها، Softmax
والاستقرار العددي مع الإثبات، Multi-Head ودور `W_O`، سبب `√d_k`، وحدود التجارب.

### 5. المشترك الذي جُهّز مبكرًا

- `README.md`: الوصف، التثبيت، أمر التشغيل الواحد، بنية المشروع، الواجهة المشتركة.
- `run_all.py`: يعمل الآن مع تجارب الشخص الأول، ويلتقط تجارب الشخص الثاني تلقائيًا.

### 6. كيف تتحقق بنفسك

من مجلد المشروع والبيئة `venv` مفعّلة:

```
python -m pytest tests -v
```

المتوقع: `9 passed`.

```
python run_all.py
```

المتوقع في آخره: `Tum deneyler basariyla tamamlandi: 3 dosya`.

### 7. ما بقي على الشخص الأول

- فهم كل سطر في الكود والقدرة على شرحه (شرط الدليل للدفاع).
- مراجعة مسودة النظرية وإعادة صياغتها بكلماته.
- مراجعة كود الشخص الثاني حين يجهز.
- نصيبه من التقرير والسيمينار والـ Live Demo.

---

## الجزء الثاني: من أين يبدأ الشخص الثاني

### ما تقوله له باختصار

> "الجزء الأساسي جاهز: دوال الانتباه مكتوبة ومختبرة. أنت لا تعيد كتابتها،
> أنت **تستدعيها** وتبني فوقها ثلاث تحليلات: تجربة الـ Scaling، و Entropy
> و Jacobian للـ Softmax، وتعقيد O(T²). عملك في أربعة ملفات جديدة، ولا تحتاج
> إلى تعديل أي ملف من ملفاتي."

### الخطوة 1: الوصول إلى المستودع

1. يقبل دعوة GitHub (تصله على بريده، أو من `github.com/notifications`).
2. ينسخ المستودع إلى جهازه:

```
git clone https://github.com/OF-ALHEMYARI/YAPAY-ZEKADA-RENME-Y-NTEMLER-.git
```

3. يدخل المجلد وينتقل إلى فرع الشخص الأول (ليحصل على الكود الجاهز):

```
git switch kisi1-attention
```

4. ينشئ فرعه الخاص منه:

```
git switch -c kisi2-analiz
```

### الخطوة 2: البيئة

```
python -m venv venv
```

```
venv\Scripts\activate
```

```
pip install -r requirements.txt
```

ثم يتأكد أن كل شيء يعمل عنده قبل أن يكتب أي شيء:

```
python -m pytest tests -v
```

المتوقع: `9 passed`. إن لم يحصل عليها فالمشكلة في بيئته، وليست في عمله.

### الخطوة 3: الواجهة التي يستعملها (لا تتغير)

```python
from src.attention import stable_softmax, scaled_dot_product_attention
from src.multi_head_attention import multi_head_attention

stable_softmax(skorlar)
# skorlar: (..., T)  ->  نفس الشكل، مجموع كل صف 1

scaled_dot_product_attention(Q, K, V, scale=True)
# Q, K: (T, d_k)   V: (T, d_v)
# يرجع: (cikti (T, d_v), dikkat_agirliklari (T, T))
# scale=False  =>  بدون القسمة على sqrt(d_k)

multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads)
# يرجع: (cikti (T, d_model), قائمة مصفوفات انتباه، كل واحدة (T, T))
```

ثلاثة أشياء ثابتة: اسم الدالة، والمعامل `scale`، وترتيب ما ترجعه.
إن احتاج تغييرًا في هذه الدوال يطلبه من الشخص الأول ولا يعدّلها بنفسه.

### الخطوة 4: ملفاته الأربعة

| الملف | المطلوب فيه |
|---|---|
| `src/softmax_analysis.py` | دوال التحليل: Entropy لكل صف من `A`، و Jacobian الـ Softmax (`∂p_i/∂z_j = p_i(δ_ij − p_j)`) |
| `src/visualization.py` | دوال رسم مشتركة لتجاربه (حفظ PDF في `results/figures/`) |
| `experiments/02_scaling.py` | تجربة الـ Scaling: مقارنة `scale=True` مع `scale=False` عند قيم مختلفة لـ `d_k` |
| `experiments/04_complexity.py` | قياس الزمن والذاكرة مع زيادة `T`، وإظهار النمو O(T²) |

### الخطوة 5: أول تجربة يكتبها (نقطة البداية الفعلية)

أسهل بداية هي تجربة الـ Scaling، لأن كل ما تحتاجه موجود:

```python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.attention import scaled_dot_product_attention

rng = np.random.default_rng(42)
T = 6
for d_k in [4, 16, 64, 256, 1024]:
    Q = rng.normal(size=(T, d_k))
    K = rng.normal(size=(T, d_k))
    V = rng.normal(size=(T, d_k))
    _, A_olcekli = scaled_dot_product_attention(Q, K, V, scale=True)
    _, A_olceksiz = scaled_dot_product_attention(Q, K, V, scale=False)
    print(d_k, "| olcekli max:", A_olcekli.max(axis=1).mean().round(3),
          "| olceksiz max:", A_olceksiz.max(axis=1).mean().round(3))
```

ما يجب أن يلاحظه: مع زيادة `d_k`، القيمة العظمى في كل صف بدون scaling تقترب
من 1 (Softmax يتشبع)، ومع scaling تبقى معتدلة. هذه هي الظاهرة التي يشرحها
ثم يقيسها بالـ Entropy والـ Jacobian.

### الخطوة 6: قواعد حتى لا يتعارض العمل

- يسمّي تجاربه `02_...py` و`04_...py` داخل `experiments/`، فيلتقطها
  `run_all.py` تلقائيًا دون تعديل.
- بذرة عشوائية ثابتة في كل ملف (`np.random.default_rng(42)`).
- الرسوم PDF متجهي في `results/figures/`.
- تعليقات الكود بالتركية وتشير إلى المعادلة المقابلة.
- إن أضاف مكتبة جديدة يحدّث `requirements.txt`.
- يرفع على فرعه `kisi2-analiz` فقط، ثم يُدمج الفرعان في `main` بعد المراجعة.

### الخطوة 7: نصيبه من النظرية

- مشتقة Softmax (Jacobian) ولماذا تقترب من الصفر عند التشبع.
- سبب القسمة على `√d_k` (الاشتقاق موجود مختصرًا في `docs/teori_kisi1.md`، القسم 3.1).
- تعقيد O(T²) في الزمن والذاكرة وأثره على العتاد.

### المشترك بينكما بعد ذلك

المراجع (8 إلى 12)، التقرير (4 إلى 6 صفحات)، السيمينار، الـ Live Demo،
ومراجعة كل منكما لكود الآخر.
