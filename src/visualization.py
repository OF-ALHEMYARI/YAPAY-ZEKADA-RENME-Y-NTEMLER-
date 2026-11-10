"""Deney 02, 04 ve 05 için ortak çizim yardımcıları.

Tüm şekiller vektörel PDF olarak results/figures/ klasörüne yazılır.
"""

import os

import matplotlib

matplotlib.use("Agg")   # pencere açmadan dosyaya çiz
import matplotlib.pyplot as plt  # noqa: E402

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEKIL_KLASORU = os.path.join(KOK, "results", "figures")


def sekil_kaydet(fig, dosya_adi):
    """Şekli results/figures/<dosya_adi> olarak vektörel PDF biçiminde kaydeder."""
    os.makedirs(SEKIL_KLASORU, exist_ok=True)
    # CreationDate=None: PDF her çalıştırmada bayt bayt aynı olur (tekrarlanabilirlik)
    fig.savefig(os.path.join(SEKIL_KLASORU, dosya_adi), metadata={"CreationDate": None})
    plt.close(fig)
    print("Sekil kaydedildi: results/figures/" + dosya_adi)


def isi_haritasi(ax, A, baslik, etiketler=None):
    """Dikkat matrisi A'yı (T, T) [0, 1] renk ölçeğiyle çizer; imshow nesnesini döndürür."""
    T = A.shape[0]
    im = ax.imshow(A, cmap="viridis", vmin=0.0, vmax=1.0)
    etiketler = etiketler or [str(i + 1) for i in range(T)]
    ax.set_xticks(range(T))
    ax.set_xticklabels(etiketler)
    ax.set_yticks(range(T))
    ax.set_yticklabels(etiketler)
    ax.set_xlabel("Anahtar (Key) token'ı")
    ax.set_title(baslik)
    return im
