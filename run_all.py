"""Tüm deneyleri sırayla çalıştırır.

Çalıştırma: python run_all.py

experiments/ klasöründeki numaralı tüm dosyaları (00_..., 01_..., ...) ad
sırasına göre çalıştırır. Yeni bir deney eklemek için dosyayı experiments/
klasörüne koymak yeterlidir; bu dosyayı değiştirmek gerekmez.
"""

import glob
import os
import subprocess
import sys

KOK = os.path.dirname(os.path.abspath(__file__))


def main():
    deneyler = sorted(glob.glob(os.path.join(KOK, "experiments", "[0-9]*.py")))
    if not deneyler:
        print("Hic deney bulunamadi.")
        return 1

    for yol in deneyler:
        ad = os.path.basename(yol)
        print("=" * 60)
        print("Calistiriliyor:", ad)
        print("=" * 60, flush=True)
        # Aynı Python yorumlayıcısı (venv) ile, proje kökünden çalıştır
        sonuc = subprocess.run([sys.executable, yol], cwd=KOK)
        if sonuc.returncode != 0:
            print("HATA:", ad, "basarisiz oldu.")
            return sonuc.returncode

    print("=" * 60)
    print("Tum deneyler basariyla tamamlandi:", len(deneyler), "dosya")
    print("Sekiller: results/figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
