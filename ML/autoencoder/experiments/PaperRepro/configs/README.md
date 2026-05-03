# Frozen Configs

Bu klasor, `PaperRepro` icin dondurulmus config dosyalarini tutar.

Kurallar:
- karar dokumanlariyla celisen gecici config tutulmaz
- yeni deney varyanti acilacaksa yeni dosya acilir, mevcut frozen dosya sessizce degistirilmez
- run sirasinda kullanilan config snapshot run klasoru altina ayrica kopyalanir

Ilk ana config:
- `frozen_paper_repro_v1.yaml`
