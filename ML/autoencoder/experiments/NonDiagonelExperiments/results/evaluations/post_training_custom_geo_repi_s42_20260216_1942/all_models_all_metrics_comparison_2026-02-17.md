# Tum Modeller - Tum Metrikler Karsilastirma

- Tarih: `2026-02-17`
- Kaynaklar:
  - `ML/autoencoder/experiments/NonDiagonel/results/evaluations/post_training_custom_geo_repi_s42_20260216_1942/model_family_eval_small_phase_20260217_1216/metrics_aggregate.json`
  - `ML/autoencoder/experiments/NonDiagonel/results/evaluations/post_training_custom_geo_repi_s42_20260216_1942/latent_dependency_tc_mi_20260217_1228/latent_dependency_summary.json`
  - `ML/autoencoder/experiments/NonDiagonel/results/evaluations/post_training_custom_geo_repi_s42_20260216_1942/latent_dependency_tc_mi_20260217_1228/pareto_scores.json`

Not: Her kolonda en iyi deger **koyu** isaretlendi. (Yuksek iyi/dukuk iyi yonleri tabloda belirtildi.)

## 1) Model Boyutu ve Egitim Sonucu

| Model | Parametre Sayisi ↓ | Best Val Loss ↓ | Best Epoch |
|:---|---:|---:|---:|
| BaselineGeo_large_ld128 | 17,584,451 | **173.7884** | 31 |
| FullCovGeo_large_ld128 | 149,201,155 | 176.5579 | 99 |
| BaselineGeo_small_ld128 | 13,050,499 | 174.1540 | 28 |
| FullCovGeo_small_ld128 | 111,895,107 | 179.3707 | 29 |
| BaselineGeo_small_ld64 | **8,348,163** | 176.8245 | 29 |
| FullCovGeo_small_ld64 | 32,864,739 | 178.4879 | 29 |

## 2) OOD Metrikleri (52 Sample)

| Model | SSIM ↑ | S-Corr ↑ | SC ↓ | STA/LTA Err ↓ | LSD ↓ | MR-LSD ↓ | Arias Err ↓ | Env Corr ↑ | DTW ↓ | XCorr ↑ |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BaselineGeo_large_ld128 | 0.6279 | 0.9688 | 0.2019 | 0.0765 | 1.7387 | 1.9421 | 0.3909 | 0.5845 | 12121.33 | 0.2142 |
| FullCovGeo_large_ld128 | 0.6118 | 0.9619 | 0.2191 | 0.0748 | 1.7088 | 1.7996 | 0.5150 | 0.5693 | 11668.84 | 0.2051 |
| BaselineGeo_small_ld128 | **0.6313** | **0.9692** | **0.1949** | **0.0658** | 1.6900 | **1.7827** | 0.4062 | **0.5892** | 12090.83 | 0.2116 |
| FullCovGeo_small_ld128 | 0.6126 | 0.9585 | 0.2266 | 0.0896 | **1.5877** | 1.8351 | 0.6674 | 0.5481 | **11634.71** | 0.1989 |
| BaselineGeo_small_ld64 | 0.6140 | 0.9649 | 0.2087 | 0.0709 | 3.2223 | 1.8249 | **0.3850** | 0.5662 | 11654.77 | **0.2149** |
| FullCovGeo_small_ld64 | 0.5941 | 0.9599 | 0.2226 | 0.0876 | 1.9718 | 1.8542 | 0.6903 | 0.5414 | 11711.58 | 0.1983 |

## 3) Latent Bagimlilik Metrikleri

| Model | TC_agg ↓ | Offdiag Mean ↓ | Offdiag p95 ↓ | Pairwise MI Mean ↓ | Posterior TC Mean ↓ |
|:---|---:|---:|---:|---:|---:|
| BaselineGeo_large_ld128 | 69.0160 | 0.1733 | 0.4412 | 0.0279 | **0.0000** |
| FullCovGeo_large_ld128 | 42.5538 | 0.1404 | 0.3814 | 0.0181 | 45.8021 |
| BaselineGeo_small_ld128 | 72.4135 | 0.1844 | 0.4776 | 0.0314 | **0.0000** |
| FullCovGeo_small_ld128 | 38.3087 | **0.1372** | **0.3550** | **0.0171** | 52.1209 |
| BaselineGeo_small_ld64 | 42.7390 | 0.2453 | 0.5730 | 0.0525 | **0.0000** |
| FullCovGeo_small_ld64 | **28.5853** | 0.1994 | 0.4958 | 0.0359 | 30.8810 |

## 4) Birlesik Skorlar (Pareto)

| Model | Quality Score ↑ | Independence Score ↑ | Pareto Nondominated |
|:---|---:|---:|:---:|
| BaselineGeo_large_ld128 | 0.6949 | 0.5108 | yes |
| FullCovGeo_large_ld128 | 0.5975 | 0.8751 | yes |
| BaselineGeo_small_ld128 | **0.8731** | 0.3993 | yes |
| FullCovGeo_small_ld128 | 0.3426 | **0.9445** | yes |
| BaselineGeo_small_ld64 | 0.6699 | 0.1693 | no |
| FullCovGeo_small_ld64 | 0.2507 | 0.5618 | no |

## 5) Kisa Okuma

- OOD metriklerinin cogunda en iyi denge: `BaselineGeo_small_ld128`.
- `small_ld128`, `large_ld128`e gore daha kucuk bir model olmasina ragmen kaliteyi koruyor ve bircok OOD metrigi iyilesiyor.
- `ld64`e inmek (ozellikle baseline tarafinda) genel kaliteyi artirmadi; bazi metriklerde belirgin bozulma var (ozellikle LSD).
- FullCov modeller secili metriklerde (LSD/DTW) guclu kalirken, genel denge skoru baseline small ld128 kadar iyi degil.
