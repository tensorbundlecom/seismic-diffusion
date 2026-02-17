# Backbone Kucultme Metodolojisi (V1)

Bu dokuman, "latent boyut kucultme" ile "backbone kucultme" etkilerini ayristirmak icin V1'de izlenecek metodolojiyi tarif eder.

## Neden Gerekli?

Sadece latent boyutu degistirerek karar vermek confound uretebilir:

- kalite dususu latentten degil, backbone daraltma tarzindan gelebilir
- offdiag/ODI artisi latentten degil, encoder-decoder kapasite dagilimindan gelebilir

Bu nedenle backbone kucultme, latent taramasindan once ayri bir fazda test edilmelidir.

## Onerilen Fazli Tasarim

## Faz-1: Backbone Policy Secimi (latent sabit)

Amaç: "Nasil kucultuldugu"nu secmek.

Sabitler:
- latent: `ld=128` (sabit)
- model ailesi: `baseline_diag`, `fullcov`
- fairness: `iso-step`
- veri/split/normalization: frozen V1

Politikalar:
1. `width-only`: kanal sayilari olceklenir, blok sayisi sabit
2. `depth-only`: blok sayisi azalir, kanal plani korunur
3. `hybrid`: hem width hem depth kontrollu azalir

Kucultme seviyeleri (ornek):
- `1.00` (referans)
- `0.75`
- `0.50`

Degerlendirme:
- primary: `Q` (STFT primary metric composite)
- guardrail: `LSD_spec`, `OnsetErr_spec`, `EnvCorr_spec` alt metrik kontrolu
- secondary: `ODI` + ham bilesenler

Faz-1 cikisi:
- iki aile icin **ortak** tek `policy + scale` secilir
- secimde kural: kalite toleransi korunurken (eps), guardrail ihlali olmayan adaylar icinde daha iyi/benzer ODI + daha dusuk karmasiklik

## Faz-2: Latent Tarama (secilen policy ile)

Amaç: policy sabitken latent boyut etkisini izole etmek.

Sabitler:
- Faz-1'de secilen policy/scale
- latent grid: `[16, 32, 48, 64, 96, 128, 160]`
- ayni seed ve fairness protokolu

Karar:
- resmi stop rule: `Q_{k+1} >= Q_k - eps` ve `ODI_{k+1} <= ODI_k + delta`

## Faz-3: Confirmatory Kapanis

- Q61-Q64 minimum kanitlar
- ID + OOD tutarliligi
- quality-only vs Q+ODI policy karsilastirmasi

## Kritik Kontroller (Confound Onleme)

1. Receptive field/downsample yapisi mumkun oldugunca sabit tutulmali.
2. Station embedding boyutu ve condition pipeline sabit tutulmali.
3. Optimizer/LR/early-stop politikasi adaylar arasi degismemeli.
4. Split ve normalization sadece frozen artefaktlardan okunmali.

## Onerilen Karar Sirasi

1. Once `en iyi backbone kucultme policy` sec
2. Sonra latent kucultme karari ver
3. En sonda Q61-Q64 ile confirmatory PASS/FAIL ver

Bu siralama, "backbone etkisi" ile "latent etkisi"ni karistirmamayi saglar.

## Butce ve Run Matrisi (V1)

Faz-1 (policy secimi):
- `2 aile x 3 policy x 3 scale x 3 seed = 54 run`
- V1 uygulama: iki-asamali
  - pilot eleme (kisa)
  - final dogrulama (tam butce)

Faz-2 (latent tarama):
- `2 aile x 7 latent x 3 seed = 42 run`

Toplam:
- `96 run` (+ evaluation)

Not: Bu matris, rigidlik ve yorumlanabilirlik icin referans butcedir.

## V1 Frozen Referanslar

- Policy grid: `configs/backbone_policy_grid_v1.yaml`
- Egitim butcesi: `configs/training_budget_v1.yaml`
- Karar kaydi: `protocol/decisions.md` (Q65-Q72)
