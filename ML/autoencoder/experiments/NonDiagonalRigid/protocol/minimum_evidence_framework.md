# OffDiagonal Minimum Evidence Framework (V1)

Bu dokuman, OffDiagonal hipotezi icin minimum kabul kosullarini operasyonel hale getirir.

## Scope

- Deney hatti: `NonDiagonalRigid`
- Model aileleri: `Baseline`, `FullCov`
- Degerlendirme: `ID/Test` + `OOD (post-training custom)`

## 4 Zorunlu Kanit

1. **Signal Reality**
   - OffDiagonal/TC sinyali null dagilimdan ayristirilmali.
2. **Predictive Utility**
   - `ODI_k`, bir sonraki kucultme adimindaki kalite dususunu ongormeli.
3. **Decision Benefit**
   - `Joint (Q+ODI)` policy, quality-only'e gore olculebilir pratik fayda getirmeli.
4. **Cross-Setting Consistency**
   - Bulgular model ailesi ve dagilimlar arasinda tutarli olmali.

## Zorunlu Cikti Dosyalari

Her confirmatory tur sonunda asagidaki dosyalar uretilir:

- `results/<run_id>/evidence/evidence_gate_summary.md`
- `results/<run_id>/evidence/evidence_gate_summary.json`
- `results/<run_id>/evidence/signal_reality_null_test.csv`
- `results/<run_id>/evidence/predictive_utility_stepwise.csv`
- `results/<run_id>/evidence/policy_benefit_comparison.csv`
- `results/<run_id>/evidence/cross_setting_consistency.csv`

## Gating Kurali

- `Q61`, `Q62`, `Q63`, `Q64` hepsi `PASS` ise:
  - OffDiagonal temelli policy iddiasi confirmatory olarak raporlanabilir.
- En az biri `FAIL` ise:
  - iddia `exploratory` seviyede tutulur.
  - hangi kanitin neden fail oldugu acikca raporlanir.

## Raporlama Standardi

- Tum sayisal sonuclar seed-ayrik + aile-ayrik verilir.
- Tum pass/fail kararlarinda esik deger, ham deger ve CI ayni tabloda yazilir.
- Sonraki turda esik degisecekse:
  - yeni versiyon (`V2`) acilir,
  - eski V1 esikleri degistirilmez.
