# Experiment 001 Spec

## Amaç

Ilk temiz ve tamamen izlenebilir baseline hattini kurmak.

## Durum

- Kodlama: Basladi (core pipeline implement edildi)
- Egitim: Baslamadi
- Degerlendirme: Baslamadi

## Zorunlu Kararlar

- D001: Gorev tipi (reconstruction mi, condition-only generation mi)
- D002: Veri split politikasi (event-wise test/ood)
- D003: Girdi temsil formu (waveform mu STFT mi)
- D004: Model ailesi (AE/VAE/CVAE)
- D005: Condition degisken seti
- D006: Normalizasyon politikasi
- D007: Loss fonksiyonu
- D008: Metrik seti
- D009: Egitim butcesi ve early stopping
- D010: Checkpoint secim kriteri
- D011: STFT sabitleri (n_fft/hop/window/frame semantigi)
- D012: Condition-only latent sampling politikasi
- D013: `best_condgen_composite` formulu ve iki-asamali gate kurallari
- D014: Onset picker parametreleri (window/confidence/failure)
- D015: Imbalance guardrail esikleri (I3)

## Karar Durumu

- D001: Frozen (C secildi; ana odak B)
- D002: Frozen (event-wise split + ayri OOD event split)
- D003: Frozen (complex STFT; sabitler D011 ile netlesti)
- D004: Frozen (CVAE)
- D005: Frozen (temel geometri + 1D-hiz-modeli turevi travel-time + station embedding)
- D006: Frozen (train-only global z-score; per-frequency notu sonraya)
- D007: Frozen (complex L1 + log-magnitude L1; MR-STFT recipe not edildi)
- D008: Frozen (complex_l1, mr_lsd, xcorr_max+lag, envelope_corr, band_energy_ratio_error, onset delta-t)
- D009: Frozen (C: 180 epoch, AdamW, RLROP, early stopping)
- D010: Frozen (C: `best_val_loss` (legacy) + `best_val_fair` (selection) + `best_condgen_composite`, duzenli run agaci)
- D011: Frozen (`256/256/32`, hann, onesided, Nyquist-drop, `2x128x220`)
- D012: Frozen (condition-only: `z~N(0,I)`, multi-sample aggregate, `K=8/32`, fixed seed bank)
- D013: Frozen (iki-asamali gate + robust-z composite, aile agirliklari `0.35/0.45/0.20`)
- D014: Frozen (max-derivative picker, `P±4s`, `S±6s`, `conf>=2.5`)
- D015: Frozen (balanced guardrail: `M>=5 +5%`, `All <=8%`, `3<=M<5 <=10%`)

Not (operasyonel):

- Condition-only secim eval'i full-val uzerinde yapilir (`cond_eval_subset_size=null`).
- `ge5` secim kapisinda kullanilmaz; yalnizca test-holdout olarak raporlanir.

## Freeze Kriteri

Tum D001-D015 kararlar `Frozen` olmadan kodlama baslamaz.

## Freeze Sonucu

- Tum zorunlu kararlar `Frozen`.
- Kodlama asamasina gecis izni verildi.

## Ek Frozen Kararlar (A0-U1)

- Veri kaynagi: external HH (`A0`)
- Window/alignment: external dataset semantigi oldugu gibi (`A1`, `A2`)
- Complex normalize: real/imag ortak scale, global RMS (`B1`, `B2`)
- STFT sabitleri: `256/256/32`, `hann`, onesided, Nyquist-drop, `2x128x220` (`D011`)
- Condition-only sampling: `z~N(0,I)`, sweep `K=8`, final `K=32`, fixed seed bank (`D012`)
- Condgen composite: stage-1 onset gate + stage-2 quality gate + robust-z aile skoru, final rerank `K=32` (`D013`)
- Onset picker: `max-derivative + confidence gate` ve sabit P/S pencereleri (`D014`)
- Imbalance secim kapisi: balanced guardrail (`D015`)
- Bilesen: Z-only (`C0`)
- Frame sabitleme: preprocess-time fixed size + right-pad/right-crop fallback (`C1`, `C2`)
- KL anti-collapse: linear warmup + free-bits (`K1`)
- Onset QA/failure protokolu: auto+manual gate, failure-rate zorunlu rapor (`P1`, `P2`, `P3`)
- Imbalance politikasi: iki-asamali + soft weighted sampler (`I1`, `I2`, `I3`)
- Filtering: external veri ek filtre olmadan (`F1`)
- Band metrik bantlari: low/high frozen (`F2`)
- Units: HH counts as-is, response removal yok (`U1`)
