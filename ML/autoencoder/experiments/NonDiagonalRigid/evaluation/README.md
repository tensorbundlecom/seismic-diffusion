# NonDiagonalRigid Evaluation

Bu klasorde Q61-Q64 minimum kanitlarinin hesaplama scriptleri yer alacak.

Su an:

- `run_evidence_gates.py`: sadece cikti klasor ve tablo semasi olusturan iskelet script.
- Final metrik hesaplari, soru seti ve esikler tamamen dondurulunca eklenecek.

Ornek kullanim:

```bash
python3 ML/autoencoder/experiments/NonDiagonalRigid/evaluation/run_evidence_gates.py \
  --run-id confirmatory_YYYYMMDD_HHMM
```

Cikti:

- `ML/autoencoder/experiments/NonDiagonalRigid/results/<run_id>/evidence/`
