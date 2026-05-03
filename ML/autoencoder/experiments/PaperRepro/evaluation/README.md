# Evaluation

Bu klasor reconstruction ve generation evaluation scriptlerini tutacak.

Plan:
- Stage-1 reconstruction evaluation
- Stage-2 generation evaluation
- paper-style metric paketleri
- test ve OOD ayri raporlar

Stage-1 script:
- `evaluate_stage1_autoencoder.py`
- input:
  - frozen config
  - Stage-1 checkpoint
- output:
  - `results/stage1_eval/<run_name>/summary.json`
  - `results/stage1_eval/<run_name>/summary.md`
  - `results/stage1_eval/<run_name>/metrics_test.json`
  - `results/stage1_eval/<run_name>/metrics_ood.json`
  - `results/stage1_eval/<run_name>/figures/test_spectrogram_grid.png`
  - `results/stage1_eval/<run_name>/figures/test_waveform_grid.png`
  - `results/stage1_eval/<run_name>/figures/ood_spectrogram_grid.png`
  - `results/stage1_eval/<run_name>/figures/ood_waveform_grid.png`

Stage-2 script:
- `evaluate_stage2_generation.py`
- input:
  - frozen config
  - Stage-1 checkpoint
  - Stage-2 checkpoint
  - classifier checkpoint
  - Stage-2 latent cache root
- output:
  - `results/stage2_eval/<run_name>/summary.json`
  - `results/stage2_eval/<run_name>/summary.md`
  - `results/stage2_eval/<run_name>/metrics_test.json`
  - `results/stage2_eval/<run_name>/metrics_ood.json`
  - `results/stage2_eval/<run_name>/bin_metrics_test.json`
  - `results/stage2_eval/<run_name>/bin_metrics_test.md`
  - `results/stage2_eval/<run_name>/bin_metrics_ood.json`
  - `results/stage2_eval/<run_name>/bin_metrics_ood.md`
  - `results/stage2_eval/<run_name>/figures/test_spectrogram_grid.png`
  - `results/stage2_eval/<run_name>/figures/test_magnitude_bin_grid.png`
  - `results/stage2_eval/<run_name>/figures/test_distance_bin_grid.png`
  - `results/stage2_eval/<run_name>/figures/test_waveform_grid.png`
  - `results/stage2_eval/<run_name>/figures/test_envelope_distribution.png`
  - `results/stage2_eval/<run_name>/figures/test_fourier_distribution.png`
  - `results/stage2_eval/<run_name>/figures/ood_spectrogram_grid.png`
  - `results/stage2_eval/<run_name>/figures/ood_magnitude_bin_grid.png`
  - `results/stage2_eval/<run_name>/figures/ood_distance_bin_grid.png`
  - `results/stage2_eval/<run_name>/figures/ood_waveform_grid.png`
  - `results/stage2_eval/<run_name>/figures/ood_envelope_distribution.png`
  - `results/stage2_eval/<run_name>/figures/ood_fourier_distribution.png`

Benchmark reference script:
- `build_benchmark_reference_set.py`
- output:
  - `results/benchmark_reference_v1/benchmark_pairs.json`
  - `results/benchmark_reference_v1/benchmark_pairs.md`
  - `results/benchmark_reference_v1/benchmark_pairs_arrays.npz`
  - `results/benchmark_reference_v1/E/benchmark_waveform_grid.png`
  - `results/benchmark_reference_v1/E/benchmark_spectrogram_grid.png`
  - `results/benchmark_reference_v1/N/benchmark_waveform_grid.png`
  - `results/benchmark_reference_v1/N/benchmark_spectrogram_grid.png`
  - `results/benchmark_reference_v1/Z/benchmark_waveform_grid.png`
  - `results/benchmark_reference_v1/Z/benchmark_spectrogram_grid.png`

Benchmark generation render script:
- `render_benchmark_generation.py`
- output:
  - `results/benchmark_generation/<run_name>/benchmark_generation_summary.json`
  - `results/benchmark_generation/<run_name>/benchmark_generation_arrays.npz`
  - `results/benchmark_generation/<run_name>/E/benchmark_generation_waveform_grid.png`
  - `results/benchmark_generation/<run_name>/E/benchmark_generation_spectrogram_grid.png`
  - `results/benchmark_generation/<run_name>/N/benchmark_generation_waveform_grid.png`
  - `results/benchmark_generation/<run_name>/N/benchmark_generation_spectrogram_grid.png`
  - `results/benchmark_generation/<run_name>/Z/benchmark_generation_waveform_grid.png`
  - `results/benchmark_generation/<run_name>/Z/benchmark_generation_spectrogram_grid.png`
