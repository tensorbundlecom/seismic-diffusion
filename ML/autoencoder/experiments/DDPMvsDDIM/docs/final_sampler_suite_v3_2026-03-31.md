# Final Sampler Suite v3 (2026-03-31)

Bu belge, `v3` varyanti uzerinde yapilacak final `DDPM vs DDIM`
karsilastirmasinin nasil kosulacagini ve artik hangi runtime/kaynak
olcumlerinin resmi olarak kaydedilecegini sabitler.

## 1. Frozen Model Choice

Final sampler karsilastirmasi icin secilen diffusion varyanti:

- `v3 = AdaLNResMLP + v-prediction`

Gerekce:

- `v1 -> v2 -> v3` zincirinde anlamli kalite iyilesmesi geldi
- `v4 = min-SNR` denemesi `v3`'un uzerine cikamadi
- bu nedenle final sampler study icin en guclu ortak model `v3`

## 2. Final Comparison Grid

Final suite su kurulumla kosulacak:

- same checkpoint: `v3 best.pt`
- same stage-1: `stage1_eventwise_v1_best.pt`
- same test split: frozen event-wise test cache
- same sampler seed policy
- same metrics:
  - `spec_corr`
  - `LSD`
  - `MR-LSD`

Karsilastirma grid'i:

- `DDPM-200`
- `DDIM-25`
- `DDIM-50`
- `DDIM-100`

Not:

- `DDPM` her kosuda ayni 200-step reverse zinciri kullanir
- `DDIM` yalnizca `num_inference_steps` degiserek 25/50/100 step ile kosulur

## 3. Runtime / Resource Instrumentation

`run_sampler_comparison.py` artik su bloklari `summary.json` icine yazar.

### 3.1 Runtime block

- `evaluation_wall_time_sec`
- `evaluation_wall_time_min`
- `samples_per_sec`
- `avg_oracle_decode_time_ms`
- `avg_ddpm_sampling_time_ms`
- `avg_ddpm_total_time_ms`
- `avg_ddim_sampling_time_ms`
- `avg_ddim_total_time_ms`

Tanımlar:

- `sampling_time_ms`: sampler'in latent uretme suresi
- `total_time_ms`: sampler + stage-1 decode suresi
- `wall_time`: tum evaluation kosusunun toplam duvar saati suresi

### 3.2 Resources block

- `cpu_percent_avg`
- `cpu_percent_peak`
- `rss_mb_avg`
- `rss_mb_peak_poll`
- `rss_mb_peak_process`
- `gpu_util_percent_avg`
- `gpu_util_percent_peak`
- `gpu_memory_used_mb_avg`
- `gpu_memory_used_mb_peak_poll`
- `gpu_memory_total_mb`
- `torch_peak_allocated_mb`
- `torch_peak_reserved_mb`

Olcum yontemi:

- CPU ve RSS: `psutil`
- process peak RSS: `resource.getrusage`
- GPU util ve memory poll: `nvidia-smi`
- torch allocator peak: `torch.cuda.max_memory_allocated/reserved`

## 4. Operational Interpretation Rule

Final karar yalnizca kaliteye bakilarak verilmeyecek.
Asagidaki iki eksen birlikte degerlendirilecek:

1. kalite
   - `spec_corr` yuksek olsun
   - `LSD`, `MR-LSD` dusuk olsun
2. operasyonel maliyet
   - `wall_time`
   - `samples_per_sec`
   - `avg_ddim_total_time_ms` / `avg_ddpm_total_time_ms`
   - peak memory / CPU / GPU kullanimi

Bu sayede su sorular kapatilacak:

- `DDPM` kalite olarak ne kadar onde?
- `DDIM-25/50/100` bu kaliteye ne kadar yaklasiyor?
- hiz kazanci kalite kaybina degiyor mu?

## 5. Official Launcher

Final suite launcher:

```bash
/home/gms/miniconda3/bin/python3.12 -m ML.autoencoder.experiments.DDPMvsDDIM.evaluation.run_v3_final_sampler_suite
```

Varsayilan output:

- `ML/autoencoder/experiments/DDPMvsDDIM/results/final_sampler_suite_v3/`

Alt klasorler:

- `ddim_025/`
- `ddim_050/`
- `ddim_100/`

Toplu ozet:

- `suite_summary.json`
- `suite_summary.md`

## 6. Smoke Validation

Instrumentation smoke dogrulamasi su sekilde gecti:

- script: `run_sampler_comparison.py`
- checkpoint: `v3 best.pt`
- `max_samples=2`
- `save_plots=none`

Smoke summary artik hem kalite metriklerini hem de runtime/resources
bloklarini dogru yaziyor.
