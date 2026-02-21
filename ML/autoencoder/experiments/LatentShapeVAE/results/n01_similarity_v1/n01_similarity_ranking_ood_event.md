# N(0,1) Similarity Ranking - ood_event

| Rank | Run | Class | KL | W2 | diag_mae | offdiag | eig_ratio | n01_abs_gap | n01_robust_gap | n01_similarity |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | lsv_stage1_vae_base_ld64_b0p1_s42 | very_close | 0.0625 | 0.0695 | 0.0168 | 0.0057 | 1.5940 | 0.2569 | -3.0382 | 0.9543 |
| 2 | lsv_stage2_vae_base_ld64_b0p1_s42 | very_close | 0.0861 | 0.1660 | 0.0065 | 0.0018 | 1.1837 | 0.2975 | -2.8180 | 0.9436 |
| 3 | lsv_stage2_vae_base_ld64_b0p1_s44 | very_close | 0.1087 | 0.2047 | 0.0061 | 0.0027 | 1.2609 | 0.3745 | -2.4015 | 0.9169 |
| 4 | lsv_stage2_vae_base_ld64_b0p1_s43 | very_close | 0.1349 | 0.2062 | 0.0093 | 0.0075 | 1.6554 | 0.4717 | -1.9119 | 0.8712 |
| 5 | lsv_stage2_vae_base_ld64_b0p1_s43_logvfixv2 | very_close | 0.1218 | 0.2082 | 0.0384 | 0.0012 | 1.1832 | 0.3913 | -1.5245 | 0.8212 |
| 6 | lsv_stage3_vae_base_ld32_fmtB_b0p1_lmax6_s44 | close | 0.1268 | 0.1793 | 0.0518 | 0.0090 | 1.8216 | 0.4819 | -0.9880 | 0.7287 |
| 7 | lsv_stage2_vae_base_ld64_b0p1_s48_logvfixv2 | very_close | 0.1668 | 0.2799 | 0.0311 | 0.0043 | 1.5649 | 0.5743 | -0.7708 | 0.6837 |
| 8 | lsv_stage3_vae_base_ld32_fmtA_b0p1_lmax8_s43 | close | 0.1926 | 0.2792 | 0.0230 | 0.0227 | 1.9280 | 0.6474 | -0.6240 | 0.6511 |
| 9 | lsv_stage2_vae_base_ld64_b0p1_s49_logvfixv2 | very_close | 0.2095 | 0.3040 | 0.0188 | 0.0093 | 2.0025 | 0.6965 | -0.4382 | 0.6078 |
| 10 | lsv_stage3_vae_base_ld32_fmtB_b0p1_lmax6_s43 | very_close | 0.1939 | 0.3314 | 0.0261 | 0.0136 | 1.6567 | 0.6646 | -0.3706 | 0.5916 |
| 11 | lsv_stage3_vae_base_ld32_fmtA_b0p1_lmax8_s42 | close | 0.2013 | 0.2636 | 0.0372 | 0.0229 | 2.1028 | 0.6693 | -0.2229 | 0.5555 |
| 12 | lsv_stage2_vae_base_ld64_b0p1_s50_logvfixv2 | very_close | 0.2244 | 0.3360 | 0.0185 | 0.0086 | 1.9644 | 0.7385 | -0.1730 | 0.5432 |
| 13 | lsv_stage2_vae_base_ld64_b0p1_s45_logvfixv2 | very_close | 0.2082 | 0.3313 | 0.0445 | 0.0053 | 1.6462 | 0.6863 | 0.1908 | 0.4524 |
| 14 | lsv_stage2_vae_base_ld64_b0p1_s43_logvfixv1 | close | 0.1960 | 0.2830 | 0.0714 | 0.0016 | 1.3117 | 0.5825 | 0.3434 | 0.4150 |
| 15 | lsv_stage2_vae_base_ld64_b0p1_s46_logvfixv2 | very_close | 0.2558 | 0.4205 | 0.0175 | 0.0083 | 1.8776 | 0.8426 | 0.4584 | 0.3874 |
| 16 | lsv_stage2_vae_base_ld64_b0p1_s44_logvfixv2 | very_close | 0.2562 | 0.3564 | 0.0268 | 0.0130 | 2.3754 | 0.8424 | 0.5275 | 0.3711 |
| 17 | lsv_stage3_vae_base_ld32_fmtA_b0p1_lmax8_s44 | close | 0.3506 | 0.5052 | 0.0447 | 0.0297 | 2.3881 | 1.0957 | 2.4931 | 0.0763 |
| 18 | lsv_stage2_vae_base_ld64_b0p1_s47_logvfixv2 | very_close | 0.4009 | 0.4980 | 0.0290 | 0.0200 | 2.9980 | 1.1879 | 2.6214 | 0.0678 |
| 19 | lsv_stage3_vae_base_ld32_fmtB_b0p1_lmax6_s42 | close | 0.3961 | 0.5088 | 0.0496 | 0.0331 | 2.8511 | 1.1916 | 3.1178 | 0.0424 |
| 20 | lsv_stage2_vae_base_ld64_b0p1_s51_logvfixv2 | close | 0.6009 | 0.8313 | 0.0280 | 0.0198 | 3.2600 | 1.7417 | 5.9725 | 0.0025 |
| 21 | lsv_stage2_vae_base_ld64_b0p1_s42_logvfixv2 | moderate | 0.7381 | 0.9163 | 0.1030 | 0.0209 | 3.5193 | 2.0205 | 9.4046 | 0.0001 |
| 22 | lsv_stage2_vae_base_ld64_b0p1_s44_logvfixv1 | close | 1.3689 | 1.7877 | 0.0501 | 0.0463 | 5.7212 | 3.6177 | 17.9416 | 0.0000 |
| 23 | lsv_stage3_vae_base_ld32_fmtC_b0p03_anneal_lmax6_s42 | moderate | 1.3471 | 2.0581 | 0.0355 | 0.0778 | 4.8470 | 3.8176 | 18.7270 | 0.0000 |
| 24 | lsv_stage3_vae_base_ld32_fmtC_b0p03_anneal_lmax6_s43 | moderate | 3.8010 | 5.5934 | 0.2761 | 0.1953 | 11.2220 | 10.1370 | 63.0463 | 0.0000 |
| 25 | lsv_stage3_vae_base_ld32_fmtC_b0p03_anneal_lmax6_s44 | moderate | 4.1960 | 6.0774 | 0.2359 | 0.1922 | 15.5177 | 11.0769 | 67.9035 | 0.0000 |

Scoring:
- `n01_abs_gap = KL + W2 + 0.5*diag_mae + 0.25*|log(eig_ratio)|`
- `n01_robust_gap = rz(KL) + rz(W2) + 0.5*rz(diag_mae) + 0.25*rz(|log(eig_ratio)|)`
- `n01_similarity = 1/(1+exp(n01_robust_gap))`
- Lower `gap` is better, higher `similarity` is better.
