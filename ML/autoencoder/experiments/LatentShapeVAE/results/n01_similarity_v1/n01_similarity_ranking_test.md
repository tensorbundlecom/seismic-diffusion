# N(0,1) Similarity Ranking - test

| Rank | Run | Class | KL | W2 | diag_mae | offdiag | eig_ratio | n01_abs_gap | n01_robust_gap | n01_similarity |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | lsv_stage1_vae_base_ld64_b0p1_s42 | very_close | 0.0080 | 0.0119 | 0.0123 | 0.0001 | 1.0923 | 0.0482 | -2.5258 | 0.9259 |
| 2 | lsv_stage1_vae_base_ld64_b0p03_anneal_s42 | very_close | 0.1585 | 0.3022 | 0.0194 | 0.0002 | 1.1721 | 0.5101 | -2.4745 | 0.9223 |
| 3 | lsv_stage2_vae_base_ld64_b0p1_s43_logvfixv2 | very_close | 0.1227 | 0.2090 | 0.0389 | 0.0013 | 1.1916 | 0.3950 | -2.4568 | 0.9211 |
| 4 | lsv_stage2_vae_base_ld64_b0p1_s42 | very_close | 0.2297 | 0.3215 | 0.0206 | 0.0111 | 2.0220 | 0.7376 | -2.3664 | 0.9142 |
| 5 | lsv_stage2_vae_base_ld64_b0p1_s43_logvfixv1 | close | 0.4619 | 0.5832 | 0.0646 | 0.0155 | 2.9259 | 1.3457 | -2.2112 | 0.9013 |
| 6 | lsv_stage2_vae_base_ld64_b0p1_s48_logvfixv2 | moderate | 2.3877 | 3.2096 | 0.1535 | 0.0504 | 7.8632 | 6.1895 | -1.6035 | 0.8325 |
| 7 | lsv_stage3_vae_base_ld32_fmtB_b0p1_lmax6_s43 | moderate | 2.4778 | 3.4167 | 0.2052 | 0.1431 | 8.6602 | 6.5368 | -1.5045 | 0.8182 |
| 8 | lsv_stage3_vae_base_ld32_fmtA_b0p1_lmax8_s44 | moderate | 3.3974 | 4.7817 | 0.2880 | 0.1632 | 10.5065 | 8.9111 | -1.2107 | 0.7704 |
| 9 | lsv_stage3_vae_base_ld32_fmtA_b0p1_lmax8_s43 | moderate | 3.6581 | 5.0724 | 0.3104 | 0.1814 | 10.7271 | 9.4790 | -1.1405 | 0.7578 |
| 10 | lsv_stage2_vae_base_ld64_b0p1_s45_logvfixv2 | moderate | 6.2868 | 9.2651 | 0.2819 | 0.1019 | 16.9858 | 16.4010 | -0.6158 | 0.6493 |
| 11 | lsv_stage2_vae_base_ld64_b0p1_s51_logvfixv2 | moderate | 7.3037 | 10.8782 | 0.2867 | 0.1440 | 19.9738 | 19.0739 | -0.3974 | 0.5981 |
| 12 | lsv_stage2_vae_base_ld64_b0p1_s49_logvfixv2 | moderate | 7.6257 | 11.4068 | 0.2824 | 0.1274 | 21.0494 | 19.9355 | -0.3339 | 0.5827 |
| 13 | lsv_stage3_vae_base_ld32_fmtA_b0p1_lmax8_s42 | distorted | 6.5884 | 9.7482 | 0.4822 | 0.2679 | 18.9596 | 17.3133 | -0.3032 | 0.5752 |
| 14 | lsv_stage2_vae_base_ld64_b0p1_s50_logvfixv2 | moderate | 8.0995 | 12.2310 | 0.3007 | 0.1325 | 21.3814 | 21.2465 | -0.2197 | 0.5547 |
| 15 | lsv_stage3_vae_base_ld32_fmtB_b0p1_lmax6_s44 | distorted | 6.9174 | 10.2614 | 0.5343 | 0.2234 | 21.8747 | 18.2172 | -0.1565 | 0.5391 |
| 16 | lsv_stage2_vae_base_ld64_b0p1_s44_logvfixv2 | moderate | 9.0049 | 13.2092 | 0.3748 | 0.1897 | 23.9681 | 23.1957 | 0.0257 | 0.4936 |
| 17 | lsv_stage2_vae_base_ld64_b0p1_s46_logvfixv2 | distorted | 10.0013 | 14.3979 | 0.4852 | 0.1561 | 22.1963 | 25.4168 | 0.2997 | 0.4256 |
| 18 | lsv_stage2_vae_base_ld64_b0p1_s47_logvfixv2 | distorted | 11.3978 | 16.9643 | 0.5587 | 0.1758 | 27.5503 | 29.4705 | 0.6983 | 0.3322 |
| 19 | lsv_stage3_vae_base_ld32_fmtB_b0p1_lmax6_s42 | distorted | 10.7799 | 16.6407 | 0.7660 | 0.2632 | 27.9772 | 28.6364 | 0.8741 | 0.2944 |
| 20 | lsv_stage2_vae_base_ld64_b0p1_s42_logvfixv2 | distorted | 17.5877 | 28.0481 | 0.6932 | 0.1881 | 39.8239 | 46.9036 | 2.1136 | 0.1078 |
| 21 | lsv_stage2_vae_base_ld64_b0p1_s44_logvfixv1 | distorted | 35.9933 | 58.4489 | 1.4189 | 0.2916 | 76.5110 | 96.2361 | 6.4805 | 0.0015 |
| 22 | lsv_stage1_beta0_base_ld64_s42_stablev1 | distorted | 43.7002 | 80.0283 | 0.5720 | 0.2297 | 26.7368 | 124.8360 | 7.2131 | 0.0007 |
| 23 | lsv_stage3_vae_base_ld32_fmtC_b0p03_anneal_lmax6_s43 | distorted | 33.9791 | 57.0441 | 2.2634 | 0.5936 | 81.8190 | 93.2559 | 7.2368 | 0.0007 |
| 24 | lsv_stage3_vae_base_ld32_fmtC_b0p03_anneal_lmax6_s42 | distorted | 53.1066 | 91.7705 | 3.3761 | 0.7294 | 140.6992 | 147.8019 | 12.3575 | 0.0000 |
| 25 | lsv_stage3_vae_base_ld32_fmtC_b0p03_anneal_lmax6_s44 | distorted | 99.0234 | 175.6447 | 5.9961 | 0.7452 | 437.5746 | 279.1864 | 24.5888 | 0.0000 |
| 26 | lsv_stage1_ae_base_ld64_s42 | distorted | 337.9953 | 109.9097 | 0.9333 | 0.9480 | 8447594.8595 | 452.3590 | 37.6668 | 0.0000 |
| 27 | lsv_stage1_ae_base_ld64_s42_stablev1 | distorted | 594.2077 | 724.8946 | 9.8477 | 0.9850 | 93780019.1390 | 1328.6152 | 106.9345 | 0.0000 |
| 28 | lsv_stage2_vae_base_ld64_b0p1_s44 | distorted | 5626.7163 | 11047.7796 | 176.0653 | 0.0197 | 11573.4839 | 16764.8677 | 1344.9753 | 0.0000 |
| 29 | lsv_stage2_vae_base_ld64_b0p1_s43 | distorted | 47233.7476 | 93603.2146 | 1476.6423 | 0.0856 | 64683.3322 | 141578.0527 | 11345.4187 | 0.0000 |

Scoring:
- `n01_abs_gap = KL + W2 + 0.5*diag_mae + 0.25*|log(eig_ratio)|`
- `n01_robust_gap = rz(KL) + rz(W2) + 0.5*rz(diag_mae) + 0.25*rz(|log(eig_ratio)|)`
- `n01_similarity = 1/(1+exp(n01_robust_gap))`
- Lower `gap` is better, higher `similarity` is better.
