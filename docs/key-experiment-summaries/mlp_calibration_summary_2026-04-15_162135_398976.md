# MLP Calibration Summary

- Run artifacts: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-15_162135_398976_mlp_calibration`
- Calibration type: training CV only, no held-out/test usage.
- Training form: one native multi-output MLP per fold/parameter combo.
- Outer CV: 5-fold stratified (random_state=42)
- Parallel target workers: 7

## Selected Defaults

- `v25_policies`: `hidden_layer_sizes=(32,)`, `alpha=0.001`
- `taste_policies`: `hidden_layer_sizes=(32,)`, `alpha=0.001`

## Top Feature Sets At Selected Params

- `v25_policies`: `lambda_policies_plus_sentence_prose`, `llm_engineering_plus_sentence_prose`
- `taste_policies`: `lambda_policies_plus_sentence_prose`, `hq_plus_sentence_prose`

## Calibration Table (feature_set x parameter combo)

| target_family | feature_set_id | hidden_layer_sizes | alpha | primary_metric | primary_mean | primary_std |
|---|---|---|---:|---|---:|---:|
| taste_policies | hq_plus_sentence_prose | (8,) | 0.0010 | f0_5 | 0.3880 | 0.2486 |
| taste_policies | hq_plus_sentence_prose | (8,) | 0.0100 | f0_5 | 0.3877 | 0.2486 |
| taste_policies | hq_plus_sentence_prose | (8,) | 0.1000 | f0_5 | 0.3803 | 0.2472 |
| taste_policies | hq_plus_sentence_prose | (16,) | 0.0010 | f0_5 | 0.6584 | 0.1971 |
| taste_policies | hq_plus_sentence_prose | (16,) | 0.0100 | f0_5 | 0.6499 | 0.2063 |
| taste_policies | hq_plus_sentence_prose | (16,) | 0.1000 | f0_5 | 0.6712 | 0.2033 |
| taste_policies | hq_plus_sentence_prose | (16, 8) | 0.0010 | f0_5 | 0.5320 | 0.2108 |
| taste_policies | hq_plus_sentence_prose | (16, 8) | 0.0100 | f0_5 | 0.5249 | 0.2057 |
| taste_policies | hq_plus_sentence_prose | (16, 8) | 0.1000 | f0_5 | 0.5147 | 0.2111 |
| taste_policies | hq_plus_sentence_prose | (32,) | 0.0010 | f0_5 | 0.7047 | 0.2020 |
| taste_policies | hq_plus_sentence_prose | (32,) | 0.0100 | f0_5 | 0.6807 | 0.2119 |
| taste_policies | hq_plus_sentence_prose | (32,) | 0.1000 | f0_5 | 0.6986 | 0.2085 |
| taste_policies | lambda_policies_plus_sentence_prose | (8,) | 0.0010 | f0_5 | 0.6048 | 0.2241 |
| taste_policies | lambda_policies_plus_sentence_prose | (8,) | 0.0100 | f0_5 | 0.6384 | 0.2219 |
| taste_policies | lambda_policies_plus_sentence_prose | (8,) | 0.1000 | f0_5 | 0.6313 | 0.2274 |
| taste_policies | lambda_policies_plus_sentence_prose | (16,) | 0.0010 | f0_5 | 0.7096 | 0.1977 |
| taste_policies | lambda_policies_plus_sentence_prose | (16,) | 0.0100 | f0_5 | 0.7008 | 0.2041 |
| taste_policies | lambda_policies_plus_sentence_prose | (16,) | 0.1000 | f0_5 | 0.7014 | 0.2038 |
| taste_policies | lambda_policies_plus_sentence_prose | (16, 8) | 0.0010 | f0_5 | 0.6149 | 0.2208 |
| taste_policies | lambda_policies_plus_sentence_prose | (16, 8) | 0.0100 | f0_5 | 0.6086 | 0.2189 |
| taste_policies | lambda_policies_plus_sentence_prose | (16, 8) | 0.1000 | f0_5 | 0.6086 | 0.2181 |
| taste_policies | lambda_policies_plus_sentence_prose | (32,) | 0.0010 | f0_5 | 0.7418 | 0.1987 |
| taste_policies | lambda_policies_plus_sentence_prose | (32,) | 0.0100 | f0_5 | 0.7440 | 0.1994 |
| taste_policies | lambda_policies_plus_sentence_prose | (32,) | 0.1000 | f0_5 | 0.7450 | 0.2025 |
| taste_policies | llm_engineering_plus_sentence_prose | (8,) | 0.0010 | f0_5 | 0.4468 | 0.2339 |
| taste_policies | llm_engineering_plus_sentence_prose | (8,) | 0.0100 | f0_5 | 0.4997 | 0.2092 |
| taste_policies | llm_engineering_plus_sentence_prose | (8,) | 0.1000 | f0_5 | 0.5243 | 0.2016 |
| taste_policies | llm_engineering_plus_sentence_prose | (16,) | 0.0010 | f0_5 | 0.5901 | 0.2192 |
| taste_policies | llm_engineering_plus_sentence_prose | (16,) | 0.0100 | f0_5 | 0.5768 | 0.2238 |
| taste_policies | llm_engineering_plus_sentence_prose | (16,) | 0.1000 | f0_5 | 0.6069 | 0.2063 |
| taste_policies | llm_engineering_plus_sentence_prose | (16, 8) | 0.0010 | f0_5 | 0.4604 | 0.2037 |
| taste_policies | llm_engineering_plus_sentence_prose | (16, 8) | 0.0100 | f0_5 | 0.5142 | 0.2106 |
| taste_policies | llm_engineering_plus_sentence_prose | (16, 8) | 0.1000 | f0_5 | 0.4813 | 0.2006 |
| taste_policies | llm_engineering_plus_sentence_prose | (32,) | 0.0010 | f0_5 | 0.6517 | 0.1961 |
| taste_policies | llm_engineering_plus_sentence_prose | (32,) | 0.0100 | f0_5 | 0.6640 | 0.1942 |
| taste_policies | llm_engineering_plus_sentence_prose | (32,) | 0.1000 | f0_5 | 0.6449 | 0.2044 |
| taste_policies | sentence_prose | (8,) | 0.0010 | f0_5 | 0.3365 | 0.2283 |
| taste_policies | sentence_prose | (8,) | 0.0100 | f0_5 | 0.3337 | 0.2273 |
| taste_policies | sentence_prose | (8,) | 0.1000 | f0_5 | 0.3160 | 0.2137 |
| taste_policies | sentence_prose | (16,) | 0.0010 | f0_5 | 0.4331 | 0.2351 |
| taste_policies | sentence_prose | (16,) | 0.0100 | f0_5 | 0.4290 | 0.2359 |
| taste_policies | sentence_prose | (16,) | 0.1000 | f0_5 | 0.4114 | 0.2376 |
| taste_policies | sentence_prose | (16, 8) | 0.0010 | f0_5 | 0.3347 | 0.2304 |
| taste_policies | sentence_prose | (16, 8) | 0.0100 | f0_5 | 0.3402 | 0.2326 |
| taste_policies | sentence_prose | (16, 8) | 0.1000 | f0_5 | 0.3184 | 0.2145 |
| taste_policies | sentence_prose | (32,) | 0.0010 | f0_5 | 0.4660 | 0.2268 |
| taste_policies | sentence_prose | (32,) | 0.0100 | f0_5 | 0.4553 | 0.2293 |
| taste_policies | sentence_prose | (32,) | 0.1000 | f0_5 | 0.4407 | 0.2391 |
| v25_policies | hq_plus_sentence_prose | (8,) | 0.0010 | r2 | 0.1655 | 0.0687 |
| v25_policies | hq_plus_sentence_prose | (8,) | 0.0100 | r2 | 0.1647 | 0.0685 |
| v25_policies | hq_plus_sentence_prose | (8,) | 0.1000 | r2 | 0.2152 | 0.0907 |
| v25_policies | hq_plus_sentence_prose | (16,) | 0.0010 | r2 | 0.3020 | 0.0976 |
| v25_policies | hq_plus_sentence_prose | (16,) | 0.0100 | r2 | 0.3073 | 0.0964 |
| v25_policies | hq_plus_sentence_prose | (16,) | 0.1000 | r2 | 0.3092 | 0.1012 |
| v25_policies | hq_plus_sentence_prose | (16, 8) | 0.0010 | r2 | 0.2487 | 0.0932 |
| v25_policies | hq_plus_sentence_prose | (16, 8) | 0.0100 | r2 | 0.2719 | 0.1038 |
| v25_policies | hq_plus_sentence_prose | (16, 8) | 0.1000 | r2 | 0.2620 | 0.1025 |
| v25_policies | hq_plus_sentence_prose | (32,) | 0.0010 | r2 | 0.3417 | 0.0937 |
| v25_policies | hq_plus_sentence_prose | (32,) | 0.0100 | r2 | 0.3455 | 0.0929 |
| v25_policies | hq_plus_sentence_prose | (32,) | 0.1000 | r2 | 0.3259 | 0.0953 |
| v25_policies | lambda_policies_plus_sentence_prose | (8,) | 0.0010 | r2 | 0.3660 | 0.0931 |
| v25_policies | lambda_policies_plus_sentence_prose | (8,) | 0.0100 | r2 | 0.3658 | 0.0941 |
| v25_policies | lambda_policies_plus_sentence_prose | (8,) | 0.1000 | r2 | 0.3768 | 0.0868 |
| v25_policies | lambda_policies_plus_sentence_prose | (16,) | 0.0010 | r2 | 0.3924 | 0.0806 |
| v25_policies | lambda_policies_plus_sentence_prose | (16,) | 0.0100 | r2 | 0.3929 | 0.0801 |
| v25_policies | lambda_policies_plus_sentence_prose | (16,) | 0.1000 | r2 | 0.3988 | 0.0790 |
| v25_policies | lambda_policies_plus_sentence_prose | (16, 8) | 0.0010 | r2 | 0.3373 | 0.0972 |
| v25_policies | lambda_policies_plus_sentence_prose | (16, 8) | 0.0100 | r2 | 0.3283 | 0.0991 |
| v25_policies | lambda_policies_plus_sentence_prose | (16, 8) | 0.1000 | r2 | 0.3394 | 0.0956 |
| v25_policies | lambda_policies_plus_sentence_prose | (32,) | 0.0010 | r2 | 0.4000 | 0.0819 |
| v25_policies | lambda_policies_plus_sentence_prose | (32,) | 0.0100 | r2 | 0.4014 | 0.0812 |
| v25_policies | lambda_policies_plus_sentence_prose | (32,) | 0.1000 | r2 | 0.4022 | 0.0800 |
| v25_policies | llm_engineering_plus_sentence_prose | (8,) | 0.0010 | r2 | 0.3451 | 0.0812 |
| v25_policies | llm_engineering_plus_sentence_prose | (8,) | 0.0100 | r2 | 0.3553 | 0.0762 |
| v25_policies | llm_engineering_plus_sentence_prose | (8,) | 0.1000 | r2 | 0.3583 | 0.0760 |
| v25_policies | llm_engineering_plus_sentence_prose | (16,) | 0.0010 | r2 | 0.3758 | 0.0738 |
| v25_policies | llm_engineering_plus_sentence_prose | (16,) | 0.0100 | r2 | 0.3805 | 0.0745 |
| v25_policies | llm_engineering_plus_sentence_prose | (16,) | 0.1000 | r2 | 0.3812 | 0.0741 |
| v25_policies | llm_engineering_plus_sentence_prose | (16, 8) | 0.0010 | r2 | 0.3104 | 0.0892 |
| v25_policies | llm_engineering_plus_sentence_prose | (16, 8) | 0.0100 | r2 | 0.3083 | 0.0871 |
| v25_policies | llm_engineering_plus_sentence_prose | (16, 8) | 0.1000 | r2 | 0.3096 | 0.0960 |
| v25_policies | llm_engineering_plus_sentence_prose | (32,) | 0.0010 | r2 | 0.3959 | 0.0721 |
| v25_policies | llm_engineering_plus_sentence_prose | (32,) | 0.0100 | r2 | 0.3917 | 0.0717 |
| v25_policies | llm_engineering_plus_sentence_prose | (32,) | 0.1000 | r2 | 0.3967 | 0.0709 |
| v25_policies | sentence_prose | (8,) | 0.0010 | r2 | 0.2984 | 0.0974 |
| v25_policies | sentence_prose | (8,) | 0.0100 | r2 | 0.3156 | 0.0999 |
| v25_policies | sentence_prose | (8,) | 0.1000 | r2 | 0.3022 | 0.1008 |
| v25_policies | sentence_prose | (16,) | 0.0010 | r2 | 0.3420 | 0.0840 |
| v25_policies | sentence_prose | (16,) | 0.0100 | r2 | 0.3568 | 0.0814 |
| v25_policies | sentence_prose | (16,) | 0.1000 | r2 | 0.3492 | 0.0812 |
| v25_policies | sentence_prose | (16, 8) | 0.0010 | r2 | 0.2926 | 0.0972 |
| v25_policies | sentence_prose | (16, 8) | 0.0100 | r2 | 0.2882 | 0.0990 |
| v25_policies | sentence_prose | (16, 8) | 0.1000 | r2 | 0.2953 | 0.0983 |
| v25_policies | sentence_prose | (32,) | 0.0010 | r2 | 0.3668 | 0.0778 |
| v25_policies | sentence_prose | (32,) | 0.0100 | r2 | 0.3638 | 0.0787 |
| v25_policies | sentence_prose | (32,) | 0.1000 | r2 | 0.3667 | 0.0780 |
