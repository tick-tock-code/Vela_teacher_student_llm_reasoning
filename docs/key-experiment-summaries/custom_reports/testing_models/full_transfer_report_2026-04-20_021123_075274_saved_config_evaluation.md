# Lambda Bundle Full Transfer Report

- Run dir: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-20_021123_075274_saved_config_evaluation`
- Target family: `v25_policies`
- Feature set: `lambda_policies_plus_sentence_bundle`
- Combo refs: ['C:\\Users\\joelb\\OneDrive\\Vela_partnerships_project\\Teacher_student_project\\Vela_teacher_student_llm_reasoning\\tmp\\ut\\c_0dbed221\\bundle::combo_ridge', 'C:\\Users\\joelb\\OneDrive\\Vela_partnerships_project\\Teacher_student_project\\Vela_teacher_student_llm_reasoning\\tmp\\ut\\c_0dbed221\\bundle::combo_xgb', 'C:\\Users\\joelb\\OneDrive\\Vela_partnerships_project\\Teacher_student_project\\Vela_teacher_student_llm_reasoning\\tmp\\ut\\c_0dbed221\\bundle::combo_mlp']
- Reproduction reference run: `repro_source`

## Reasoning Transfer Summary

| model_set_id | r2_mean | r2_std | rmse_mean | mae_mean |
|---|---:|---:|---:|---:|
| ridge | 0.2000 | 0.0000 | 0.3000 | 0.1000 |

## Success Transfer Metrics

| model_set_id | branch_id | f0_5 | roc_auc | pr_auc | precision | recall | threshold |
|---|---|---:|---:|---:|---:|---:|---:|
| ridge | pred_reasoning_only | 0.4000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.5000 |

## Reproduction Consistency Check

- Tolerance: ±0.005 F0.5

| experiment_id | headline_target_f0_5 | reproduced_test_f0_5 | delta_f0_5 | abs_delta_f0_5 | within_tolerance |
|---|---:|---:|---:|---:|---|
| hq_only | 0.2730 | 0.2720 | -0.0010 | 0.0010 | True |

## Combined Best Assignment (CV-R2 Source)

| target_id | selected_model_id | selected_combo_id | cv_r2 |
|---|---|---|---:|
| v25_p1 | ridge | combo_ridge | 0.4000 |

## Reasoning Per-Target Metrics

| model_set_id | target_id | r2 | rmse | mae |
|---|---|---:|---:|---:|
| ridge | v25_p1 | 0.2000 | 0.3000 | 0.1000 |
