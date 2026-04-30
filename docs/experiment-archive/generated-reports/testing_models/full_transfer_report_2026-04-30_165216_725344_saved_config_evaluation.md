# Lambda Bundle Full Transfer Report

- Run dir: `C:\Users\joelb\OneDrive\Vela_partnerships_project\Teacher_student_project\Vela_teacher_student_llm_reasoning\tmp\runs\teacher_student_distillation_v1\2026-04-30_165216_725344_saved_config_evaluation`
- Target family: `v25_policies`
- Feature set: `lambda_policies_plus_sentence_bundle`
- Combo refs: ['C:\\Users\\joelb\\OneDrive\\Vela_partnerships_project\\Teacher_student_project\\Vela_teacher_student_llm_reasoning\\tmp\\ut\\c_233afff0\\bundle::combo_ridge', 'C:\\Users\\joelb\\OneDrive\\Vela_partnerships_project\\Teacher_student_project\\Vela_teacher_student_llm_reasoning\\tmp\\ut\\c_233afff0\\bundle::combo_xgb', 'C:\\Users\\joelb\\OneDrive\\Vela_partnerships_project\\Teacher_student_project\\Vela_teacher_student_llm_reasoning\\tmp\\ut\\c_233afff0\\bundle::combo_mlp']
- Reproduction reference run: `repro_source`

## CV Validation Performance (Source Runs)

These are CV metrics taken from the source model-testing runs used to build the saved model bundles.

| model_set_id | r2_mean | r2_std | rmse_mean | mae_mean |
|---|---:|---:|---:|---:|
| ridge | 0.3000 | 0.0000 | 0.2000 | 0.1000 |

## Held-out Test Performance (Reasoning Agreement)

| model_set_id | r2_mean | r2_std | rmse_mean | mae_mean |
|---|---:|---:|---:|---:|
| ridge | 0.2000 | 0.0000 | 0.3000 | 0.1000 |

## Held-out Test Performance (Success Transfer)

### reasoning_pred-only

| model_set_id | model_variant | hq_exit_override_branch | Avg Train CV F0.5 +/- std | Test F0.5 | roc_auc | pr_auc | precision | recall | threshold |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| ridge | single_model | with_override | n/a | 0.4000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.5000 |

### HQ + reasoning_pred

| model_set_id | model_variant | hq_exit_override_branch | Avg Train CV F0.5 +/- std | Test F0.5 | roc_auc | pr_auc | precision | recall | threshold |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

### LLM-eng + reasoning_pred

| model_set_id | model_variant | hq_exit_override_branch | Avg Train CV F0.5 +/- std | Test F0.5 | roc_auc | pr_auc | precision | recall | threshold |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Train Threshold Sweep (F0.5)

Top 3 train thresholds by F0.5 per branch and model variant. Full sweep in `success_transfer_train_threshold_sweep.csv`.

| branch_id | model_variant | hq_exit_override_branch | threshold | train_f0_5 | train_precision | train_recall | selected_threshold | selected_train_f0_5 | selected_test_f0_5 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Reproduction Consistency Check

- Tolerance: ±0.005 F0.5

| experiment_id | headline_target_f0_5 | reproduced_test_f0_5 | delta_f0_5 | abs_delta_f0_5 | within_tolerance |
|---|---:|---:|---:|---:|---|
| hq_only | 0.2730 | 0.2720 | -0.0010 | 0.0010 | True |

## Detailed Data Tables (CSV Artifacts)

- `reasoning_transfer_cv_summary.csv`
- `reasoning_transfer_cv_per_target.csv`
- `reasoning_transfer_per_target.csv`
- `reasoning_transfer_summary.csv`
- `success_transfer_metrics.csv`
- `success_transfer_train_threshold_sweep.csv`
- `reproduction_consistency_check.csv`
