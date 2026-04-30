# Combination Screening Step 1 (Train CV, Ridge)

- Run: `2026-04-21_132505_959940_model_testing`
- Mode: `model_testing_mode` (train-only screening)
- Target family: `v25_policies`
- Model family/output: `linear_l2` / `single_target`
- Candidate feature sets: 7 locked combination sets
- Repeats (`n_runs`): `4`

## Results

| Rank | Feature set | R2 mean | R2 std | RMSE | MAE | Screen score |
|---|---|---:|---:|---:|---:|---:|
| 1 | `hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle` | 0.4286 | 0.0009 | 0.2638 | 0.1758 | 0.4281 |
| 2 | `hq_plus_lambda_policies_plus_sentence_bundle` | 0.4284 | 0.0013 | 0.2636 | 0.1760 | 0.4277 |
| 3 | `llm_engineering_plus_lambda_policies_plus_sentence_bundle` | 0.4218 | 0.0020 | 0.2651 | 0.1770 | 0.4208 |
| 4 | `lambda_policies_plus_sentence_bundle` | 0.4211 | 0.0015 | 0.2651 | 0.1775 | 0.4203 |
| 5 | `hq_plus_llm_engineering_plus_sentence_bundle` | 0.4204 | 0.0009 | 0.2658 | 0.1808 | 0.4199 |
| 6 | `hq_plus_sentence_bundle` | 0.4187 | 0.0012 | 0.2659 | 0.1815 | 0.4181 |
| 7 | `llm_engineering_plus_sentence_bundle` | 0.3894 | 0.0007 | 0.2737 | 0.1892 | 0.3891 |

## Selected Combo For Step 2

- Feature set: `hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle`
- Combo ref: `data/saved_model_configs/2026-04-21_132505_959940_model_testing::v25_policies__hq_plus_llm_engineering_plus_lambda_policies_plus_sentence_bundle__ridge__single_target__0007`

## Step 2 Results (Success Screening, Train CV Only)

- Run: `2026-04-21_133521_705091_saved_config_evaluation`
- Mode: `saved_config_evaluation_mode` with `combination_transfer_report`
- Held-out evaluation: disabled
- Repeats (`n_runs`): `4`

| Rank | success_branch_id | train_cv_f0_5 | train_cv_f0_5_std | train_cv_roc_auc | train_cv_pr_auc | selected_c_final | threshold |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `hq_plus_llm_engineering__without_override` | 0.3396 | 0.0067 | 0.7643 | 0.2735 | 0.0510 | 0.2950 |
| 2 | `llm_engineering__without_override` | 0.3357 | 0.0048 | 0.7644 | 0.2731 | 0.1585 | 0.2925 |
| 3 | `hq_plus_llm_engineering_plus_lambda_policies__without_override` | 0.3344 | 0.0050 | 0.7610 | 0.2698 | 0.0193 | 0.3050 |

## Step 3 Results (Held-out Test, Selected Branch)

- Run: `2026-04-21_141338_266758_saved_config_evaluation`
- Mode: `saved_config_evaluation_mode` with `combination_transfer_report`
- Held-out evaluation: enabled
- Selected branch: `hq_plus_llm_engineering__without_override`

| success_branch_id | test_f0_5 | test_roc_auc | test_pr_auc | test_precision | test_recall | selected_c_final | threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hq_plus_llm_engineering__without_override` | 0.3218 | 0.7330 | 0.2734 | 0.3918 | 0.1877 | 0.0500 | 0.3000 |
