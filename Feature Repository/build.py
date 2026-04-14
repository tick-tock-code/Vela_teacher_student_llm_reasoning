"""Build the VCBench LLM-derived feature repository.

Reads scattered source files from various experiment directories and writes
clean, standardised CSVs at experiments/feature_repository/{policies,
rrf_questions, hq_baseline, splits, similarity_lookup}.

Usage:
    poetry run python experiments/feature_repository/build.py
    poetry run python experiments/feature_repository/build.py --step splits
    poetry run python experiments/feature_repository/build.py --step policies

This script is idempotent: re-running overwrites the outputs.

!!! IMPORTANT — NOT STANDALONE !!!
This script only runs inside the full think-reason-learn repository.
It reads raw data from sibling directories (e.g. `.private/`,
`experiments/policy_induction_vcbench/`, `experiments/joel_hq_features/`,
`experiments/autoresearch_vcbench/cache/`, and the sibling repo
`Vela_internship_TRL/` for LLM-engineered features).

If you received this file as part of a zipped feature_repository bundle,
you do NOT need to run build.py. The data files (CSVs, NPY) have already
been generated and committed to the bundle, and everything in
REPRODUCING_HEADLINE_SCORES.md and FOR_JOEL_DISTILLATION_PROJECT.md reads
those data files directly without calling this script.

build.py is included in the bundle purely for provenance / as a reference
implementation, so you can see exactly how each file was produced.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_DIR = REPO_ROOT / ".private"
EXPERIMENTS = REPO_ROOT / "experiments"
OUT = EXPERIMENTS / "feature_repository"

PUBLIC_CSV = PRIVATE_DIR / "vcbench_final_public.csv"
PRIVATE_CSV = PRIVATE_DIR / "vcbench_final_private_with_success.csv"

# v25 source files — use the "recreated_preds_*" files (Ben's canonical pipeline)
# rather than the older "v25_*" files (Roy's original). The recreated files are
# the ones that reproduce the documented F0.5 = 0.294 and are used in the dashboard.
# See experiments/policy_induction_vcbench/SCORE_DISCREPANCIES.md for details.
V25_TRAIN_FOLDS = [
    EXPERIMENTS / "policy_induction_vcbench" / "results" / f"recreated_preds_train_fold{i}" / "founder_predictions.csv"
    for i in (1, 2, 3)
]
# Test: fold1 + fold2 + base dir (which plays the role of fold3)
V25_TEST_FOLDS = [
    EXPERIMENTS / "policy_induction_vcbench" / "results" / "recreated_preds_fold1" / "founder_predictions.csv",
    EXPERIMENTS / "policy_induction_vcbench" / "results" / "recreated_preds_fold2" / "founder_predictions.csv",
    EXPERIMENTS / "policy_induction_vcbench" / "results" / "recreated_preds" / "founder_predictions.csv",
]
V25_ANALYSIS_JSON = EXPERIMENTS / "micro-internship-project" / "report_analysis_results.json"
V25_REPORT_MD = EXPERIMENTS / "micro-internship-project" / "POLICY_ABLATION_REPORT.md"

# Taste source files
TASTE_PUBLIC_CSV = EXPERIMENTS / "taste_policy_decomposition" / "results" / "public" / "policy_features.csv"
TASTE_PRIVATE_CSV = EXPERIMENTS / "taste_policy_decomposition" / "results" / "private" / "policy_features.csv"
TASTE_DEFINITIONS_PY = EXPERIMENTS / "taste_policy_decomposition" / "run_taste_policies.py"
TASTE_CV_JSON = EXPERIMENTS / "taste_policy_decomposition" / "results" / "lr_cv_results.json"
TASTE_HOLDOUT_JSON = EXPERIMENTS / "taste_policy_decomposition" / "results" / "lr_holdout_results.json"

# RRF source files
RRF_DIR = EXPERIMENTS / "rrf_vcbench" / "results" / "vcbench_rrf_curated"
RRF_QUESTIONS_PARQUET = RRF_DIR / "questions.parquet"
RRF_TRAIN_SUMMARY = RRF_DIR / "training_summary.json"
RRF_ANSWER_MATRIX = RRF_DIR / "test_predictions" / "answer_matrix.csv"
RRF_PROVENANCE = RRF_DIR / "rrf.json"

# Lambda policy features (from autoresearch_vcbench cache)
AUTORESEARCH_CACHE = EXPERIMENTS / "autoresearch_vcbench" / "cache"
LAMBDA_RULE_FILES = [
    "policy_decomposed_v1.json",
    "policy_decomposed_gpt-4_1.json",
    "policy_decomposed_gemini-2_5-flash.json",
    "targeted_o3-mini.json",
    "targeted_gemini-2_5-pro.json",
    "targeted_gpt-4_1.json",
]

# LLM-Engineered features (Joel's internship — set_05)
# Lives in a sibling repo; not guaranteed to be present
INTERN_REPO = REPO_ROOT.parent / "Vela_internship_TRL"
LLM_ENG_POOL_PARQUET = (
    INTERN_REPO / "LLM_Reasoning_Main" / "docs" / "paper_stats"
    / "features" / "llm_engineered" / "set_05" / "engineered_pool.parquet"
)
LLM_ENG_TEST_PARQUET = (
    INTERN_REPO / "LLM_Reasoning_Main" / "docs" / "paper_stats"
    / "features" / "llm_engineered" / "set_05" / "engineered_test.parquet"
)
LLM_ENG_META_JSON = (
    INTERN_REPO / "LLM_Reasoning_Main" / "docs" / "paper_stats"
    / "features" / "llm_engineered" / "set_05" / "engineered_meta.json"
)
LLM_ENG_SEED_JSON = (
    INTERN_REPO / "LLM_Reasoning_Main" / "features_storage"
    / "llm_engineered" / "seed_100.json"
)

# HQ source
sys.path.insert(0, str(EXPERIMENTS / "joel_hq_features"))


# ---------------------------------------------------------------------------
# Step 1: splits
# ---------------------------------------------------------------------------
def build_splits() -> None:
    """Write canonical train/test UUIDs and labels CSV."""
    print("[splits] Loading VCBench public and private CSVs...")
    train_df = pd.read_csv(PUBLIC_CSV)
    test_df = pd.read_csv(PRIVATE_CSV)

    train_df = train_df.dropna(subset=["success"]).reset_index(drop=True)
    test_df = test_df.dropna(subset=["success"]).reset_index(drop=True)
    train_df["success"] = train_df["success"].astype(int)
    test_df["success"] = test_df["success"].astype(int)

    out_dir = OUT / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "train_uuids.txt").write_text(
        "\n".join(train_df["founder_uuid"].astype(str).tolist()) + "\n"
    )
    (out_dir / "test_uuids.txt").write_text(
        "\n".join(test_df["founder_uuid"].astype(str).tolist()) + "\n"
    )

    labels = pd.concat(
        [
            pd.DataFrame(
                {
                    "founder_uuid": train_df["founder_uuid"].astype(str),
                    "split": "train",
                    "success": train_df["success"],
                }
            ),
            pd.DataFrame(
                {
                    "founder_uuid": test_df["founder_uuid"].astype(str),
                    "split": "test",
                    "success": test_df["success"],
                }
            ),
        ],
        ignore_index=True,
    )
    labels.to_csv(out_dir / "labels.csv", index=False)

    print(f"  train: {len(train_df)} founders, {int(train_df['success'].sum())} positive")
    print(f"  test:  {len(test_df)} founders, {int(test_df['success'].sum())} positive")
    print(f"  wrote {out_dir}")


# ---------------------------------------------------------------------------
# Step 2: policies
# ---------------------------------------------------------------------------
V25_POLICY_IDS = [1, 11, 38, 52, 55, 58, 72, 80, 112, 116, 121, 135, 143, 150, 157, 161]


def parse_v25_policy_text() -> dict[str, str]:
    """Parse the POLICY_ABLATION_REPORT.md to extract full text per policy id."""
    md = V25_REPORT_MD.read_text()
    # Match: ### Policy <id> ... \n\n> <full text>
    pattern = re.compile(
        r"^### Policy (\d+) [—-].*?\n\n> (.+?)(?=\n\n|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    out: dict[str, str] = {}
    for match in pattern.finditer(md):
        pid = match.group(1)
        text = re.sub(r"\s+", " ", match.group(2)).strip()
        out[pid] = text
    return out


def parse_taste_policies() -> list[dict[str, str]]:
    """Extract the 20 Taste policy id+text pairs from run_taste_policies.py."""
    src = TASTE_DEFINITIONS_PY.read_text()
    # Find the POLICIES list and parse each {"id": ..., "text": (...)} block.
    # Use a regex to find each entry; we tolerate parenthesised multi-line strings.
    # Pattern: "id": "...", followed by "text": ( "..."  "..." )
    entries = []
    pattern = re.compile(
        r'"id":\s*"([^"]+)",\s*"text":\s*\((.*?)\)\s*,?\s*\}',
        re.DOTALL,
    )
    for m in pattern.finditer(src):
        pid = m.group(1)
        # Stitch together the inner string literals
        chunks = re.findall(r'"([^"]*)"', m.group(2))
        text = " ".join(chunks).strip()
        text = re.sub(r"\s+", " ", text)
        entries.append({"id": pid, "text": text})
    if len(entries) != 20:
        raise RuntimeError(f"Expected 20 Taste policies, parsed {len(entries)}")
    return entries


def load_v25_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate the 3 train and 3 test fold CSVs into single (4500, 17) frames."""
    train = pd.concat([pd.read_csv(p) for p in V25_TRAIN_FOLDS], ignore_index=True)
    test = pd.concat([pd.read_csv(p) for p in V25_TEST_FOLDS], ignore_index=True)
    pcols = [f"p_true_policy_{i}" for i in V25_POLICY_IDS]
    train_out = train[["founder_uuid"] + pcols].copy()
    test_out = test[["founder_uuid"] + pcols].copy()
    return train_out, test_out


def build_policies() -> None:
    """Build the policies/ subdirectory: policies.csv + train/test predictions + prompts."""
    print("[policies] Building...")
    out_dir = OUT / "policies"
    prompt_dir = out_dir / "prompts"
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    # Load canonical orderings
    train_uuids = (OUT / "splits" / "train_uuids.txt").read_text().strip().split("\n")
    test_uuids = (OUT / "splits" / "test_uuids.txt").read_text().strip().split("\n")
    labels = pd.read_csv(OUT / "splits" / "labels.csv")
    train_y = labels[labels.split == "train"].set_index("founder_uuid")["success"]
    test_y = labels[labels.split == "test"].set_index("founder_uuid")["success"]

    # --- v25: 16 policies with probability scores ---
    print("  v25 policies...")
    v25_text = parse_v25_policy_text()
    with open(V25_ANALYSIS_JSON) as f:
        analysis = json.load(f)
    v25_uc = analysis["univariate_policy_cv"]
    v25_co = analysis["coefficients"]["policy_only"]["coefficients"]

    v25_train_pred, v25_test_pred = load_v25_predictions()
    # Reorder to canonical UUID order
    v25_train_pred = v25_train_pred.set_index("founder_uuid").reindex(train_uuids).reset_index()
    v25_test_pred = v25_test_pred.set_index("founder_uuid").reindex(test_uuids).reset_index()

    # Rename columns: p_true_policy_X -> v25_p<X>
    rename_map = {f"p_true_policy_{i}": f"v25_p{i}" for i in V25_POLICY_IDS}
    v25_train_pred.rename(columns=rename_map, inplace=True)
    v25_test_pred.rename(columns=rename_map, inplace=True)

    v25_rows = []
    for pid in V25_POLICY_IDS:
        feat_id = f"v25_p{pid}"
        uc = v25_uc[f"p_true_policy_{pid}"]
        co = v25_co[f"p_true_policy_{pid}"]
        full_text = v25_text.get(str(pid), uc.get("short_name", "?"))
        # Fire rate: probability scores aren't binary; use mean
        scores = v25_train_pred[feat_id].values
        # For "positive lift", binarise at 0.5 for a stable comparison
        binarised = (scores >= 0.5).astype(int)
        train_y_aligned = train_y.reindex(v25_train_pred["founder_uuid"]).values
        fire_rate = float(binarised.mean())
        if binarised.sum() > 0:
            base_rate = float(train_y_aligned.mean())
            pos_rate_among_fired = float(train_y_aligned[binarised == 1].mean())
            pos_lift = pos_rate_among_fired / base_rate if base_rate > 0 else float("nan")
        else:
            pos_lift = float("nan")

        v25_rows.append(
            {
                "feature_id": feat_id,
                "source": "policy_induction_v25",
                "original_id": str(pid),
                "short_name": uc.get("short_name", ""),
                "category": uc.get("category", ""),
                "text": full_text,
                "llm": "gemini-2.5-flash",
                "scoring_format": "probability",
                "prompt_template_ref": "prompts/policy_induction_v25.md",
                "fire_rate_train": round(fire_rate, 4),
                "pos_lift_at_0.5": round(pos_lift, 3) if not np.isnan(pos_lift) else None,
                "univariate_cv_f0.5": uc.get("cv_f05_mean"),
                "univariate_cv_precision": uc.get("cv_precision_mean"),
                "univariate_cv_recall": uc.get("cv_recall_mean"),
                "pointbiserial_r": uc.get("pointbiserial_r"),
                "lr_coefficient_in_ensemble": co.get("coefficient"),
                "notes": "16-policy v25 ensemble; full ensemble CV F0.5 = 0.294",
            }
        )

    # --- Taste: 20 policies with binary YES/NO scores ---
    print("  Taste policies...")
    taste_defs = parse_taste_policies()
    taste_pub = pd.read_csv(TASTE_PUBLIC_CSV)
    taste_priv = pd.read_csv(TASTE_PRIVATE_CSV)

    taste_id_cols = [d["id"] for d in taste_defs]
    # Reorder to canonical
    taste_pub = taste_pub.set_index("founder_uuid").reindex(train_uuids).reset_index()
    taste_priv = taste_priv.set_index("founder_uuid").reindex(test_uuids).reset_index()

    # Convert YES/NO -> 1/0
    for col in taste_id_cols:
        taste_pub[col] = (taste_pub[col].astype(str).str.upper() == "YES").astype(int)
        taste_priv[col] = (taste_priv[col].astype(str).str.upper() == "YES").astype(int)

    # Build feature columns named taste_<id>
    taste_train_pred = taste_pub[["founder_uuid"] + taste_id_cols].copy()
    taste_test_pred = taste_priv[["founder_uuid"] + taste_id_cols].copy()
    taste_rename = {col: f"taste_{col}" for col in taste_id_cols}
    taste_train_pred.rename(columns=taste_rename, inplace=True)
    taste_test_pred.rename(columns=taste_rename, inplace=True)

    train_y_aligned = train_y.reindex(taste_train_pred["founder_uuid"]).values
    base_rate = float(train_y_aligned.mean())

    taste_rows = []
    for d in taste_defs:
        feat_id = f"taste_{d['id']}"
        scores = taste_train_pred[feat_id].values
        fire_rate = float(scores.mean())
        if scores.sum() > 0:
            pos_rate_among_fired = float(train_y_aligned[scores == 1].mean())
            pos_lift = pos_rate_among_fired / base_rate if base_rate > 0 else float("nan")
        else:
            pos_lift = float("nan")
        # Univariate F0.5 at threshold 0.5 (i.e. predict positive if YES)
        from sklearn.metrics import fbeta_score
        f05 = float(
            fbeta_score(
                train_y_aligned,
                scores.astype(int),
                beta=0.5,
                zero_division=0,
            )
        )
        taste_rows.append(
            {
                "feature_id": feat_id,
                "source": "taste_policy_decomposition",
                "original_id": d["id"],
                "short_name": d["id"],
                "category": "accept" if d["id"].startswith("A")
                else "reject" if d["id"].startswith("R")
                else "meta",
                "text": d["text"],
                "llm": "gemini-2.0-flash",
                "scoring_format": "binary",
                "prompt_template_ref": "prompts/taste_policy_decomposition.md",
                "fire_rate_train": round(fire_rate, 4),
                "pos_lift_at_0.5": round(pos_lift, 3) if not np.isnan(pos_lift) else None,
                "univariate_cv_f0.5": round(f05, 4),
                "univariate_cv_precision": None,
                "univariate_cv_recall": None,
                "pointbiserial_r": None,
                "lr_coefficient_in_ensemble": None,
                "notes": "20-policy ensemble; CV F0.5 ~0.308",
            }
        )

    # --- Combine into one policies.csv ---
    policies_df = pd.DataFrame(v25_rows + taste_rows)
    policies_df.to_csv(out_dir / "policies.csv", index=False)
    print(f"  wrote policies.csv ({len(policies_df)} rows)")

    # --- Combined train/test prediction matrices ---
    train_out = v25_train_pred.merge(taste_train_pred, on="founder_uuid", how="left")
    test_out = v25_test_pred.merge(taste_test_pred, on="founder_uuid", how="left")

    # Reorder columns: founder_uuid first, then v25, then taste
    feat_cols = [f"v25_p{i}" for i in V25_POLICY_IDS] + [f"taste_{d['id']}" for d in taste_defs]
    train_out = train_out[["founder_uuid"] + feat_cols]
    test_out = test_out[["founder_uuid"] + feat_cols]

    # Force canonical UUID order (defensive — merge can reshuffle)
    train_out = train_out.set_index("founder_uuid").reindex(train_uuids).reset_index()
    test_out = test_out.set_index("founder_uuid").reindex(test_uuids).reset_index()

    train_out.to_csv(out_dir / "predictions_train.csv", index=False)
    test_out.to_csv(out_dir / "predictions_test.csv", index=False)
    print(f"  wrote predictions_train.csv {train_out.shape}")
    print(f"  wrote predictions_test.csv {test_out.shape}")

    # --- Prompt templates ---
    write_policy_prompts(prompt_dir)


def write_policy_prompts(prompt_dir: Path) -> None:
    """Write prompt template markdown files."""
    v25_prompt = """\
# Policy Induction v25 — Scoring Prompt

Each policy was scored against each founder using Gemini 2.5 Flash. The model
returns the probability P(success | policy applies, founder).

## System message (paraphrased from the policy_induction module)

> You are evaluating venture capital founders against a specific success
> criterion. Given a founder's anonymised profile and a policy describing
> a success indicator, decide how likely the founder is to succeed
> conditional on the policy. Output a probability between 0 and 1.

## User template

> POLICY: {policy_text}
>
> FOUNDER PROFILE:
> {founder_profile}
>
> Conditional on this policy, what is the probability that this founder
> will succeed (raise $500M+ or achieve a $500M+ exit)?

## Notes

- Scoring format: probability (float in [0, 1]) extracted from the model's
  next-token logprobs over "True"/"False"
- 16 policies are included (the v25 best subset). Original generation
  produced 200+ candidate policies; these 16 were selected as the best
  ensemble.
- Source experiment: `experiments/policy_induction_vcbench/`
"""
    (prompt_dir / "policy_induction_v25.md").write_text(v25_prompt)

    taste_prompt = """\
# Taste Policy Decomposition — Scoring Prompt

Each policy was scored against each founder using Gemini 2.0 Flash. The
model returns a binary YES/NO answer.

## System message

> You are a precise classification agent. You will be given a founder's
> professional profile and a specific policy statement. Your job is to
> determine whether the founder's profile satisfies the policy.
>
> Rules:
> - Base your answer ONLY on the information provided in the profile.
> - If the profile does not contain enough information to confirm the
>   policy, answer NO.
> - Do NOT guess, infer, or use external knowledge.
> - Respond with valid JSON only: {"answer": "YES"} or {"answer": "NO"}

## User template

> POLICY: {policy}
>
> FOUNDER PROFILE:
> {profile}
>
> Does this founder's profile satisfy the policy above?
> Respond with JSON only: {"answer": "YES"} or {"answer": "NO"}

## Notes

- Scoring format: binary 0/1 (converted from YES/NO)
- 20 hand-crafted policies covering accept signals (A1-A8), reject
  signals (R1-R7), and meta signals (M1-M5)
- Source: `experiments/taste_policy_decomposition/run_taste_policies.py`
"""
    (prompt_dir / "taste_policy_decomposition.md").write_text(taste_prompt)


# ---------------------------------------------------------------------------
# Step 3: RRF questions
# ---------------------------------------------------------------------------
def build_rrf() -> None:
    """Build the rrf_questions/ subdirectory."""
    print("[rrf] Building...")
    out_dir = OUT / "rrf_questions"
    prompt_dir = out_dir / "prompts"
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    # Load source files
    questions_pq = pd.read_parquet(RRF_QUESTIONS_PARQUET)
    with open(RRF_TRAIN_SUMMARY) as f:
        train_summary = json.load(f)
    with open(RRF_PROVENANCE) as f:
        provenance = json.load(f)
    answer_matrix = pd.read_csv(RRF_ANSWER_MATRIX)

    # Identify the 20 selected questions from the answer matrix columns
    qid_cols = [c for c in answer_matrix.columns if c.startswith("Q")]
    selected_qids = [c.replace("Q", "") for c in qid_cols]
    print(f"  {len(selected_qids)} selected questions: {selected_qids}")

    # questions_pq is indexed by question id (0-39 as zero-padded strings).
    # The index is the question id. Pull rows for selected ids.
    questions_pq.index = questions_pq.index.astype(str).str.zfill(3)
    selected_rows = questions_pq.loc[selected_qids].copy()

    # Build questions.csv
    rows = []
    for qid in selected_qids:
        row = selected_rows.loc[qid]
        rows.append(
            {
                "feature_id": f"rrf_Q{qid}",
                "source": "rrf_vcbench_curated",
                "original_id": qid,
                "text": row["question"],
                "llm": "gpt-4o-mini",
                "scoring_format": "binary",
                "prompt_template_ref": "prompts/rrf_curated.md",
                "precision": float(row["precision"]) if pd.notna(row["precision"]) else None,
                "recall": float(row["recall"]) if pd.notna(row["recall"]) else None,
                "f1_score": float(row["f1_score"]) if pd.notna(row["f1_score"]) else None,
                "f_beta_0.5": float(row["f_beta_score"]) if pd.notna(row["f_beta_score"]) else None,
                "accuracy": float(row["accuracy"]) if pd.notna(row["accuracy"]) else None,
                "exclusion": str(row["exclusion"]) if pd.notna(row["exclusion"]) else None,
                "notes": "Top-20 selected from 40 curated; metrics computed on screening sample (~500 founders)",
            }
        )
    questions_df = pd.DataFrame(rows)
    questions_df.to_csv(out_dir / "questions.csv", index=False)
    print(f"  wrote questions.csv ({len(questions_df)} rows)")

    # Train predictions: reorder to canonical UUID order
    train_uuids = (OUT / "splits" / "train_uuids.txt").read_text().strip().split("\n")
    answer_matrix = answer_matrix.set_index("founder_uuid").reindex(train_uuids).reset_index()
    # Rename Q000 -> rrf_Q000
    answer_matrix.columns = ["founder_uuid"] + [f"rrf_Q{c.replace('Q','')}" for c in qid_cols]
    answer_matrix.to_csv(out_dir / "predictions_train.csv", index=False)
    print(f"  wrote predictions_train.csv {answer_matrix.shape}")

    # Test predictions: NaN matrix
    test_uuids = (OUT / "splits" / "test_uuids.txt").read_text().strip().split("\n")
    test_pred = pd.DataFrame({"founder_uuid": test_uuids})
    for col in answer_matrix.columns[1:]:
        test_pred[col] = np.nan
    test_pred.to_csv(out_dir / "predictions_test.csv", index=False)
    print(f"  wrote predictions_test.csv {test_pred.shape} (all NaN)")

    # Prompt template
    qgen_template = provenance.get("qgen_instructions_template", "(not stored)")
    prompt_md = f"""\
# RRF Curated — Question Generation & Scoring Prompts

## Question generation (qgen)

- LLM: {provenance['qgen_llmc'][0]['provider']}/{provenance['qgen_llmc'][0]['model']}
- Temperature: {provenance['qgen_temperature']}

### Generation prompt template

```
{qgen_template}
```

## Question scoring (qanswer)

- LLM: {provenance['qanswer_llmc'][0]['provider']}/{provenance['qanswer_llmc'][0]['model']}
- Temperature: {provenance['qanswer_temperature']}

### Standard system message (from think_reason_learn/rrf/_prompt_presets.py)

> You are a VC analyst evaluating founders. Your task is to decide whether
> the provided question applies to each founder based on their anonymised
> summary. Return your answer as 'Yes' or 'No'.

### User template

> Given the following anonymised founder summary, answer the question
> concisely as 'Yes' or 'No'.
>
> **Question:** {{question}}
>
> **Sample:** {{founder_profile}}

## Notes

- Scoring format: binary YES/NO (1/0 in the predictions CSV)
- Generation: 40 candidate questions (mix of v1 and v2 hand-curated),
  39 retained after semantic dedup, top-20 selected by F-beta after a
  ~500-founder screening run.
- Source experiment: `experiments/rrf_vcbench/`
"""
    (prompt_dir / "rrf_curated.md").write_text(prompt_md)


# ---------------------------------------------------------------------------
# Step 4: HQ baseline
# ---------------------------------------------------------------------------
def build_hq() -> None:
    """Re-extract Joel's 28 HQ features for both train and test sets."""
    print("[hq] Building...")
    out_dir = OUT / "hq_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    from hq_features import extract_features  # noqa: E402

    HQ_FEATURES = [
        "has_prior_ipo", "has_prior_acquisition", "exit_count",
        "max_company_size_before_founding", "prestige_sacrifice_score",
        "years_in_large_company", "comfort_index", "founding_timing",
        "edu_prestige_tier", "field_relevance_score", "prestige_x_relevance",
        "degree_level", "stem_flag", "best_degree_prestige",
        "max_seniority_reached", "seniority_is_monotone",
        "company_size_is_growing", "restlessness_score",
        "founding_role_count", "longest_founding_tenure",
        "industry_pivot_count", "industry_alignment",
        "total_inferred_experience", "is_serial_founder",
        "exit_x_serial", "sacrifice_x_serial",
        "industry_prestige_penalty", "persistence_score",
    ]

    train_uuids = (OUT / "splits" / "train_uuids.txt").read_text().strip().split("\n")
    test_uuids = (OUT / "splits" / "test_uuids.txt").read_text().strip().split("\n")

    for split, src_csv, uuids in [("train", PUBLIC_CSV, train_uuids), ("test", PRIVATE_CSV, test_uuids)]:
        print(f"  extracting {split}...")
        df = pd.read_csv(src_csv)
        df = extract_features(df)
        df = df[["founder_uuid", "success"] + HQ_FEATURES]
        df = df.set_index("founder_uuid").reindex(uuids).reset_index()
        df["success"] = df["success"].astype(int)
        # float_format="%.17g" preserves full float64 precision across CSV
        # roundtrip. Without this, persistence_score (a division result)
        # loses ~1e-16 precision, which is enough to perturb XGBoost with
        # subsample<1.0 and shift downstream F0.5 by ~0.01.
        df.to_csv(out_dir / f"features_{split}.csv", index=False,
                  float_format="%.17g")
        print(f"  wrote features_{split}.csv {df.shape}")


# ---------------------------------------------------------------------------
# Step 4b: LLM-Engineered features (from Joel's internship, set_05)
# ---------------------------------------------------------------------------
def build_llm_eng() -> None:
    """Collate the 17 LLM-engineered features (set_05) into the repository.

    Source: Vela_internship_TRL/LLM_Reasoning_Main/docs/paper_stats/features/
    llm_engineered/set_05/ — if absent, this step is skipped with a warning.
    """
    print("[llm_eng] Building...")
    out_dir = OUT / "llm_engineering"

    # If the sibling repo isn't available, skip
    if not LLM_ENG_POOL_PARQUET.exists():
        print(f"  WARNING: {LLM_ENG_POOL_PARQUET} not found — skipping LLM-eng")
        print(f"  (clone Vela_internship_TRL next to this repo to enable)")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load source files
    pool = pd.read_parquet(LLM_ENG_POOL_PARQUET)
    test = pd.read_parquet(LLM_ENG_TEST_PARQUET)
    with open(LLM_ENG_META_JSON) as f:
        meta = json.load(f)
    with open(LLM_ENG_SEED_JSON) as f:
        seed = json.load(f)

    print(f"  pool: {pool.shape}, test: {test.shape}")
    print(f"  {pool.shape[1]} LLM-engineered features (gpt-4.1-nano)")

    # Load the raw public/private CSVs to get founder_uuid order
    public_df = pd.read_csv(PUBLIC_CSV)
    private_df = pd.read_csv(PRIVATE_CSV)

    # Map pool rows to founder_uuids (skip the 100 seed indices)
    seed_indices = set(seed["indices"])
    pool_mask = [i not in seed_indices for i in range(len(public_df))]
    pool_uuids = public_df.loc[pool_mask, "founder_uuid"].values
    assert len(pool) == len(pool_uuids), (
        f"pool rows ({len(pool)}) != non-seed rows ({len(pool_uuids)})"
    )
    pool = pool.copy()
    pool["founder_uuid"] = pool_uuids

    # Test rows correspond to private_df rows directly
    test = test.copy()
    test["founder_uuid"] = private_df["founder_uuid"].values

    # Rename feature columns with le_ prefix
    feature_cols = [c for c in pool.columns if c != "founder_uuid"]
    rename_map = {c: f"le_{c}" for c in feature_cols}
    pool = pool.rename(columns=rename_map)
    test = test.rename(columns=rename_map)
    le_cols = [f"le_{c}" for c in feature_cols]

    # Align to canonical UUID orders. For train, the 100 seed founders will
    # have NaN feature values (they were reserved as LLM seed examples and
    # not scored). For test, all 4500 founders are present.
    train_uuids = (
        (OUT / "splits" / "train_uuids.txt").read_text().strip().split("\n")
    )
    test_uuids = (
        (OUT / "splits" / "test_uuids.txt").read_text().strip().split("\n")
    )

    train_out = (
        pool[["founder_uuid"] + le_cols]
        .set_index("founder_uuid")
        .reindex(train_uuids)
        .reset_index()
    )
    test_out = (
        test[["founder_uuid"] + le_cols]
        .set_index("founder_uuid")
        .reindex(test_uuids)
        .reset_index()
    )

    train_out.to_csv(out_dir / "features_train.csv", index=False)
    test_out.to_csv(out_dir / "features_test.csv", index=False)
    print(f"  wrote features_train.csv {train_out.shape}  "
          f"({train_out[le_cols[0]].isna().sum()} seed rows NaN)")
    print(f"  wrote features_test.csv  {test_out.shape}")

    # Write seed UUID list so users know which rows are NaN in train
    seed_uuids = [u for u in train_uuids if u not in set(pool_uuids)]
    (out_dir / "seed_uuids.txt").write_text("\n".join(seed_uuids) + "\n")
    print(f"  wrote seed_uuids.txt ({len(seed_uuids)} founders)")

    # Write features.csv metadata (short name + fire rate + positive lift)
    labels = pd.read_csv(OUT / "splits" / "labels.csv")
    train_y = (
        labels[labels.split == "train"]
        .set_index("founder_uuid")
        .reindex(train_uuids)["success"]
        .values
    )
    # Restrict to non-seed rows for stats
    non_seed_mask = ~train_out[le_cols[0]].isna().values
    train_y_ns = train_y[non_seed_mask]
    base_rate = float(train_y_ns.mean())

    from sklearn.metrics import fbeta_score
    rows = []
    for col in le_cols:
        values = train_out[col].values[non_seed_mask].astype(int)
        fire_rate = float(values.mean())
        if values.sum() > 0:
            pos_rate_among_fired = float(train_y_ns[values == 1].mean())
            pos_lift = pos_rate_among_fired / base_rate if base_rate > 0 else float("nan")
        else:
            pos_lift = float("nan")
        f05 = float(fbeta_score(train_y_ns, values, beta=0.5, zero_division=0))
        rows.append({
            "feature_id": col,
            "source": "llm_engineering_set_05",
            "original_name": col[3:],
            "text": col[3:].replace("_", " "),
            "llm": meta.get("model", "gpt-4.1-nano"),
            "scoring_format": "binary",
            "fire_rate_train": round(fire_rate, 4),
            "pos_lift": round(pos_lift, 3) if not np.isnan(pos_lift) else None,
            "univariate_train_f0.5": round(f05, 4),
            "notes": "Binary LLM-engineered feature. 100 seed founders NaN in train.",
        })
    pd.DataFrame(rows).to_csv(out_dir / "features.csv", index=False)
    print(f"  wrote features.csv ({len(rows)} features)")


# ---------------------------------------------------------------------------
# Step 4c: Lambda policy features (from autoresearch_vcbench)
# ---------------------------------------------------------------------------
def build_lambda_policies() -> None:
    """Evaluate the 172 autoresearch lambda policy rules for train + test.

    Source: 6 JSON rule files in ``experiments/autoresearch_vcbench/cache/``.
    Each rule is a Python lambda that operates on structured founder dicts
    and returns a binary 0/1 value. We evaluate all rules on both the
    training and test founders, deduplicate by column name, and write
    aligned (4500, 172) binary feature matrices to the repository.
    """
    print("[lambda_policies] Building...")
    out_dir = OUT / "lambda_policies"

    # Check for source cache files
    missing = [
        f for f in LAMBDA_RULE_FILES
        if not (AUTORESEARCH_CACHE / f).exists()
    ]
    if missing:
        print(f"  WARNING: missing lambda rule files {missing} — skipping")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load rules from all 6 cache files
    from think_reason_learn.datasets._vcbench import (  # noqa: E402
        VCBENCH_HELPERS, load_vcbench,
    )
    from think_reason_learn.features import (  # noqa: E402
        FeatureEvaluator, Rule,
    )

    rules: list[Rule] = []
    for fn in LAMBDA_RULE_FILES:
        with open(AUTORESEARCH_CACHE / fn) as f:
            raw = json.load(f)
        for r in raw:
            rules.append(
                Rule(
                    name=r["name"],
                    description=r["description"],
                    expression=r["expression"],
                )
            )
    print(f"  loaded {len(rules)} raw lambda rules from 6 cache files")

    # Evaluate on train + test founder records
    train_records, _ = load_vcbench(str(PUBLIC_CSV))
    test_records, _ = load_vcbench(str(PRIVATE_CSV))
    evaluator = FeatureEvaluator(rules=rules, helpers=VCBENCH_HELPERS)

    df_tr = evaluator.evaluate_df(train_records)
    df_te = evaluator.evaluate_df(test_records)
    print(f"  evaluated: train {df_tr.shape}, test {df_te.shape}")
    pol_names = list(df_tr.columns)

    # Attach founder UUIDs (from the raw CSVs, matching load_vcbench order)
    train_df = pd.read_csv(PUBLIC_CSV)
    test_df = pd.read_csv(PRIVATE_CSV)
    df_tr["founder_uuid"] = train_df["founder_uuid"].values
    df_te["founder_uuid"] = test_df["founder_uuid"].values

    # Rename columns with lam_ prefix to distinguish from v25 policies
    rename_map = {c: f"lam_{c}" for c in pol_names}
    df_tr = df_tr.rename(columns=rename_map)
    df_te = df_te.rename(columns=rename_map)
    lam_cols = [f"lam_{c}" for c in pol_names]

    # Reorder to canonical UUIDs
    train_uuids = (
        (OUT / "splits" / "train_uuids.txt").read_text().strip().split("\n")
    )
    test_uuids = (
        (OUT / "splits" / "test_uuids.txt").read_text().strip().split("\n")
    )
    train_out = (
        df_tr[["founder_uuid"] + lam_cols]
        .set_index("founder_uuid")
        .reindex(train_uuids)
        .reset_index()
    )
    test_out = (
        df_te[["founder_uuid"] + lam_cols]
        .set_index("founder_uuid")
        .reindex(test_uuids)
        .reset_index()
    )

    train_out.to_csv(out_dir / "predictions_train.csv", index=False)
    test_out.to_csv(out_dir / "predictions_test.csv", index=False)
    print(f"  wrote predictions_train.csv {train_out.shape}")
    print(f"  wrote predictions_test.csv  {test_out.shape}")

    # Compute per-feature metadata: fire rate, lift, L1 rank (HQ-based)
    labels = pd.read_csv(OUT / "splits" / "labels.csv")
    y_train = (
        labels[labels.split == "train"]
        .set_index("founder_uuid")
        .reindex(train_uuids)["success"]
        .values
    )

    # L1 ranking on HQ + lambda (matches the autoresearch experiment)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import fbeta_score

    hq_train = pd.read_csv(OUT / "hq_baseline" / "features_train.csv")
    HQ_FEATURES = [
        c for c in hq_train.columns if c not in ("founder_uuid", "success")
    ]
    X_hq = hq_train[HQ_FEATURES].fillna(0).values.astype(float)
    X_lam = train_out[lam_cols].values.astype(float)
    X_comb = np.column_stack([X_hq, X_lam])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_comb)
    lr = LogisticRegression(
        C=0.05, penalty="l1", solver="liblinear",
        class_weight="balanced", max_iter=2000, random_state=42,
    )
    lr.fit(X_scaled, y_train)
    coefs = lr.coef_[0]
    n_hq = len(HQ_FEATURES)

    # Map: column_idx -> L1 rank (1-based), NaN if not a survivor
    pol_coefs = [(i, abs(coefs[n_hq + i])) for i in range(len(lam_cols))]
    survivors = sorted(
        [(i, c) for i, c in pol_coefs if c > 1e-6],
        key=lambda x: -x[1],
    )
    rank_map = {i: r + 1 for r, (i, _) in enumerate(survivors)}

    base_rate = float(y_train.mean())
    rows = []
    for i, name in enumerate(lam_cols):
        values = X_lam[:, i].astype(int)
        fire_rate = float(values.mean())
        if values.sum() > 0:
            pos_rate = float(y_train[values == 1].mean())
            lift = pos_rate / base_rate if base_rate > 0 else float("nan")
        else:
            lift = float("nan")
        f05 = float(
            fbeta_score(
                y_train, values, beta=0.5, zero_division=0,
            )
        )
        rows.append({
            "feature_id": name,
            "source": "autoresearch_lambda_decomposition",
            "original_name": name[4:],
            "text": name[4:].replace("_", " "),
            "scoring_format": "binary",
            "fire_rate_train": round(fire_rate, 4),
            "pos_lift": round(lift, 3) if not np.isnan(lift) else None,
            "univariate_train_f0.5": round(f05, 4),
            "l1_rank_hq_based": rank_map.get(i),
            "abs_l1_coef_hq_based": round(
                abs(coefs[n_hq + i]), 4,
            ),
            "notes": (
                "Binary Python lambda on founder dict. "
                "l1_rank_hq_based: rank among 172 when L1 is fit "
                "on HQ + lambda training features (C=0.05)."
            ),
        })
    pd.DataFrame(rows).to_csv(out_dir / "features.csv", index=False)
    print(
        f"  wrote features.csv ({len(rows)} features, "
        f"{len(survivors)} L1 survivors)"
    )


# ---------------------------------------------------------------------------
# Step 5: similarity lookup
# ---------------------------------------------------------------------------
def build_similarity() -> None:
    """Build SentenceTransformer embeddings + lookup utility."""
    print("[similarity] Building...")
    out_dir = OUT / "similarity_lookup"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collate all LLM-evaluated feature texts
    policies = pd.read_csv(OUT / "policies" / "policies.csv")
    questions = pd.read_csv(OUT / "rrf_questions" / "questions.csv")
    llm_eng_path = OUT / "llm_engineering" / "features.csv"
    llm_eng = pd.read_csv(llm_eng_path) if llm_eng_path.exists() else None

    rows = []
    for _, r in policies.iterrows():
        rows.append({
            "feature_id": r["feature_id"],
            "source": r["source"],
            "type": "policy",
            "text": r["text"],
        })
    for _, r in questions.iterrows():
        rows.append({
            "feature_id": r["feature_id"],
            "source": r["source"],
            "type": "rrf_question",
            "text": r["text"],
        })
    if llm_eng is not None:
        for _, r in llm_eng.iterrows():
            rows.append({
                "feature_id": r["feature_id"],
                "source": r["source"],
                "type": "llm_engineered",
                "text": r["text"],
            })
    feature_index = pd.DataFrame(rows)
    feature_index["row_idx"] = range(len(feature_index))
    feature_index.to_csv(out_dir / "feature_index.csv", index=False)
    print(f"  feature_index.csv: {len(feature_index)} entries")

    # Compute embeddings — try sentence-transformers first, fall back to TF-IDF
    texts = feature_index["text"].tolist()
    backend, embeddings = _compute_embeddings(texts)
    np.save(out_dir / "embeddings.npy", embeddings.astype(np.float32))
    (out_dir / "embedding_backend.txt").write_text(backend + "\n")
    print(f"  embeddings.npy: shape {embeddings.shape}, dtype {embeddings.dtype}")
    print(f"  backend: {backend}")


def _compute_embeddings(texts: list[str]) -> tuple[str, np.ndarray]:
    """Compute embeddings, preferring sentence-transformers, fall back to TF-IDF."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        print("  using sentence-transformers (all-mpnet-base-v2)...")
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        emb = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return "sentence-transformers/all-mpnet-base-v2", emb
    except ImportError:
        print("  sentence-transformers not installed; using TF-IDF fallback")
        print("  (install sentence-transformers for higher-quality semantic matches)")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            min_df=1,
            max_df=0.95,
        )
        emb = vec.fit_transform(texts).toarray().astype(np.float32)
        emb = normalize(emb, norm="l2", axis=1)
        # Save the vectorizer vocabulary so the lookup script can transform new text
        (Path(__file__).parent / "similarity_lookup" / "tfidf_vocab.json").write_text(
            json.dumps(
                {
                    "vocabulary": {k: int(v) for k, v in vec.vocabulary_.items()},
                    "idf": vec.idf_.tolist(),
                    "ngram_range": list(vec.ngram_range),
                }
            )
        )
        return "tfidf", emb


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify() -> None:
    """Run shape, alignment, and reproducibility checks."""
    print("\n[verify] Running checks...")
    errors = []

    # Shape checks
    train_uuids = (OUT / "splits" / "train_uuids.txt").read_text().strip().split("\n")
    test_uuids = (OUT / "splits" / "test_uuids.txt").read_text().strip().split("\n")
    if len(train_uuids) != 4500:
        errors.append(f"train_uuids.txt has {len(train_uuids)} (expected 4500)")
    if len(test_uuids) != 4500:
        errors.append(f"test_uuids.txt has {len(test_uuids)} (expected 4500)")

    pol_train = pd.read_csv(OUT / "policies" / "predictions_train.csv")
    pol_test = pd.read_csv(OUT / "policies" / "predictions_test.csv")
    if pol_train.shape != (4500, 37):
        errors.append(f"policies/predictions_train.csv shape {pol_train.shape} (expected (4500, 37))")
    if pol_test.shape != (4500, 37):
        errors.append(f"policies/predictions_test.csv shape {pol_test.shape} (expected (4500, 37))")

    rrf_train = pd.read_csv(OUT / "rrf_questions" / "predictions_train.csv")
    rrf_test = pd.read_csv(OUT / "rrf_questions" / "predictions_test.csv")
    if rrf_train.shape != (4500, 21):
        errors.append(f"rrf predictions_train shape {rrf_train.shape} (expected (4500, 21))")
    if rrf_test.shape != (4500, 21):
        errors.append(f"rrf predictions_test shape {rrf_test.shape} (expected (4500, 21))")

    hq_train = pd.read_csv(OUT / "hq_baseline" / "features_train.csv")
    hq_test = pd.read_csv(OUT / "hq_baseline" / "features_test.csv")
    if hq_train.shape != (4500, 30):
        errors.append(f"hq features_train shape {hq_train.shape} (expected (4500, 30))")
    if hq_test.shape != (4500, 30):
        errors.append(f"hq features_test shape {hq_test.shape} (expected (4500, 30))")

    # UUID alignment
    if list(pol_train["founder_uuid"]) != train_uuids:
        errors.append("policies predictions_train.csv UUID order mismatch")
    if list(pol_test["founder_uuid"]) != test_uuids:
        errors.append("policies predictions_test.csv UUID order mismatch")
    if list(hq_train["founder_uuid"]) != train_uuids:
        errors.append("hq features_train UUID order mismatch")

    # Label integrity
    labels = pd.read_csv(OUT / "splits" / "labels.csv")
    train_pos = int(labels[labels.split == "train"]["success"].sum())
    test_pos = int(labels[labels.split == "test"]["success"].sum())
    print(f"  labels: train +{train_pos}/{len(train_uuids)}, test +{test_pos}/{len(test_uuids)}")

    # Embeddings shape: 36 policies + 20 RRF questions + 17 LLM-eng (if present) = 73
    emb_path = OUT / "similarity_lookup" / "embeddings.npy"
    if emb_path.exists():
        emb = np.load(emb_path)
        llm_eng_path = OUT / "llm_engineering" / "features.csv"
        expected = 73 if llm_eng_path.exists() else 56
        if emb.shape[0] != expected:
            errors.append(
                f"embeddings.npy shape {emb.shape} (expected {expected} rows)"
            )
        else:
            print(f"  embeddings: {emb.shape}")

    # Sanity check: reproduce the documented v25 F0.5 (~0.294-0.31) using the
    # exact pipeline from evaluate_policy_on_test.py. See SCORE_DISCREPANCIES.md
    # for context. This both validates data integrity AND confirms we're using
    # the canonical recreated_preds_* files (which reproduce the dashboard #).
    print("  sanity-checking v25 F0.5 reproduction (documented ~0.294-0.31)...")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import fbeta_score
        from sklearn.model_selection import StratifiedKFold

        v25_cols = [c for c in pol_train.columns if c.startswith("v25_")]
        v25_cols = sorted(v25_cols)  # stable order to match source script
        X = pol_train[v25_cols].values.astype(float)
        y = labels[labels.split == "train"].set_index("founder_uuid").reindex(train_uuids)["success"].values

        # Standardise using train mean/std (same as source script)
        train_mean = X.mean(axis=0)
        train_std = X.std(axis=0)
        train_std[train_std == 0] = 1.0
        X_z = (X - train_mean) / train_std

        # 5-fold stratified, LR L2 C=0.1 max_iter=3000 solver=lbfgs
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.full(len(y), np.nan)
        for ti, vi in skf.split(X_z, y):
            m = LogisticRegression(
                penalty="l2", C=0.1, max_iter=3000,
                random_state=42, solver="lbfgs",
            )
            m.fit(X_z[ti], y[ti])
            oof[vi] = m.predict_proba(X_z[vi])[:, 1]

        # Threshold sweep 0.05 to 0.95 step 0.01
        best_f, best_t = 0.0, 0.5
        for t in np.arange(0.05, 0.95, 0.01):
            f = fbeta_score(y, (oof >= t).astype(int), beta=0.5, zero_division=0)
            if f > best_f:
                best_f, best_t = f, float(t)
        print(f"    v25 OOF F0.5 reproduced = {best_f:.4f} at threshold={best_t:.2f}")
        print(f"    (documented: 0.294-0.31; tolerance ±0.03)")

        # Accept the reproduction if it's in the 0.27-0.33 range
        if best_f < 0.27 or best_f > 0.34:
            errors.append(
                f"v25 F0.5 reproduction out of expected range: {best_f:.4f}"
                f" (expected ~0.29-0.31)"
            )
    except Exception as e:
        errors.append(f"v25 reproduction failed: {e}")

    # Sanity check: reproduce HQ F0.5 ~0.274 on test
    print("  sanity-checking HQ F0.5 reproduction on test...")
    try:
        from sklearn.metrics import fbeta_score
        from sklearn.model_selection import StratifiedKFold
        from xgboost import XGBClassifier

        HQ_FEATURES = [c for c in hq_train.columns if c not in ("founder_uuid", "success")]
        X_tr = hq_train[HQ_FEATURES].fillna(0.0).values
        y_tr = hq_train["success"].values
        X_te = hq_test[HQ_FEATURES].fillna(0.0).values
        y_te = hq_test["success"].values
        ex_tr = hq_train["exit_count"].values
        ex_te = hq_test["exit_count"].values

        # Joel's exact XGB
        m = XGBClassifier(
            n_estimators=227, max_depth=1, learning_rate=0.0674,
            subsample=0.949, colsample_bytree=0.413, scale_pos_weight=10,
            min_child_weight=14, gamma=4.19, reg_alpha=0.73, reg_lambda=15.0,
            eval_metric="logloss", random_state=42, n_jobs=1,
        )
        # OOF threshold from 5-fold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.full(len(y_tr), np.nan)
        for ti, vi in skf.split(X_tr, y_tr):
            mm = XGBClassifier(
                n_estimators=227, max_depth=1, learning_rate=0.0674,
                subsample=0.949, colsample_bytree=0.413, scale_pos_weight=10,
                min_child_weight=14, gamma=4.19, reg_alpha=0.73, reg_lambda=15.0,
                eval_metric="logloss", random_state=42, n_jobs=1,
            )
            mm.fit(X_tr[ti], y_tr[ti])
            r = mm.predict_proba(X_tr[vi])[:, 1]
            r[ex_tr[vi] > 0] = 1.0
            oof[vi] = r
        best_t, best_f = 0.5, 0.0
        for t in np.arange(0.05, 0.95, 0.01):
            f = fbeta_score(y_tr, (oof >= t).astype(int), beta=0.5, zero_division=0)
            if f > best_f:
                best_f, best_t = f, float(t)

        m.fit(X_tr, y_tr)
        probs = m.predict_proba(X_te)[:, 1]
        probs[ex_te > 0] = 1.0
        fold_f = []
        for i in range(3):
            s, e = i * 1500, (i + 1) * 1500
            yf = y_te[s:e]
            pf = probs[s:e]
            fold_f.append(fbeta_score(yf, (pf >= best_t).astype(int), beta=0.5, zero_division=0))
        hq_test_f = float(np.mean(fold_f))
        print(f"    HQ test F0.5 (3-fold mean) = {hq_test_f:.4f} (target ~0.274)")
        if abs(hq_test_f - 0.274) > 0.030:
            errors.append(f"HQ test F0.5 reproduction off: {hq_test_f:.4f} vs ~0.274")
    except Exception as e:
        errors.append(f"HQ reproduction failed: {e}")

    if errors:
        print("\n[verify] FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\n[verify] PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Build the feature repository.")
    parser.add_argument(
        "--step",
        choices=[
            "splits", "policies", "rrf", "hq", "llm_eng",
            "lambda_policies", "similarity", "verify", "all",
        ],
        default="all",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    steps = {
        "splits": build_splits,
        "policies": build_policies,
        "rrf": build_rrf,
        "hq": build_hq,
        "llm_eng": build_llm_eng,
        "lambda_policies": build_lambda_policies,
        "similarity": build_similarity,
        "verify": verify,
    }

    if args.step == "all":
        for name in [
            "splits", "policies", "rrf", "hq", "llm_eng",
            "lambda_policies", "similarity", "verify",
        ]:
            steps[name]()
    else:
        steps[args.step]()


if __name__ == "__main__":
    main()
