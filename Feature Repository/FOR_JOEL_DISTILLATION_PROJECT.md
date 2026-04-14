# For Joel: Distillation Project Guide

This file is written specifically for Joel's policy-teacher distillation
project. It answers the orientation questions from your Slack messages,
explains where to find the right policy set, and frames two concrete
project directions you can pick between.

## Direct answers to your questions

### "I found 48 policy features on train and 10 on test. Are they the right ones?"

Probably not. The 10-policy set you found on `founder-success-policy-induction`
is from an **earlier** version of the pipeline (policies 36, 16, 5, 30,
45, 4, 26, 18, 7, 35). The best-performing policy set on VCBench is a
**different 16-policy ensemble** called `v25` (policies 1, 11, 38, 52,
55, 58, 72, 80, 112, 116, 121, 135, 143, 150, 157, 161).

The v25 set was selected by bootstrap significance testing from a pool
of ~160 candidate policies as the best ensemble, and achieves test
F0.5 = **0.290** as a standalone approach (vs the HQ baseline of 0.273).
This is the "strong teacher" you should distil.

**Where to find them in this repo**:

| File | What's in it |
|------|-------------|
| [`policies/policies.csv`](policies/policies.csv) | All 36 policies (16 v25 + 20 Taste). Filter by `source == "policy_induction_v25"` to get just the 16 you want. Has full text, short name, L2 ensemble coefficient, per-policy univariate CV F0.5, and pointbiserial r with success. |
| [`policies/predictions_train.csv`](policies/predictions_train.csv) | 4500 training founders × 16 v25 columns (plus 20 Taste columns). UUID-aligned with `splits/train_uuids.txt`. |
| [`policies/predictions_test.csv`](policies/predictions_test.csv) | Same shape for test. |
| [`policies/prompts/policy_induction_v25.md`](policies/prompts/policy_induction_v25.md) | The scoring prompt. The original *text-generation* prompt (the one beginning "You are an expert venture capital strategist...") lives in the `policy_induction_vcbench` experiment folder of the main think-reason-learn repo. Ask Ben if you need the exact text. |

### "Are the values scaled 0–1?"

Yes. The v25 scores are **continuous probabilities in [0, 1]**, computed
from Gemini 2.5 Flash next-token logprobs over True/False. They represent
`P(success | policy applies, founder)`, NOT whether the policy matches the
profile. (The column names are `v25_p1`, `v25_p11`, ... `v25_p161`.)

The 20 Taste policies in the same file are instead **binary 0/1** (YES/NO
from Gemini). They're a separate source and you can ignore them for the
distillation project if you only want v25.

### "Where are the prompts?"

Scoring prompt: [`policies/prompts/policy_induction_v25.md`](policies/prompts/policy_induction_v25.md)
(this is what gets the Gemini probability score for each
founder×policy pair).

The original policy *text-generation* prompt (the one beginning "You
are an expert venture capital strategist...") isn't included in this
feature-repository bundle — it lives in the main think-reason-learn
repo under `experiments/policy_induction_vcbench/`. If you need it,
ping Ben and he'll send it over.

## Before you start: reproduce the baselines

Before doing any distillation, run the full reproduction guide to make
sure you can reproduce all 9 headline scores from this repo:

```bash
# Copy-paste the Python script from REPRODUCING_HEADLINE_SCORES.md
# It reproduces all 9 of the following on your machine in ~5 minutes.
```

See [`REPRODUCING_HEADLINE_SCORES.md`](REPRODUCING_HEADLINE_SCORES.md)
for the exact script. You should get:

| # | Configuration | # Feat | Test F0.5 |
|---|---|---:|---:|
| 1 | HQ only | 28 | 0.273 |
| 2 | LLM Engineering only | 17 | 0.284 |
| 3 | **Policy Induction only (v25)** | **16** | **0.290** |
| 4 | HQ + top-30 lambda policy | 58 | 0.284 |
| 5 | HQ + top-40 lambda policy | 68 | 0.293 |
| 6 | **HQ + Policy Induction (v25)** | **44** | **0.300** |
| 7 | LLM Engineering + top-30 lambda | 47 | 0.268 |
| 8 | LLM Engineering + top-40 lambda | 57 | 0.283 |
| 9 | **LLM Engineering + Policy Induction (v25)** | **33** | **0.334** |

The bolded ones (3, 6, 9) are the ones you care about for distillation.

## Policy decomposition — a baseline you need to know about

Before proposing distillation, I should explain the "lambda policy" rows
(configs 4, 5, 7, 8) because they're a separate approach I tried that
partially overlaps with what distillation would do.

**Policy decomposition** is: take each v25 policy (natural-language text),
ask an LLM to decompose it into 3–6 concrete Python lambda features
(e.g., *"founder has an MBA"*, *"founder has a prior IPO AND an MBA"*),
then evaluate those lambdas deterministically without any LLM calls at
inference time. The key idea is that LLM policies often naturally break
into conjunctions like *"elite education AND prior exit"* — which become
testable binary rules.

I ran this on VCBench with 172 total lambda features from 6 LLMs and
auto-selected subsets via L1 ranking. The short paper is included in
this bundle at
[`reference_docs/POLICY_DECOMPOSITION_REPORT.pdf`](reference_docs/POLICY_DECOMPOSITION_REPORT.pdf).

**The relevance for your distillation project**:

- Policy decomposition is **effectively a manual distillation of
  policies into interpretable features** — it does the compression by
  having an LLM author the features once, then uses pure Python at
  inference.
- Your distillation project should be framed as "learn this compression
  end-to-end" instead of "have an LLM hand-write it." The comparison
  between distillation and decomposition is itself a useful experimental
  finding.

## Two project framings — pick one

You have two distinct framings for this project. Both are valid. The
second is more interesting but harder to make work.

### Option A: HQ-only (no LLM-Engineering)

- **Baseline**: HQ only = 0.273 (config 1)
- **Ceiling**: HQ + Policy Induction = 0.300 (config 6)
- **Gap to close**: 0.027 F0.5 without paying for LLM inference

**Already-known alternative**: Policy decomposition (config 5: HQ +
top-40 lambda) already reaches 0.293, closing 75% of the gap. If all
you care about is the HQ-only world, policy decomposition is already a
pretty good solution.

**Why distillation is still interesting in this framing**:

1. Policy decomposition is unpublished and relies on a fairly manual
   prompt-engineering process with the LLM authoring the lambdas. It's
   not a principled compression method.
2. A distilled student (LR on HQ features → predicted v25 scores) is
   principled: you're fitting a supervised model against a known teacher
   target, with standard ML evaluation.
3. If the student recovers, say, 80% of the 0.290 policy signal from
   HQ features alone, that's a clean empirical statement about *how
   much of the teacher's reasoning can be predicted from the profile
   features the teacher already has access to*.

**Interpretation**: if the student hits ~0.29 from HQ features alone,
it means most of the policy signal was latent in the profile and the
LLM was mostly reorganising existing information. If the student only
reaches ~0.28, it means the LLM is genuinely adding something beyond
the profile features.

### Option B: With LLM-Engineering (the more interesting framing)

- **Baseline**: LLM Engineering only = 0.284 (config 2)
- **Ceiling**: LLM Engineering + Policy Induction = 0.334 (config 9)
- **Gap to close**: 0.050 F0.5 — **almost twice the HQ gap**

**This is where policy decomposition fails**: configs 7 and 8 show that
adding top-30/40 lambda policies to LLM-Engineering doesn't help
(0.268 and 0.283). The lambda policies are redundant with whatever
LLM-Engineering already captures — both are essentially "observable
patterns in the profile data."

So policy decomposition is **not a solution** in this framing. The
+0.050 gap between LLM-Eng alone (0.284) and LLM-Eng + Policy Induction
(0.334) remains unexplained by any simple feature engineering approach
I've tried.

**Why distillation matters here**:

- The v25 policies capture something that LLM-Engineering features don't,
  and it's not just "observable profile patterns" (otherwise
  decomposition would close the gap).
- The most likely explanation: v25 uses **continuous reasoning scores**
  from Gemini logprobs that encode graded judgements about context (e.g.,
  *"how well does this founder match the Macro-market tailwinds pattern?"*)
  rather than binary pattern matches.
- A student model trained on the v25 scores (as teacher targets) would
  need to learn this graded reasoning from its inputs (HQ features,
  LLM-Eng features, profile text embeddings, etc.). That's a much
  harder compression problem than the decomposition approach — and
  correspondingly more interesting if it works.

**Clean experimental story**: show that decomposition fails (configs
7/8) but distillation partially closes the gap. Report how much of the
0.050 can be recovered, and which input view (metadata only, text only,
or both) recovers the most.

### My recommendation

**Option B** is the stronger research project because:
1. It has a larger headroom (0.050 vs 0.027).
2. It has a clear negative baseline (decomposition fails here but
   succeeds in Option A).
3. The answer is unknown — I don't currently know if a student can
   bridge this gap, which makes it a genuine research question rather
   than confirming something I already suspect.

But start with Option A as a **smoke test**: if you can reproduce
0.273 → ~0.29 with a simple distilled student on HQ features, you've
confirmed the pipeline works, and you can then attempt the harder
Option B.

## Important: use one consistent model for fair comparisons

The 9-config table in this repo mixes two XGBoost configs:
- Configs 1, 4, 5 use XGBoost
- Configs 2, 3, 6, 7, 8, 9 use nested L2 logistic regression

This mixture exists because each headline number was computed with
the model that was tuned for its feature set. But **it's a
footgun for your project**: if you compare distilled-student vs
teacher-ensemble using different models, you can't cleanly attribute
differences to the distillation itself.

**Recommendation**: for this project, use **nested L2 LogisticRegression
everywhere**, with the standard C grid:

```python
C_GRID = (0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0)
```

and inner 3-fold CV to pick C. See `nested_l2_fit()` in
[REPRODUCING_HEADLINE_SCORES.md](REPRODUCING_HEADLINE_SCORES.md) for a
reference implementation.

This will cost you ~0.01 F0.5 on the HQ baseline (0.273 → ~0.263 with
LR), but every comparison you make — teacher vs student, HQ-only vs
HQ+student, etc. — will be clean. Report that single-model baseline
as your reference point and build from there.

## Suggested order of attack

1. **Reproduce the baselines**. Run the script from
   `REPRODUCING_HEADLINE_SCORES.md` and confirm you can hit configs 1, 3,
   6, and 9 within ±0.005.
2. **Switch to LR everywhere**. Re-run configs 1, 6, 9 with nested L2 LR
   and note the new numbers — those are your reference points.
3. **Build the teacher**. Load `policies/predictions_train.csv` and
   `predictions_test.csv`; the 16 v25 columns ARE your teacher signals.
   You don't need to run policy induction yourself.
4. **Train a simple student**. LR on HQ features → predicted v25 scores.
   Evaluate: (a) teacher-student agreement (MSE or correlation on the
   16 scores), (b) downstream F0.5 when you use the predicted scores
   as features in the final classifier.
5. **Scale up the student inputs**. Try HQ + profile text embeddings
   (e.g., sentence-transformers on `anonymised_prose`), then HQ +
   LLM-Eng features.
6. **Compare to the ceiling**. Report how much of the 0.300 (Option A)
   or 0.334 (Option B) can be reached by a student that never calls
   Gemini at inference time.

## Things I'd love to know from your project

- How much of the v25 signal is recoverable from HQ features alone?
- Does adding profile text embeddings meaningfully help the student?
- Is it better to distil at the policy-score level (16 regression
  targets) or directly at the final-prediction level (1 classification
  target)?
- Does the student reveal which policies are "latent in the profile"
  vs "the LLM is genuinely inferring something new"? This would be a
  nice interpretability story for the paper.

Ping me anytime.
