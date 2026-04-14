# RRF Questions

Binary YES/NO questions generated and scored via the Random Rule Forest (RRF)
methodology. Each "feature" is a single question; an LLM evaluates whether
the question applies to a given founder.

## What's in here

| File | Shape | Description |
|------|-------|-------------|
| `questions.csv` | (20, 14) | Metadata for the 20 selected questions: text, llm, precision, recall, F-beta, etc. |
| `predictions_train.csv` | (4500, 21) | `founder_uuid` + 20 binary question columns. Aligned with `../splits/train_uuids.txt`. |
| `predictions_test.csv` | (4500, 21) | **All NaN** — test predictions were not cached. |
| `prompts/rrf_curated.md` | — | Question generation + scoring prompts |

## The source

Source experiment: `experiments/rrf_vcbench/results/vcbench_rrf_curated/`

The RRF pipeline:
1. **Generate** ~40 candidate yes/no questions about founders (mix of v1
   and v2 hand-curated questions in this set)
2. **Semantic dedup** to 39 questions (Q039 was removed for being too
   similar to another)
3. **Screen** by running each question on a ~500-founder sample and
   computing F-beta (0.5)
4. **Select** the top 20 questions for full scoring

The selected 20 questions cover education (top-10/top-50 QS, STEM, technical
degrees), career level (C-level, board, VP), domain experience (VC/PE,
biotech, financial services), and trajectory (rapid progression, advanced
degree depth).

## Coverage

- **Train**: full coverage (4500 founders × 20 questions, all binary)
- **Test**: NOT cached. The original RRF run only scored the training set.
  Re-running on the test set would require approximately 90,000 LLM calls
  (4500 founders × 20 questions × 1 call each). The metadata, questions, and
  prompts are documented for reference, but `predictions_test.csv` contains
  only NaN values.

## Performance

The 20 questions individually have univariate F-beta (0.5) in the range
0.107 – 0.191. The strongest are education questions (Q000 top-10 QS,
Q024 advanced degree top-50 QS) and VC/PE experience (Q008, Q025).

These metrics come from the original screening sample (~500 founders), not
the full 4500. They are stored in `questions.csv` for reference but should
be treated as approximate.

## LLM and prompts

- **Question generation LLM**: gpt-4o-mini, temperature=0
- **Scoring LLM**: gpt-4o-mini, temperature=0
- **Total LLM calls**: 175,500 (for the curated 39-question pre-screening
  run on the full 4500 founders)
- **Total tokens**: 47.3M

See `prompts/rrf_curated.md` for the full system messages and templates.

## Quick load example

```python
import pandas as pd

questions = pd.read_csv("questions.csv")
train = pd.read_csv("predictions_train.csv")

# Find the strongest single question
top = questions.sort_values("f_beta_0.5", ascending=False).head(5)
print(top[["feature_id", "text", "f_beta_0.5"]])

# Use the binary features in a model
v25_pol = pd.read_csv("../policies/predictions_train.csv")
combined = train.merge(v25_pol, on="founder_uuid")
# combined.shape == (4500, 1 + 20 + 36)
```
