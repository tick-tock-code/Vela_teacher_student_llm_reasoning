# RRF Curated — Question Generation & Scoring Prompts

## Question generation (qgen)

- LLM: openai/gpt-4o-mini
- Temperature: 0.0

### Generation prompt template

```
None
```

## Question scoring (qanswer)

- LLM: openai/gpt-4o-mini
- Temperature: 0.0

### Standard system message (from think_reason_learn/rrf/_prompt_presets.py)

> You are a VC analyst evaluating founders. Your task is to decide whether
> the provided question applies to each founder based on their anonymised
> summary. Return your answer as 'Yes' or 'No'.

### User template

> Given the following anonymised founder summary, answer the question
> concisely as 'Yes' or 'No'.
>
> **Question:** {question}
>
> **Sample:** {founder_profile}

## Notes

- Scoring format: binary YES/NO (1/0 in the predictions CSV)
- Generation: 40 candidate questions (mix of v1 and v2 hand-curated),
  39 retained after semantic dedup, top-20 selected by F-beta after a
  ~500-founder screening run.
- Source experiment: `experiments/rrf_vcbench/`
