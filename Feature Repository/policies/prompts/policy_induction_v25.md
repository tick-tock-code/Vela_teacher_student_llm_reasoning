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
