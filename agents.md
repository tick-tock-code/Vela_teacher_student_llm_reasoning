Core Principles
1. No Guessing
Do not make assumptions about data, model structure, or intended functionality.
If uncertainty exists:
Pause and ask for clarification before proceeding.
Avoid silent defaults or implicit behavior.
2. Scientific Rigor
Treat all work as part of a research pipeline, not just engineering.
Ensure:
Clear inputs and outputs
Traceable transformations
Reproducible results
Prefer explicitness over abstraction when it improves interpretability.
3. Ask Before Structuring
If unsure about:
Model architecture
Data schema
Training setup
Evaluation methodology
→ Ask before implementing.

Do not “fill in gaps” with assumed best practices.

4. Continuous Refactoring
Regularly refactor to:
Simplify logic
Remove redundancy
Improve readability
Prefer:
Small, composable modules
Clear naming
Minimal coupling

Refactoring is not optional—it is part of normal workflow.

5. Simplicity First
Default to:
Simple baselines before complex models
Transparent pipelines before optimised ones
Avoid premature optimisation or overengineering.
Project Structure

Use a clear and minimal folder hierarchy:

/data/        # Reusable datasets and global data artifacts
/tmp/         # Intermediate outputs, experiment artifacts, scratch files
/docs/        # Formal summaries, reports, and interpreted results
/src/         # Core code (models, pipelines, utilities)
/experiments/ # Experiment configs, runs, and comparisons
/tests/       # Unit and integration tests
Folder Guidelines
/data/
Canonical, reusable datasets
Versioned where necessary
No temporary or partial outputs
/tmp/
Intermediate artifacts:
Model outputs
Teacher labels
Embeddings
Safe to delete or regenerate
/docs/
Human-readable outputs:
Experiment summaries
Evaluation reports
Key findings
Should interpret results from /tmp/ or /data/
/src/

Suggested substructure:

/src/
  teacher/        # Policy induction + scoring pipeline
  student/        # Distilled models
  data/           # Data loading + preprocessing
  evaluation/     # Metrics and comparisons
  utils/          # Shared utilities
/experiments/
Config-driven experiments
Each experiment should:
Declare inputs
Declare outputs
Be reproducible
Coding Guidelines
Clarity Over Cleverness
Write code that is easy to read and reason about.
Avoid hidden logic or implicit flows.
Explicit Interfaces
Clearly define:
Inputs
Outputs
Assumptions
Minimal Dependencies
Avoid unnecessary libraries.
Keep the stack lightweight unless justified.
Experimentation Standards
Every experiment should:
Be reproducible
Log configuration and results
Separate:
Data generation (teacher)
Model training (student)
Evaluation
When in Doubt

If any of the following are unclear:

Data format
Model choice
Training objective
Evaluation metric
File structure

→ Stop and ask for clarification.

Anti-Patterns (Avoid)
Silent assumptions
Hardcoded values without justification
Overly complex abstractions
Mixing experimental and production logic
Writing code before clarifying requirements