# Intermediary Features

This folder stores reusable generated feature banks derived from raw VCBench inputs.

Current canonical banks:

- `mirror/v1/`
- `sentence_transformer/prose/all-MiniLM-L6-v2/`
- `sentence_transformer/structured/all-MiniLM-L6-v2/`

Each bank should contain:

- `public.parquet`
- `private.parquet`
- `manifest.json`

Sentence-transformer banks should also store rendered text artifacts used before embedding.
