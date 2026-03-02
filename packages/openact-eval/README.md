# openact-eval

Evaluation and labeling toolkit for OpenAct runs.

- Parser-based evaluators (GSM8K, MATH, MMLU, etc.)
- LLM-as-judge and safety evaluators (optional extras)
- Metrics, pipelines, and label export (Parquet)

Install: `pip install -e .` (from this directory). Optional: `pip install -e ".[safety]"` or `.[llm]`.
