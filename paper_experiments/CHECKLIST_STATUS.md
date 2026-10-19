# Release checklist status

## Implemented

- README, separated source layout, no large files/cache/private paths/plaintext keys.
- Configurable paths, main pipeline entry, named baseline wrappers and minimal example.
- Text/multimodal embedding, four native RAG retrieval systems, DirectQA/RAG generation.
- Unified question output metadata: method, model, seed, config, run ID.
- Stage and question-level resume, intermediate flush, unified JSONL.
- Conda, requirements and Docker environment definitions.
- Public model/API metadata templates, model-only download script, public prompts/parsers.
- Strict Doc/Page, Gold/Equivalent modes, deduplication, short Top-K and invalid evidence rules.
- Seven unit tests including gold hit, equivalent hit, partial miss and no alternative mixing.
- Public four-axis Judge prompts, raw output, retry, malformed output handling and aggregation.
- Papers/Patents, Text/Figure/Meta, document-count, All/Some/None and Miss-1..5 analysis.
- Per-question files, group sizes, small-sample marker, table CSV and SVG plotting code.

## Must be resolved by the repository/paper owner before public release

1. Replace the restrictive `LICENSE` selection notice with the legally approved license.
2. Replace every mutable model revision `main` with the exact camera-ready commit SHA.
3. Fill the exact external API model snapshot and call date in `configs/generation_models.yaml`.
4. Recover and record the actual MinerU version used to create released page data.
5. Run `scripts/capture_environment.sh` for every experiment and report actual GPU count,
   cuDNN, PyTorch, Transformers, vLLM, FAISS and other installed versions.
6. Add authoritative expected paper values/tolerances from the final run manifest.
7. Replace citation placeholders and semantic paper mappings with camera-ready metadata.

These values cannot be inferred safely from source code and were deliberately not invented.
