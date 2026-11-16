# PatchDedup development

- Use Python 3.11, the `src/` package layout and `uv`. Keep runtime dependencies light.
- Before changing behavior, write and run a failing test for the intended behavior.
  Implement it, observe green, then refactor. Commit passing tests with the implementation.
- Use small Conventional Commits in English. Keep main documentation in Portuguese.
- Validate with `uv run --locked pytest -q`, `uv run --locked ruff check .`,
  `uv run --locked ruff format --check .` and `uv build` after relevant changes.
- Read README.md for implemented scope; docs/SPECIFICATION.md includes future work.
- Current version is experimental research with real local checkpoints and inference.
  Do not present test fixtures, mocks or original base weights as a trained PatchDedup model.
  Distinguish cosine similarity from any cross-encoder decision score; neither is calibrated.
  Keep original v01 tests as historical diagnostics; v02 has a separately frozen holdout.
- Never tune on the frozen test split. Split scenarios before constructing pairs.
  Preserve material facts, provenance, source revisions and dataset hashes.
- Keep credentials, private text and weights out of Git. Hub publishing is manual,
  uses explicit files and requires verifying destination contents and revisions.
