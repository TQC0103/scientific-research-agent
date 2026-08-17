# Repository working agreement

This repository treats architecture and project-state documentation as part of
the implementation. Before finishing any change, update the documents affected
by that change:

1. Update `docs/SYSTEM_VISUALIZATION.md` when a module, dependency, data flow,
   branch, retry rule, or persistence boundary changes.
2. Append to `docs/DEVELOPMENT_LOG.md` when a decision is made, a meaningful
   failure is found, or an evaluation is run.
3. Refresh `docs/PROJECT_STATE.md` after implementation: current version,
   completed capabilities, verification result, known issues, and next work.
4. Update `README.md`, `.env.example`, and `docs/ROADMAP.md` when setup, runtime
   configuration, user-facing commands, or priorities change.

Keep the overview diagram module-level only. Put internal nodes and branches in
the matching detailed module diagram. Derive diagrams from code rather than
planned behavior, label future work explicitly, and never commit runtime PDFs,
indexes, databases, evaluation outputs, credentials, or `.remember/` state.

Run Ruff and pytest before handoff, then record the actual test count in
`docs/PROJECT_STATE.md`. For batch LLM/model benchmarks, prefer the user's Kaggle
Control Plane/GPU when available; use the laptop for unit tests and smoke runs.
