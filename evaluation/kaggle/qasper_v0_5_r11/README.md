# QASPER v0.5 R11 Kaggle snapshot

This directory preserves the exact final Kaggle source bundle used for the
2026-08-21 QASPER development run discussed in the project documentation. It is
an immutable provenance snapshot, not the canonical implementation for future
evaluation changes.

## Contents

- `main.py`: self-contained Kaggle entry point. It embeds the application source
  used by R11, downloads the public QASPER v0.3 archive, verifies its SHA-256,
  creates an isolated pinned environment, verifies a Tesla T4, and runs the
  lexical, dense, and hybrid-plus-generation configurations.
- `kernel-metadata.json`: private Kaggle script metadata used by the Control
  Plane source bundle. The `replace-me` owner is replaced during submission.
- `source_manifest.json`: hashes recorded when the source bundle was produced.

The executed Windows `main.py` had SHA-256
`f6dc6c080374d00894b181344745abac9aeecd29425e8112c12dd0931f5dd157`.
The manifest also records hashes for the LF-normalized repository files. The
committed kernel metadata replaces the submitted Kaggle owner with `replace-me`,
so its repository hash intentionally differs from the executed metadata hash.
Restoring CRLF reproduces the recorded executed `main.py` hash.

## Recorded result

The completed R11 log reported 1,005 predictions with no missing cases:

| Configuration | Recall@5 | MRR | Evidence F1 | Answer F1 |
|---|---:|---:|---:|---:|
| Lexical, no generator | 0.4605 | 0.3072 | 0.1606 | 0.1350 |
| Dense, no generator | 0.4957 | 0.3538 | 0.1823 | 0.1350 |
| Hybrid + Qwen2.5-1.5B-Instruct | 0.5237 | 0.3886 | 0.2396 | 0.1651 |

Retrieval metrics cover 892 eligible dev cases. The hybrid generation phase
made 1,005 model calls in one batched pipeline invocation and took about 7,332
seconds; the complete job took about 8,013 seconds on a Tesla T4. These numbers
are an external development checkpoint, not a production-quality claim or the
v0.5 regression baseline.

Do not commit the generated predictions, downloaded dataset, model cache,
resolved environment, or Kaggle result artifacts. Future runs should be
submitted through Kaggle Control Plane with the explicit `NvidiaTeslaT4` shape.
