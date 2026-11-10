# Extraction validation

Checked on 2026-09-26 using Python 3.12.13 and PyTorch 2.9.0+cu130 on Windows.

- Training entry point `--help`: passed, with the extracted package imports.
- BEATX cache builder `--help`: passed.
- Existing Condition C model checks: 12 passed.
- Existing Condition C trainer checks: 8 passed.
- Existing RVQ depth regularization smoke checks: passed.
- All extracted Python files parsed successfully.
- Resolved selected configuration matched the original project, excluding the
  configuration filename and timestamp-derived output name.
- Learning-rate schedule: 1e-4 at epochs 0, 1 and 200; 5.05e-5 at epoch 5100;
  1e-6 at epoch 10000.
- Distillation weights: 0.6 at epoch 0 and 6.0 at epoch 1.
- RVQ ramp: 0 at the start, 0.5 halfway through epoch 0, and 1 at epoch 1.
- All 84 source-manifest entries matched their recorded hashes. 82 files were
  copied byte-for-byte; two package import registries were narrowed.
- The 60 loaded local modules inspected during the configuration/schedule
  check resolved inside this extraction, rather than the original repository.

No complete training run, fresh environment installation, dataset download or
checkpoint download was performed. The original dataset/cache and codec assets
are required for full training. The checks establish extraction integrity and
synthetic behavior, not convergence or reproduction of reported paper scores.
