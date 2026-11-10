# Selected reviewer training extraction

The reviewer directory uses the explicitly selected existing Condition C YAML:
batch 64, no warmup, cosine decay starting at epoch 200, distillation weight
0.6 -> 6 over one epoch, and RVQ regularization ramped over one epoch.
The unchanged YAML specifies 10,000 training epochs, face weight 2.5, original
released regional codecs, and codebook-step-specific kinematic weights.

The supplied draft supplement S2 currently describes a different recipe:
batch 128, 50-epoch conditioning warmup, step learning-rate decay, distillation
0.1 -> 1 starting at epoch 50 over 15 epochs, and a 50-epoch RVQ ramp starting
at epoch 50. It also describes AdamW default momentum parameters, whereas the
existing parser supplies betas (0.9, 0.95). Reconcile these descriptions with
the actual experiment used for reported results. No manuscript was edited.

This extraction preserves the selected code and configuration; it does not
establish which historical checkpoint produced a paper table. For exact-run
reproduction, retain that run's resolved config, checkpoint, codec hashes,
dataset split/cache provenance and environment versions.

The archive excludes data, model weights, experiment logs, local account
configuration, this author note, and the unrelated analysis report. It retains
upstream attribution and licenses. No training job or upload was started.
