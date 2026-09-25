# Experiment configuration matrix

`experiment_matrix.json` is the frozen configuration inventory for the roadmap. It contains configuration only and no measured result.

Generate and validate it with:

```bash
python Experiments/configs/build_experiment_matrix.py
python -m pytest -q Experiments/configs/test_experiment_matrix.py
```

The matrix fixes seed 42, the verified V1 fold hashes, the Plan-C transformer control, the Plan-A YOLO control, existing augmentations, one-epoch smoke tests, ten-epoch screens, and full 50/80-epoch budgets. New architectures inherit a clearly labelled 50-epoch resource assumption until smoke evidence measures runtime and memory.

Any entry with `unresolved_references` is deliberately `runnable: false`. Bind each reference to a measured evidence record before dispatch. The three holdout slots also remain blocked because the existing legacy holdout has already been inspected; they require an unused labeled holdout or an explicit revised protocol.

All inference views must come from one checkpoint. Cross-model and cross-checkpoint prediction fusion are prohibited throughout the matrix.
