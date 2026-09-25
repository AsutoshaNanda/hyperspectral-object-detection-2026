# Code

Copied as it was used. Paths inside some scripts point to my local folders or to `/workspace` on the rented boxes; change them before running. Competition data, model weights and prediction files are not included.

| Folder | What it is |
|---|---|
| [runner/submission01_yolo16M.py](runner/submission01_yolo16M.py) | Main training script (Kaggle era). PNG mosaic to 16 bands, train-only normalization, YOLO26m or RT-DETR-L via Ultralytics 8.4.147, square padding, RGB-mean 16-band stem, spectral augmentation, localization tail (with the warm-up fix), holdout gate, test + ranking CSV writer |
| [runner/generate_plan_c_notebook.py](runner/), `generate_plan_d_notebook.py` | Build the Plan C and Plan D Kaggle notebooks |
| [experiments/](experiments/) | Roadmap code: fold builder, size-AP evaluator, normalization / MNF / Savitzky-Golay / MSC, TTA / multi-scale / sliced inference / Soft-NMS, D-FINE and other model adapters, SAM diagnostic, pseudo-label tools, experiment ledger and queue, notebook generators |
| [kaggle_notebooks/](kaggle_notebooks/) | The notebooks pushed to Kaggle: fold baselines, day-1 diagnostics, inference screen, SAM1, top-10 screens, localization tails, architecture screens; plus the Colab Cascade R-CNN attempts |
| [vastai_phase2/](vastai_phase2/) | Scripts from the rented boxes: `hsi_runner.py` (the runner plus `--train-on-all`), run scripts for combo / self-training, pseudo-label builder, Mac prediction with flip TTA, Cascade R-CNN (MMDetection) config, prep and fair scorer, single-GPU setup |
| [tests/](tests/) | Unit tests for folds, preprocessing, inference, size evaluation, the runner and notebook generation |

Main settings of the best recipe (Combo): RT-DETR-L from the public COCO `rtdetr-l.pt`, 16-band input stem started from the averaged RGB weights (scaled by 3/16), square padding, 1024 px, batch 2, 50 epochs, seed 42, one spectral-augmentation copy per image (brightness up to 5%, band tilt up to 3%, noise about 1% of each band's spread), prediction with confidence floor 0.001 and up to 300 boxes per image.
