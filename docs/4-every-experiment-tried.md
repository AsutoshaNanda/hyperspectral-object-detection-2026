# 4. Every experiment I tried

One row per attempt, in order, including short tests, crashed runs and runs that never started. 90 rows in total.

How to read it:

- **Length**: epochs (full passes over the training images). "Smoke" = a tiny check that the code runs at all. "Inference" = no training, only prediction with an existing model.
- **Result**: mAP50-95, a score from 0 to 1 (higher is better), unless marked otherwise.
  - "holdout" = the shared 300 held-out labelled images.
  - "screen" = 10-epoch fold-0 validation, only comparable to the 10-epoch control (0.6201).
  - "public" = Kaggle public board, about 51% of the test images (until 25 Sep).
  - "full test" = all 1,000 test images. After the Phase 1 freeze on 25 Sep, Kaggle re-scored every submission this way, so Kaggle's page now shows these numbers instead of the old 51% ones.
- **Outcome**: Done, Failed (crashed or broken), Partly (ran but gave no usable score), Not run.

Detailed numbers (mAP50, mAP75, per-class) are in [5-results-by-location.md](5-results-by-location.md).

## A. Kaggle notebooks: first models (before 11 Sep to 19 Sep)

| # | Experiment | Length | Result | Outcome |
|---:|---|---|---|---|
| 1 | Plan A first check: YOLO26m, 16 bands | 1 epoch | Crashed: invalid CUDA device | Failed |
| 2 | Plan B v1: RT-DETR-L, 16 bands, 640 px | 1 epoch | val 0.281, public 0.242, full test 0.233 (submitted by mistake) | Done |
| 3 | Plan A recovery: YOLO26m, 16 bands, 1024 px | 80 | val 0.699, holdout 0.655, public 0.590, full test 0.581 | Done |
| 4 | Plan B v2: RT-DETR-L, 640 px, image stretched to square | 50 | val 0.666, holdout 0.637, public 0.608, full test 0.603 (submitted 3 times, same score) | Done |
| 5 | Plan D v1: Plan B v2 model, keep boxes down to 0.001 confidence, 300 per image | Inference | Crashed on a P100 GPU (CUDA kernel error) | Failed |
| 6 | Plan D v2: same, run on CPU | Inference | public 0.610, full test **0.60486**. This is the file the organizers froze as my Phase 1 score | Done |
| 7 | Plan C: RT-DETR-L, image padded to square | 50 | val 0.701, holdout 0.670, public 0.607, full test 0.599. The Kaggle session stopped after epoch 35; training was resumed from the saved checkpoint | Done |
| 8 | Plan C with 0.001 confidence floor, 300 boxes | Inference | public 0.610, full test 0.604 | Done |

## B. Roadmap diagnostics (19 to 20 Sep, Kaggle and Colab)

| # | Experiment | Length | Result | Outcome |
|---:|---|---|---|---|
| 9 | Plan C evidence audit | Check | Saved confusion matrices were all zeros, so treated as unavailable | Done |
| 10 | V1: build 3 cross-validation folds | Check | 2,697 images, every class in every fold; no capture info, so grouped per image | Done |
| 11 | Day-1 diagnostics, first run | Check | Some checkpoint diagnostics failed (a checkpoint was missing) | Failed |
| 12 | Day-1 diagnostics, repaired (D1 accuracy by object size) | Check | YOLO small/medium/large 0.664 / 0.751 / 0.536; RT-DETR-L (square pad) 0.669 / 0.739 / 0.728. Square padding beat stretching at every size | Done |
| 13 | Fold-0 control: Plan C recipe on fold 0 | 50 | val 0.689, mAP75 0.832, 5.6 GPU-hours | Done |
| 14 | Fold-1 control | 50 | Finished (about 5.7 GPU-hours); score not saved | Partly |
| 15 | Fold-2 control | 50 | Launched on Kaggle together with fold 1; its outcome and score were never recorded | Partly |
| 16 | Plan B and YOLO runs on folds 0, 1, 2 (6 notebooks) | 50 / 80 | Notebooks prepared, never started | Not run |
| 17 | T1: flip TTA (predict normal + mirrored image) on the fold-0 model | Inference | +0.0025 | Done |
| 18 | T2: multi-scale TTA | Inference | +0.0008 | Done |
| 19 | T3: flip + multi-scale | Inference | -0.0005 | Done |
| 20 | P3: sliced inference (predict on overlapping crops) | Inference | -0.112 | Done |
| 21 | T4: full image + slices | Inference | -0.070 | Done |
| 22 | SAM1: can the 16-band spectrum tell real from plastic? | Check (CPU) | AUC 0.708 apple, 0.719 egg (0.5 = guessing, 1.0 = perfect) | Done |
| 23 | D-FINE-S smoke on Colab at 128 px (both input-layer starts) | Smoke | Failed: position encoding expects 640 px | Failed |
| 24 | D-FINE-S smoke on Colab at 640 px, random 16-band input layer | Smoke | Forward, prediction, loss and backward checks passed | Done |
| 25 | D-FINE-S smoke on Colab at 640 px, RGB-expanded input layer | Smoke | Failed the prediction check | Failed |
| 26 | RT-DETRv2-R18 smoke on Colab (Hugging Face version), several tries | Smoke | Failed every time: the library rejected 16-channel input | Failed |

## C. Kaggle 10-epoch screens and the first tries of the new models (21 to 22 Sep)

All screens: RT-DETR-L, square pad, 640 px, fold 0. Pass = at least +0.005 over the control.

| # | Experiment | Length | Result | Outcome |
|---:|---|---|---|---|
| 27 | Screen control (no change) | 10 | screen 0.6201 | Done |
| 28 | S1b: start the 16-band input layer from averaged RGB weights | 10 | screen 0.6420 (+0.022) | Done |
| 29 | N1: per-band z-score normalization | 10 | screen 0.6223 (+0.002, no real gain) | Done |
| 30 | H1b: one spectral-augmentation copy of every image | 10 | screen 0.6539 (+0.034, best) | Done |
| 31 | M1d: multi-scale training 0.10 | 10 | screen 0.6273 (+0.007) | Done |
| 32 | P1: 1024 px instead of 640 | 10 | screen 0.6343 (+0.014) | Done |
| 33 | L1b: 10-epoch low-learning-rate box-tightening tail on the fold-0 model | 10 | Collapsed from 0.689 to 0.031 (bug) | Failed |
| 34 | L1c: 12-epoch tail | 12 | Ran with the same bug; result invalid | Failed |
| 35 | L1d: 15-epoch tail | 15 | Queued; Kaggle GPU quota ran out | Not run |
| 36 | N3: SNV normalization (Kaggle notebook) | 10 | Prepared as a backup; not run on Kaggle (run later on vast.ai, #53) | Not run |
| 37 | DF1: D-FINE-S on Kaggle | 10 | Crashed at start: a stats step was hard-coded for 3 channels. Patched, then moved to Colab; no record of it running there | Failed |
| 38 | E1a: RT-DETRv2 on Kaggle / Colab | 10 | Moved to Colab; no record of it running there | Not run |
| 39 | CR1: Cascade R-CNN on Kaggle | 10 | Notebook built; push refused because the weekly GPU quota was used up | Not run |
| 40 | CR1: Cascade R-CNN on Colab, 7 tries | 10 | Never reached training: read-only folder, hidden error, 403 three times, wrong download flag, then 401. Kaggle blocks Colab's servers | Failed |
| 41 | SAM2: spectral-angle feature inside the detector | - | Never built (needed Cascade R-CNN first) | Not run |

## D. vast.ai box 1 (22 to 23 Sep, 2 x RTX 3090)

| # | Experiment | Length | Result | Outcome |
|---:|---|---|---|---|
| 42 | Combo = H1b + S1b + P1 (spectral-aug copy, RGB-mean input layer, 1024 px), fold-0 training images (1,796) | 50 | val 0.701, holdout **0.672**, public 0.599, full test 0.588. First launch crashed on a folder-path bug, fixed | Done |
| 43 | SP2: Savitzky-Golay spectral smoothing | 10 | screen 0.6264 (barely passed) | Done |
| 44 | L1 bug reproduction: 3-epoch tail | 3 | All 941 weights loaded, score still fell to 0.000. Cause found: warm-up learning-rate spike | Done |
| 45 | L1 fix test: 3-epoch tail with warm-up turned off | 3 | 0.689 to 0.698 | Done |
| 46 | L1b: 10-epoch tail (fixed) | 10 | 0.697, mAP75 0.832 to 0.836 | Done |
| 47 | G1c: native aspect ratio (no padding, no stretching) | 10 | screen 0.6051 (worse) | Done |
| 48 | L1c: 12-epoch tail (fixed) | 12 | 0.700 (best tail) | Done |
| 49 | L1d: 15-epoch tail (fixed) | 15 | 0.697 | Done |
| 50 | SP3: MSC scatter correction | 10 | screen 0.5587 (worse) | Done |
| 51 | SP1: MNF denoising | 10 | screen 0.6198 (no gain) | Done |
| 52 | N2: robust min-max normalization | 10 | screen 0.617 (worse) | Done |
| 53 | N3: SNV normalization | 10 | screen 0.594 (worse) | Done |
| 54 | N4: area normalization | 10 | screen 0.597 (worse) | Done |
| 55 | DF1: D-FINE-S, 16 bands (after installing missing packages and patching the code for 16 bands) | 10 | 0.664 on its own scorer (not comparable) | Done |
| 56 | E1a: RT-DETRv2, 16 bands (official repo) | 10 | 0.673 on its own scorer | Done |
| 57 | DF2: D-FINE-M, 16 bands | 10 | 0.694 on its own scorer | Done |
| 58 | DF1: D-FINE-S | 50 | Own scorer 0.779; fair holdout check **0.670** (tie) | Done |
| 59 | E1a: RT-DETRv2 | 50 | Own scorer 0.776 (not comparable). It was never re-scored on the holdout; at the time it was called a tie without that check | Done |
| 60 | DF2: D-FINE-M | 50 | Own scorer 0.786; fair holdout check **0.672** (tie) | Done |
| 61 | SP2: Savitzky-Golay, full length | 50 | holdout 0.660 (lost to Combo) | Done |
| 62 | Combo + 12-epoch tail | 12 | holdout 0.672, no gain (original kept) | Done |
| 63 | Cascade R-CNN install test | - | mmcv would not build on PyTorch 2.14 | Failed |
| 64 | PL1 (simplified): count and confidence of Combo's pseudo-labels on test images | Check | 971 of 1,000 images, about 3.6 boxes each, mean confidence 0.87, all 18 classes | Done |
| 65 | Self-training, first launch | - | Crashed: image and label IDs did not match (symlinks) | Failed |
| 66 | Self-training: new RT-DETR-L on 1,796 real + 971 pseudo-labelled test images | 50 | holdout 0.668, public 0.584, full test 0.579 | Done |
| 67 | Combo + Ultralytics built-in TTA | Inference | holdout 0.672, identical to Combo. Later found the option does nothing for RT-DETR, so not a real test | Failed |
| 68 | P2: patch training (image tiles upscaled 2x) | 30 | 0.690 on tiles, not comparable with full images | Partly |
| 69 | P4: P2 model with sliced inference, 2 tries | Inference | 0.0000 both times (box-coordinate bug); dropped | Failed |
| 70 | Full-data retrain of Combo on 2,397 images | 50 | Stopped at epoch 15 for budget; no score | Partly |
| 71 | NMS IoU sweep (inference tuning) on Combo | Inference | Failed: no test split built, cache filled the disk | Failed |
| 72 | Queued extras: 1280 px, RT-DETR-X (bigger model), hard-class focus | - | Never started (box stopped for budget) | Not run |
| 73 | D-FINE-L (larger D-FINE) | - | Skipped because D-FINE only tied | Not run |

## E. vast.ai box 2 (24 to 25 Sep, 2 x RTX 3090)

The Combo and Self-training models had been lost with box 1, so they were retrained on all labelled images.

| # | Experiment | Length | Result | Outcome |
|---:|---|---|---|---|
| 74 | combo_all: Combo recipe on all 2,997 labelled images | 50 | public 0.602; full test 0.596 | Done |
| 75 | selftrain_all: all 2,997 + 971 pseudo-labelled test images | 50 | public 0.583; full test 0.575 | Done |
| 76 | Cascade R-CNN short test run (separate PyTorch 2.1 environment) | Smoke | Training, scoring and prediction all ran | Done |
| 77 | Scorer check: my copy of Ultralytics' scoring vs Ultralytics itself | Check | Difference 0.00003 | Done |
| 78 | Cascade R-CNN, ResNet-50 FPN, COCO-pretrained, 16-band stem, 2,397 images | 24 | holdout **0.652** (MMDetection's own scorer 0.653). Lost to Combo | Done |

## F. Mac and other tries (25 Sep)

| # | Experiment | Length | Result | Outcome |
|---:|---|---|---|---|
| 79 | Predict test images on the Mac with combo_all and selftrain_all | Inference | About 3 min per 1,000 images; both files submitted (#74, #75) | Done |
| 80 | Built-in TTA check on combo_all | Inference | Same box count as without TTA (260,155): confirmed it does nothing | Done |
| 81 | Manual flip TTA written and checked | Inference | 282,784 boxes; 93.7% of confident boxes match the normal prediction | Done |
| 82 | Phase 2 files with flip TTA: combo_all, selftrain_all, Plan C (1,000 test + 1,000 ranking) | Inference | 559,119 / 511,520 / 525,108 rows, all checks passed | Done |
| 83 | Plan C resubmitted | Inference | public 0.610, full test 0.604 (same file, same score) | Done |
| 84 | Lightning AI (80 free GPU hours): H200 requested | - | Machine never assigned; dropped | Failed |

## G. vast.ai box 3 (25 to 26 Sep, 2 x RTX 3090) and final submissions

| # | Experiment | Length | Result | Outcome |
|---:|---|---|---|---|
| 85 | selftrain_v2: all 2,997 + 993 test images pseudo-labelled by combo_all with flip TTA | 50 (best round 48) | full test **0.582** (+0.006 over selftrain_all) | Done |
| 86 | combo_st: Combo recipe on the same 3,990 images | 50 (best round 50) | full test 0.574 (-0.021 vs combo_all, got worse) | Done |
| 87 | combo_all Phase 2 file with flip TTA | Inference | full test **0.595** (0.596 without TTA, so for this model TTA made no difference) | Done |
| 88 | selftrain_all Phase 2 file with flip TTA (submitted 27 Sep) | Inference | full test 0.579 (+0.004 over the same model without TTA) | Done |
| 89 | Plan C Phase 2 file with flip TTA (submitted 27 Sep) | Inference | full test **0.608**, best Phase 2 file (+0.004 over the same model without TTA) | Done |
| 90 | Kaggle re-score of every earlier file on all 1,000 test images | Check | Every file scored lower than on the 51% board (by 0.005 to 0.010). Best Phase 1 file: Plan D 0.60486 | Done |

## Counts

| Outcome | Rows |
|---|---:|
| Done | 62 |
| Failed | 16 |
| Partly | 4 |
| Not run | 8 |

Best results on the full 1,000 test images: Phase 1 file **Plan D 0.60486** (frozen as my Phase 1 score, rank 104 of 300; first place 0.67179). Phase 2 file **Plan C with flip TTA 0.60798**.

**Final result (provisional, [743830](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/743830)):** Phase 2 score on the 1,000 ranking images **0.61683**, final score **0.61085**, **rank 46 of 68**. The organizers took my best ranking-set score over all five Phase 2 files (rows 85 to 89) and did not say which file it was. I had marked Plan C and combo_all as finals on 27 Sep, but the marking did not affect the score.
