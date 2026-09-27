# 5. Results by location

The same experiments as [4-every-experiment-tried.md](4-every-experiment-tried.md), with more detail (mAP50, mAP75, notes), grouped by where they ran.

How to read the numbers:

- Scores are mAP50-95 (0 to 1, higher is better) unless marked mAP50 (loose) or mAP75 (strict).
- **Holdout** = the same 300 labelled images kept out of training (split: train 2,397 / validation 300 / holdout 300). This holdout was looked at many times, so by the end it was not an untouched test.
- **Fold 0** = the cross-validation split used for screens: train 1,796 / validation 901 / the same 300 holdout.
- **Screen** = 10 epochs on fold 0. Only comparable to the 10-epoch control (0.6201), not to 50-epoch runs.
- **Public** = Kaggle's public board, about 51% of the 1,000 test images (until 25 Sep).
- **Full test** = all 1,000 test images. After the Phase 1 freeze on 25 Sep, Kaggle's board scores every file this way.

## A. Kaggle notebooks (11 to 19 Sep)

| Run | Setup | Val | Holdout | Public (full test) | Note |
|---|---|---:|---:|---:|---|
| Plan B v1 | RT-DETR-L, 16 bands, 1 epoch, 640 px | 0.28070 | not measured | 0.24195 (0.23293) | Pipeline smoke test. Submitted by mistake |
| Plan A | YOLO26m, 16 bands, 80 epochs, 1024 px | 0.69894 | 0.65471 | 0.58967 (0.58079) | Holdout mAP50 0.952, mAP75 0.760. Weakest classes: car 0.345, stone_block 0.377, e-bike 0.404, people 0.414. 3.15 h |
| Plan B v2 | RT-DETR-L, 50 epochs, 640 px, stretched to square | 0.66587 | 0.63706 | 0.60836 (0.60264) | mAP50 0.942, mAP75 0.753. stone_block only 0.186. 6.6 h |
| Plan D | Plan B v2 model, confidence floor 0.001, 300 boxes per image | - | - | **0.61008 (0.60486)** | Inference only. First try failed on a P100 GPU (CUDA kernel error), rerun on CPU. This file is my frozen Phase 1 score |
| Plan C | RT-DETR-L, 50 epochs, 640 px, padded to square | 0.70146 | 0.66981 | 0.60691 (0.59928) | mAP50 0.949, mAP75 0.784. Kaggle session stopped after epoch 35; resumed from the saved checkpoint to finish |
| Plan C low-conf | Same model, confidence floor 0.001, 300 boxes | - | - | 0.61001 (0.60371) | Inference only |

Plan C beat Plan B v2 by 0.035 on the holdout but was not better on the public board, and on all 1,000 test images it was slightly worse (0.599 vs 0.603). That disagreement was the first warning sign (see [mistakes](7-problems-and-mistakes.md)).

## B. Roadmap diagnostics on Kaggle (19 to 21 Sep)

| Item | Result |
|---|---|
| V1 folds | 3 folds over 2,697 images, every class in every fold. No capture or session information in the files, so images were grouped individually |
| D1 size accuracy, YOLO26m | small 0.664, medium 0.751, large 0.536 |
| D1 size accuracy, RT-DETR-L (square pad) | small 0.669, medium 0.739, large 0.728 |
| Small objects | 709 of 1,017 validation objects are small (box area under 32 x 32 px) |
| Fold-0 control, RT-DETR-L square pad, 50 epochs | val 0.68923, mAP75 0.83196, box gap 0.115, 5.56 GPU-hours |
| TTA on that model | flip +0.00250, multi-scale +0.00075, both -0.00054 |
| Sliced inference on that model | sliced only -0.11154, full + sliced -0.07007 |
| SAM1 (real vs plastic) | AUC 0.708 apple / apple_plastic, 0.719 egg / egg_plastic |
| Fold-1 control | Finished (about 5.7 GPU-hours); score not saved |
| Fold-2 control | Launched together with fold 1; outcome and score never recorded |
| Plan B / YOLO on folds 0 to 2 | Notebooks prepared, never started |
| Day-1 diagnostics, first run | Some checkpoint diagnostics failed; repaired run gave the D1 numbers above |

### Colab smoke tests (no competition data, only "does the model run with 16 bands")

| Model | Result |
|---|---|
| D-FINE-S at 128 px, both input-layer starts | Failed: its position encoding expects 640 px |
| D-FINE-S at 640 px, random 16-band input layer | Passed forward, prediction, loss and backward checks |
| D-FINE-S at 640 px, RGB-expanded input layer | Failed the prediction check |
| RT-DETRv2-R18 (Hugging Face version), several tries | Failed: the library rejected 16-channel input |

## C. 10-epoch screens on Kaggle (21 Sep)

RT-DETR-L, square pad, 640 px, batch 2, fold 0. Pass = at least +0.005 over the control.

| Screen | Change | mAP50-95 | mAP50 | mAP75 | vs control |
|---|---|---:|---:|---:|---:|
| Control | none | 0.6201 | 0.9044 | 0.7569 | - |
| H1b | 1 spectral-augmentation copy | **0.6539** | 0.9369 | 0.7920 | +0.0338 |
| S1b | RGB-mean 16-band input layer | 0.6420 | 0.9306 | 0.7867 | +0.0219 |
| P1 | 1024 px | 0.6343 | 0.9290 | 0.7760 | +0.0142 |
| M1d | Multi-scale 0.10 | 0.6273 | 0.9184 | 0.7636 | +0.0072 |
| N1 | Per-band z-score | 0.6223 | 0.9200 | 0.7554 | +0.0022 |
| L1b (first try) | 10-epoch localization tail on the 0.689 model | 0.03104 | - | 0.00996 | Collapsed (bug) |
| L1c (first try) | 12-epoch tail | - | - | - | Same bug, invalid |

Also on Kaggle, 21 to 22 Sep:
- L1d (15-epoch tail) and N3 (SNV) were queued but never ran on Kaggle.
- D-FINE-S crashed at start: a model-statistics step was hard-coded for 3 channels. It was patched, then moved to Colab; there is no record of it running there. RT-DETRv2 was also moved to Colab and not run.
- Cascade R-CNN: the Kaggle notebook was refused (weekly GPU quota used up). Seven Colab tries never reached training: read-only folder, hidden error, 403 three times, a wrong download flag, then 401.

## D. vast.ai box 1 (22 to 23 Sep, 2 x RTX 3090)

### More 10-epoch screens (fold 0, compare with 0.6201)

| Screen | mAP50-95 | Verdict |
|---|---:|---|
| SP2 Savitzky-Golay | 0.6264 | Barely passed |
| SP1 MNF | 0.6198 | No gain |
| N2 robust min-max | 0.617 | Worse |
| N4 area normalization | 0.597 | Worse |
| N3 SNV | 0.594 | Worse |
| G1c native aspect ratio | 0.6051 | Worse |
| SP3 MSC | 0.5587 | Worse |

### Localization tail after the fix (on the fold-0 control, base 0.68919, fold-0 validation)

| Tail | mAP50-95 | mAP75 |
|---|---:|---:|
| 3 epochs (fix test) | 0.69815 | 0.83234 |
| L1b, 10 epochs | 0.69681 | 0.83641 |
| L1c, 12 epochs | 0.700 | - |
| L1d, 15 epochs | 0.697 | - |

### Other detector designs

| Model | 10 epochs (own scorer, fold-0 val) | 50 epochs (own scorer) | 50 epochs, fair holdout check |
|---|---:|---:|---:|
| D-FINE-S | 0.664 (AP75 0.805) | 0.779 | **0.670** (AP75 0.776) |
| D-FINE-M | 0.694 | 0.786 | **0.672** (AP75 0.776) |
| RT-DETRv2 | 0.673 | 0.776 | never run (no holdout score exists) |

The "own scorer" numbers are not comparable with the rest; they looked like a big win until the two D-FINE models were scored on the same holdout the same way. RT-DETRv2 never got that check, so its true holdout score is unknown. See [mistakes](7-problems-and-mistakes.md).

### Full-length runs

| Run | Setup | Val | Holdout | Public |
|---|---|---:|---:|---:|
| **Combo** | RT-DETR-L, square pad, RGB-mean stem, 1 spectral-aug copy, 1024 px, 50 epochs, fold-0 train (1,796) | 0.70145 | **0.67216** (mAP50 0.952, mAP75 0.787) | 0.59852 (full test 0.58813) |
| SP2 full | Savitzky-Golay, 50 epochs | - | 0.66019 (mAP75 0.777) | - |
| Combo + L1 tail | 12-epoch tail on Combo | - | 0.67216, no gain (original model kept) | - |
| Combo + TTA | Built-in Ultralytics TTA | - | 0.67216 | - |
| L1 bug reproduction | 3-epoch tail, before the fix | - | 0.00000 on fold-0 val (all 941 weights loaded) | - |
| Self-training | Teacher = Combo's test predictions with confidence at least 0.5 (971 of 1,000 test images, about 3.6 boxes per image, mean confidence 0.87, all 18 classes). Student = new RT-DETR-L on 1,796 real + 971 pseudo-labelled images | 0.697 | 0.66837 (mAP75 0.789) | 0.58423 (full test 0.57928) |
| P2 patch training | Tiles upscaled 2x, 30 epochs | 0.690 on tiles | not comparable | - |
| P4 sliced inference | P2 model on holdout tiles, 2 tries | - | 0.0000 both times (bug) | - |
| Full-data retrain | Combo recipe on 2,397 images | - | none, stopped at epoch 15 of 50 | - |
| NMS IoU sweep | Inference tuning on Combo | - | failed | - |
| Cascade R-CNN | MMDetection | - | could not install (PyTorch 2.14) | - |
| Self-training, first launch | - | - | crashed (image / label ID mismatch from symlinks) | - |
| Queued: 1280 px, RT-DETR-X, hard-class focus | - | - | never started | - |

Notes:
- "Combo + TTA" was not a real test. Ultralytics' built-in TTA silently does nothing for RT-DETR, so the output was byte-identical to Combo. This was found on 25 Sep.
- The self-training public score is flattered, because the student learned from those test images through pseudo-labels.

## E. vast.ai box 2 (24 to 25 Sep, 2 x RTX 3090)

The Combo and Self-training model files had been lost with box 1, so both were retrained, this time on all labelled images.

| Run | Training images | Holdout | Public (full test in brackets) |
|---|---|---:|---:|
| combo_all | 2,997 (all labelled) | none (holdout used for training) | **0.60242** (full test 0.59579) |
| selftrain_all | 2,997 + 971 pseudo-labelled test images | none | 0.58251 (full test 0.57546) |
| Cascade R-CNN | ResNet-50 FPN, COCO-pretrained, 16-band stem, 24 epochs, 2,397 images | **0.652** (mAP50 0.949, mAP75 0.767) | - |

- Cascade R-CNN was scored with a copy of Ultralytics' scoring code. That copy matched Ultralytics' own number within 0.00003 on a known model. MMDetection's own scorer gave 0.653. Files: [results/cascade_rcnn/](../results/cascade_rcnn/).
- Before the full Cascade run, a short test run confirmed training, scoring and prediction worked in the separate PyTorch 2.1 environment.
- Plan C was resubmitted on 25 Sep and scored 0.61001 again (0.60371 on the full test).
- Round-by-round logs for these runs were lost with the box; only the final settings files survive ([results/final_retrains/](../results/final_retrains/)).

## F. Mac, 25 Sep (prediction only)

- Plain prediction (no TTA) on the Mac: about 3 minutes per 1,000 images. These files were the 25 Sep submissions of combo_all and selftrain_all.
- Built-in Ultralytics TTA gave exactly the same number of boxes as no TTA (260,155), which is how the no-op was found.
- Lightning AI (80 free GPU hours) was tried for the last retrains; the H200 machine was never assigned, so it was dropped.
- Manual flip TTA: each image predicted normally and mirrored, boxes merged per class (NMS IoU 0.65), top 300 per image. On Combo's test predictions it gave 282,784 boxes vs 260,155 without TTA, and 93.7% of boxes with confidence at least 0.5 matched the normal prediction. Its effect was measured later on the full test (section H): +0.004 for Plan C and selftrain_all, about zero for combo_all.
- Phase 2 files, each covering 1,000 test + 1,000 ranking images with flip TTA:

| File | Boxes |
|---|---:|
| combo_all | 559,119 |
| selftrain_all | 511,520 |
| Plan C | 525,108 |

## G. vast.ai box 3 (25 to 26 Sep, 2 x RTX 3090)

Both runs: all 2,997 labelled images + 993 test images pseudo-labelled by combo_all with flip TTA (3,696 boxes with confidence at least 0.5). No ranking images. Predictions made on the Mac with flip TTA, covering 1,000 test + 1,000 ranking images.

| Run | Best round | Rows in Phase 2 file | Full test | vs previous version |
|---|---|---:|---:|---|
| selftrain_v2 | 48 of 50 (round 50 almost identical) | 508,731 | **0.58155** | +0.006 over selftrain_all (0.57546) |
| combo_st | 50 of 50 | 489,904 | 0.57443 | -0.021 vs combo_all (0.59579), got worse |

- The round-by-round scores in [results/final_retrains/](../results/final_retrains/) are measured on images these models trained on, so they only show that training was healthy (they end at 0.778 and 0.813). They are not real scores.
- Why combo_st got worse is not known. One guess (not tested): the spectral-augmentation copy doubled the effect of mistakes in the pseudo-labels.
- combo_all's Phase 2 file with flip TTA scored 0.59517 on the full test, against 0.59579 without TTA, so for this model flip TTA made no real difference. For Plan C and selftrain_all it added about +0.004 (see section H).

## H. All Kaggle submissions (19)

Taken from Kaggle's submissions list on 27 Sep. "Public" is the score shown at the time (about 51% of the test images). "Full test" is Kaggle's re-score of the same file on all 1,000 test images after the Phase 1 freeze. Times are UTC.

### Phase 1 files (test images only)

| Date | Submission | Public | Full test |
|---|---|---:|---:|
| 11 Sep 16:47 | Plan B v1, 1 epoch (submitted by mistake) | 0.24195 | 0.23293 |
| 11 Sep 19:47 | Plan A YOLO26m | 0.58967 | 0.58079 |
| 12 Sep 07:19 | Plan B v2 | 0.60836 | 0.60264 |
| 17 Sep 12:57 | Plan B v2 (resubmitted, twice) | 0.60836 | 0.60264 |
| 17 Sep 12:57 | **Plan D** | **0.61008** | **0.60486** (frozen as my Phase 1 score) |
| 18 Sep 14:08 | Plan C | 0.60691 | 0.59928 |
| 18 Sep 15:00 | Plan C low-conf | 0.61001 | 0.60371 |
| 22 Sep 18:29 | Combo (check) | 0.59852 | 0.58813 |
| 23 Sep 16:05 | Combo, tagged "FINAL Phase 1" (same file) | 0.59852 | 0.58813 |
| 23 Sep 16:05 | Self-training, tagged "FINAL Phase 1" | 0.58423 | 0.57928 |
| 25 Sep 04:32 | combo_all | 0.60242 | 0.59579 |
| 25 Sep 04:32 | selftrain_all | 0.58251 | 0.57546 |
| 25 Sep 04:45 | Plan C low-conf (resubmitted) | 0.61001 | 0.60371 |

The organizers take the best Phase 1 file re-scored on all 1,000 test images: Plan D, 0.60486, rank 104 of 300.

### Phase 2 files (1,000 test + 1,000 ranking images, flip TTA)

| Date | Submission | Full test (test half only) | Same model without flip TTA |
|---|---|---:|---:|
| 26 Sep 04:51 | selftrain_v2 | 0.58155 | - |
| 26 Sep 05:07 | combo_st | 0.57443 | - |
| 26 Sep 05:12 | combo_all | 0.59517 (**final pick**) | 0.59579 |
| 27 Sep 04:40 | selftrain_all | 0.57949 | 0.57546 |
| 27 Sep 04:42 | **Plan C** | **0.60798** (**final pick**) | 0.60371 |

- Flip TTA (predicting each image normally and mirrored, then merging) changed the score by +0.004 for Plan C, +0.004 for selftrain_all and -0.001 for combo_all. The files without TTA were made earlier (Plan C's on Kaggle), so this is a close but not perfectly controlled comparison.
- I marked Plan C and combo_all as finals on 27 Sep (Kaggle's API does not report this, so it is from my own record). In the end it did not matter: the organizers used each team's best ranking-set score over all its Phase 2 files.

### Provisional final result ([743830](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/743830))

| | Score |
|---|---:|
| Phase 1 (frozen) | 0.60486 |
| Phase 2 (best of my Phase 2 files on the 1,000 ranking images; which file was not stated) | 0.61683 |
| **Final** (0.5 x Phase 1 + 0.5 x Phase 2) | **0.61085** |
| **Rank** | **46 of 68** ranked teams |

The per-file ranking-set scores were never shown to me (Kaggle's private board was still blank when the provisional ranking came out).
