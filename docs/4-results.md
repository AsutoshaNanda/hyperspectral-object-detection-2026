# 4. Results

How to read the numbers:

- Scores are mAP50-95 (0 to 1, higher is better) unless marked mAP50 (loose) or mAP75 (strict).
- **Holdout** = the same 300 labelled images kept out of training (split: train 2,397 / validation 300 / holdout 300). This holdout was looked at many times, so by the end it was not an untouched test.
- **Fold 0** = the cross-validation split used for screens: train 1,796 / validation 901 / the same 300 holdout.
- **Screen** = 10 epochs on fold 0. Only comparable to the 10-epoch control (0.6201), not to 50-epoch runs.
- **Public** = Kaggle's public board, about 51% of the 1,000 test images.

## A. Kaggle notebooks (11 to 19 Sep)

| Run | Setup | Val | Holdout | Public | Note |
|---|---|---:|---:|---:|---|
| Plan B v1 | RT-DETR-L, 16 bands, 1 epoch, 640 px | 0.28070 | not measured | 0.24195 | Pipeline smoke test. Submitted by mistake |
| Plan A | YOLO26m, 16 bands, 80 epochs, 1024 px | 0.69894 | 0.65471 | 0.58967 | Holdout mAP50 0.952, mAP75 0.760. Weakest classes: car 0.345, stone_block 0.377, e-bike 0.404, people 0.414. 3.15 h |
| Plan B v2 | RT-DETR-L, 50 epochs, 640 px, stretched to square | 0.66587 | 0.63706 | 0.60836 | mAP50 0.942, mAP75 0.753. stone_block only 0.186. 6.6 h |
| Plan D | Plan B v2 model, confidence floor 0.001, 300 boxes per image | - | - | **0.61008** | Inference only. First try failed on a P100 GPU (CUDA kernel error), rerun on CPU |
| Plan C | RT-DETR-L, 50 epochs, 640 px, padded to square | 0.70146 | 0.66981 | 0.60691 | mAP50 0.949, mAP75 0.784 |
| Plan C low-conf | Same model, confidence floor 0.001, 300 boxes | - | - | 0.61001 | Inference only |

Plan C beat Plan B v2 by 0.035 on the holdout but was not better on the public board. That disagreement was the first warning sign (see [mistakes](6-problems-and-mistakes.md)).

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
| RT-DETRv2 | 0.673 | 0.776 | about 0.67 (exact value not saved) |

The "own scorer" numbers are not comparable with the rest; they looked like a big win until the models were scored on the same holdout the same way. See [mistakes](6-problems-and-mistakes.md).

### Full-length runs

| Run | Setup | Val | Holdout | Public |
|---|---|---:|---:|---:|
| **Combo** | RT-DETR-L, square pad, RGB-mean stem, 1 spectral-aug copy, 1024 px, 50 epochs, fold-0 train (1,796) | 0.70145 | **0.67216** (mAP50 0.952, mAP75 0.787) | 0.59852 |
| SP2 full | Savitzky-Golay, 50 epochs | - | 0.66019 (mAP75 0.777) | - |
| Combo + L1 tail | 12-epoch tail on Combo | - | 0.67216, no gain (original model kept) | - |
| Combo + TTA | Built-in Ultralytics TTA | - | 0.67216 | - |
| Self-training | Teacher = Combo's test predictions with confidence at least 0.5 (971 of 1,000 test images, about 3.6 boxes per image, mean confidence 0.87, all 18 classes). Student = new RT-DETR-L on 1,796 real + 971 pseudo-labelled images | 0.697 | 0.66837 (mAP75 0.789) | 0.58423 |
| P2 patch training | Tiles upscaled 2x, 30 epochs | 0.690 on tiles | not comparable | - |
| P4 sliced inference | P2 model on holdout tiles | - | 0.0000 (bug) | - |
| Full-data retrain | Combo recipe on 2,397 images | - | none, stopped at epoch 15 of 50 | - |
| NMS IoU sweep | Inference tuning on Combo | - | failed | - |
| Cascade R-CNN | MMDetection | - | could not install (PyTorch 2.14) | - |

Notes:
- "Combo + TTA" was not a real test. Ultralytics' built-in TTA silently does nothing for RT-DETR, so the output was byte-identical to Combo. This was found on 25 Sep.
- The self-training public score is flattered, because the student learned from those test images through pseudo-labels.

## E. vast.ai box 2 (24 to 25 Sep, 2 x RTX 3090)

The Combo and Self-training model files had been lost with box 1, so both were retrained, this time on all labelled images.

| Run | Training images | Holdout | Public |
|---|---|---:|---:|
| combo_all | 2,997 (all labelled) | none (holdout used for training) | **0.60242** |
| selftrain_all | 2,997 + 971 pseudo-labelled test images | none | 0.58251 |
| Cascade R-CNN | ResNet-50 FPN, COCO-pretrained, 16-band stem, 24 epochs, 2,397 images | **0.652** (mAP50 0.949, mAP75 0.767) | - |

- Cascade R-CNN was scored with a copy of Ultralytics' scoring code. That copy matched Ultralytics' own number within 0.00003 on a known model. MMDetection's own scorer gave 0.653. Files: [results/cascade_rcnn/](../results/cascade_rcnn/).
- Plan C was resubmitted on 25 Sep and scored 0.61001 again.

## F. Mac, 25 Sep (prediction only)

- Manual flip TTA: each image predicted normally and mirrored, boxes merged per class (NMS IoU 0.65), top 300 per image. On Combo's test predictions it gave 282,784 boxes vs 260,155 without TTA, and 93.7% of boxes with confidence at least 0.5 matched the normal prediction. Its effect on the score was not measured.
- Phase 2 files, each covering 1,000 test + 1,000 ranking images with flip TTA:

| File | Boxes |
|---|---:|
| combo_all | 559,119 |
| selftrain_all | 511,520 |
| Plan C | 525,108 |

## G. vast.ai box 3 (25 Sep, still running)

| Run | Training images | Status |
|---|---|---|
| selftrain_v2 | 2,997 + 993 test images pseudo-labelled by combo_all with flip TTA (3,696 boxes, confidence at least 0.5) | Training, no result yet |
| combo_st | Same images + 1 spectral-aug copy | Training, no result yet |

## H. All Kaggle submissions and public scores

| Submission | Public |
|---|---:|
| Plan B v1 (1 epoch) | 0.24195 |
| Plan A YOLO26m | 0.58967 |
| Plan B v2 | 0.60836 |
| Plan D | 0.61008 |
| Plan C | 0.60691 |
| Plan C low-conf (submitted before 19 Sep, and again on 25 Sep) | 0.61001 |
| Combo (22 Sep as a check, 23 Sep tagged "FINAL Phase 1") | 0.59852 |
| Self-training (23 Sep, tagged "FINAL Phase 1") | 0.58423 |
| combo_all (25 Sep) | 0.60242 |
| selftrain_all (25 Sep) | 0.58251 |

Organizers' re-score on all 1,000 test images (Phase 1, frozen): **0.60486**, rank 104 of 300.
