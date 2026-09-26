# 3. The plan (110 experiments) and what was actually done

## How the plan grew

| Date | Plan | Size |
|---|---|---|
| 11 Sep | Simple order: check data, 16-band YOLO26m, 16-band RF-DETR, tighter boxes, TTA, small objects, weak classes, pseudo-labels later, one final model | 9 steps |
| 13 Sep | Ranked queue: low-confidence inference (Plan D), square padding (Plan C), RGB-mean stem, localization tail, YOLO TTA, YOLO multi-scale, spectral augmentation, 3-fold check, final refit, Phase 2 | 10 steps |
| 19 Sep | 5-day roadmap: every technique tested alone first, then combined, then verified on 3 folds | 110 experiment records, 227 run slots |
| 19 Sep (later) | Focus on a top 30 first | 30 |
| 20 Sep | Rolling top 10, chosen from evidence | 10 |
| 22 Sep | Tier list (A to E) for the rented GPU, with a $20 to $25 budget | about 30 items, most cut |

The 19 Sep plan is in [plan/registry.json](../plan/registry.json) and [plan/experiment_matrix.json](../plan/experiment_matrix.json).

## What the 110-record plan contained

- 75 individual experiments, controls and diagnostics:
  V1 (3-fold CV), D1 (accuracy by object size), A1a-c (missing labels), N0-N4 (normalization), SP0-SP4 (spectral denoising), H1a-d (spectral augmentation), S1 (16-band input layer), G1a-c (image shape), L1a-d (localization fine-tune), DF1-DF3 (D-FINE), M1a-d (multi-scale training), T1-T4 (TTA), SN1-SN2 (Soft-NMS), SAM1-SAM3 (spectral angle for real vs plastic), PL1-PL3 (pseudo-labels), CR1 (Cascade R-CNN), P1-P5 (small objects), E1a-E1j (ten outside models).
- 35 combinations: sets A to G, 5 each (for example "best architecture + best preprocessing + best stem").
- Staged checks: every candidate on 3 folds, top 15 on a second fold, top 10 on all 3 folds at full length, top 3 once on the holdout.
- Total: 110 records, 117 staged obligations, 227 run slots, 33 "definition of done" checks.
- Promotion rule: a small change had to gain at least +0.005 mAP50-95 (or +0.005 mAP75 with no loss), without dropping any class by more than 0.03. A new architecture had to gain at least +0.01.

## Why it was cut down

Measured run times on a Kaggle T4 GPU:

| Run | Time |
|---|---|
| 10-epoch RT-DETR screen | 1 to 1.5 h |
| Full 50-epoch RT-DETR-L | about 6.6 h |
| Full 80-epoch YOLO26m | about 3.2 h |
| One RT-DETR experiment on all 3 folds | about 20 h |
| Top-10 combinations on 3 folds (one stage only) | about 198 GPU-hours |

Estimates made at the time for the full roadmap: 700 to 1,500 GPU-hours, roughly $520 to $5,240 depending on the GPU. A focused version was 150 to 300 GPU-hours.

What I actually had:
- Kaggle free GPU: 30 hours per week, used up on 21 Sep.
- Google Colab: blocked from downloading Kaggle data.
- Budget: first about $15, then a hard cap of $25 on vast.ai.
- Time: less than 5 days to the (then) deadline.

So the plan was cut to a top 30, then to a rolling top 10, then to a tier list.

## The top 10 (20 Sep)

Chosen from measured evidence, run as 10-epoch screens on fold 0 against a 10-epoch control.

| ID | Change | Reason at the time |
|---|---|---|
| L1b, L1c, L1d | 10, 12, 15-epoch low-learning-rate localization tail | Box-tightness gap of 0.115 on the fold-0 control |
| S1b | Start the 16-band input layer from the RGB weights (averaged, scaled by 3/16) | The best model used a random input layer |
| N1 | Per-band z-score normalization | No result existed for any normalization |
| H1b | One spectral-augmentation copy of each image | Implemented but never tested |
| M1d | Multi-scale training at 0.10 | Small objects and box tightness were weak |
| P1 | 1024 px input instead of 640 | 709 of 1,017 validation objects were small |
| DF1 | D-FINE-S | Built for tight boxes |
| E1a | RT-DETRv2 | Successor of the main model family |

Was this really the strongest 10 out of 110? No. Almost none of the 110 had been run, so they could not be ranked by score. The 10 were an evidence-guided bet: they targeted the weaknesses that had been measured (loose boxes, small objects) and skipped ideas that early tests had already ruled out (sliced inference, most TTA).

## The tier list (22 Sep, for vast.ai)

| Tier | Items | Decision at the time |
|---|---|---|
| A | Combo = H1b + S1b + P1 (the three best 10-epoch screens), 50 epochs | Run first |
| B | DF1, E1a, L1b/c/d, G1c; DF2, DF3, CR1 if budget | Localization bets |
| C | SP1, SP2, SP3, SN1, M1a/b/c, N2/N3/N4, P2, P4, P5 | Cheap screens |
| D | SAM2, SAM3 | Skip (needs Cascade R-CNN first, budget) |
| E | E1d to E1j (7 outside models) | Skip (7 separate framework builds) |
| Optional | Self-training | Newly allowed on 22 Sep |
| Final | TTA and Soft-NMS on the chosen model | Cheap polish |
| Not allowed | A1 / PL2 hand-added labels | Banned by the rules |
| Too big | 35 combinations | Not possible before the deadline |

## Status of every planned item

"Holdout" = the shared 300-image holdout. "Screen" = 10 epochs on fold 0, compared with the 10-epoch control (0.6201). Full numbers are in [5-results-by-location.md](5-results-by-location.md).

| ID | What | Status | Result or reason |
|---|---|---|---|
| V1 | 3-fold CV | Partly | Folds built (2,697 images; no capture or session info existed, so grouped by image). Only fold 0 was used. The fold-1 control finished but its score was not saved. Fold 2 and the six Plan B / YOLO fold notebooks never ran. **No result was ever confirmed on 3 folds.** |
| D1 | Accuracy by object size | Done | YOLO small/medium/large 0.664 / 0.751 / 0.536; RT-DETR-L 0.669 / 0.739 / 0.728 |
| A1a-c | Hand-checked missing labels | Not run | Not allowed by the rules |
| N1 | z-score | Screen | 0.6223, no real gain |
| N2 / N3 / N4 | Robust min-max / SNV / area | Screen | 0.617 / 0.594 / 0.597, all worse (N3 had a Kaggle notebook ready but ran on vast.ai) |
| SP1 / SP3 | MNF / MSC | Screen | 0.6198 / 0.5587, worse |
| SP2 | Savitzky-Golay | Screen, then 50 epochs | 0.6264 (barely passed), then holdout 0.660, lost to Combo |
| SP4 | Best SP + augmentation | Not run | SP2 lost at full length |
| H1b | Spectral augmentation copy | Screen | 0.6539, best screen, used in Combo |
| H1c / H1d | Class-targeted / with SP | Not run | Time |
| S1b | RGB-mean input layer | Screen | 0.6420, used in Combo |
| S1 ViT variants | Patch-embedding start for ViT models | Not run | No ViT model was run |
| G1a vs G1b | Stretch vs square padding | Done (old split) | Plan B v2 0.637 vs Plan C 0.670 on holdout |
| G1c | Native aspect ratio | Screen | 0.6051, worse |
| L1a-d | Localization tail | Done after a bug fix | On Kaggle: 10-epoch run collapsed to 0.031, 12-epoch run had the same bug, 15-epoch never ran. On vast.ai the bug was reproduced (score 0.000 with all weights loaded) and fixed. After the fix, on the fold-0 control: 0.697 / 0.700 / 0.697 vs base 0.689. On Combo: no gain |
| DF1 | D-FINE-S | Done | Colab smoke: failed at 128 px, passed at 640 px with a random input layer, failed with the RGB-expanded one. Kaggle run crashed at start (3-channel stats step), patched. Ran on vast.ai: holdout 0.670, tie |
| DF2 | D-FINE-M | Done | Holdout 0.672, tie |
| DF3 | D-FINE box head inside RT-DETR | Not run | D-FINE only tied; custom research code |
| M1d | Multi-scale 0.10 (transformer) | Screen | 0.6273, small pass, not added to Combo |
| M1a/b/c | YOLO multi-scale | Not run | M1a = no multi-scale; M1b matched M1d's setting; M1c untested |
| T1 / T2 / T3 | Flip / multi-scale / both | Done (fold 0) | +0.0025 / +0.0008 / -0.0005, all under the +0.005 bar |
| T4, P3 | Sliced inference | Done (fold 0) | -0.070 and -0.112 |
| SN1 / SN2 | Soft-NMS | Not run | Time; other inference tweaks gained almost nothing |
| SAM1 | Real vs plastic spectral test | Done | AUC 0.708 (apple) and 0.719 (egg), where 0.5 = guessing |
| SAM2 / SAM3 | SAM feature inside the detector | Not run | Never built: needed Cascade R-CNN first, which only worked on 24 Sep; weak SAM1 result; time |
| PL1 | Pseudo-label audit against hidden true labels | Not run as planned | Only a count/confidence summary of the test pseudo-labels was done |
| PL2 | Train with hand-confirmed missing labels | Not run | Not allowed by the rules |
| PL3 | Teacher-student | Done as self-training | Holdout 0.668 |
| CR1 | Cascade R-CNN | Done on the last try | Kaggle push refused (quota); 7 Colab tries never reached training (Kaggle blocks Colab); install failed on vast.ai box 1 (PyTorch 2.14); ran on box 2 in a PyTorch 2.1 environment: holdout 0.652, lost |
| P1 | 1024 px | Screen | 0.6343, used in Combo |
| P2 | Patch training | Partly | 0.690 on image tiles, not comparable to full images |
| P4 | Patch training + sliced inference | Failed | Coordinate bug, scored 0.0000 on two tries, dropped |
| P5 | Extra high-resolution feature level | Not run | Time |
| E1a | RT-DETRv2 | Done | Hugging Face version failed the Colab smoke test (rejected 16 channels). The official repo version ran on vast.ai: holdout about 0.67 (exact number not saved), tie |
| E1b / E1c | D-FINE-S / M | Done | Same as DF1 / DF2 |
| E1d-E1j | DINO, Co-DETR, InternImage, Swin, FocalNet, ConvNeXt V2, DINOv2 | Not run | Seven separate setups; the MMDetection-based ones could not install on the first GPU box; budget |
| A-C1 to G-C5 | 35 combinations | Not run | Only one hand-picked combination was run: Combo |
| Top-15 / top-10 / top-3 stages | Multi-fold verification | Not run | Time and budget |

Every single attempt, including smoke tests and crashed runs, is listed in [4-every-experiment-tried.md](4-every-experiment-tried.md).

The formal experiment tracker ended with only 3 of 227 run slots marked "verified" (V1 and the two D1 checks). Later results were recorded in other files ([results/](../results/)) and in run logs, not in the tracker.

## Added during the work (not in the 110)

| Item | Why | Result |
|---|---|---|
| Combo | Combine the three best screens | Holdout 0.672, public 0.599 |
| Self-training | Allowed from 22 Sep | Holdout 0.668, public 0.584 |
| Full-data retrain (2,397 images) | More data | Stopped at epoch 15 of 50 (budget), no score |
| NMS IoU sweep | Cheap inference tuning | Failed (missing test split, filled the disk) |
| combo_all / selftrain_all | Model files were lost; retrain on all 2,997 images | Public 0.602 / 0.583 |
| Manual flip TTA | Built-in TTA did nothing for RT-DETR | Works; no holdout score measured |
| selftrain_v2 / combo_st | All labelled images + 993 pseudo-labelled test images | Full test 0.582 (better than selftrain_all) / 0.574 (worse than combo_all) |
| Queued extras: 1280 px, RT-DETR-X, hard-class focus | Last ideas on box 1 | Never started |
| D-FINE-L | Bigger D-FINE | Skipped: D-FINE only tied |
