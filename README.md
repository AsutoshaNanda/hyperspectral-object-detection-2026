# Hyperspectral Object Detection Challenge 2026 (Track 1)

This is my record of the Kaggle competition: what I studied, what I planned, what I actually ran on Kaggle and on rented vast.ai GPUs, the results, the problems, and the mistakes.

Every number here comes from a saved result file, a run log, or a Kaggle submission. Where a number was never saved, the docs say so instead of guessing.

- Competition: https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026
- Kaggle profile: https://www.kaggle.com/itsasup (team "Asutosha", solo)
- Period covered: 11 Sep 2026 to 27 Sep 2026

## Bottom line

| Item | Result |
|---|---|
| **Final result (provisional, before code review)** | **Rank 46 of 68 ranked teams, final score 0.61085** ([organizers' table](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/743830)) |
| Phase 1 score (frozen, all 1,000 test images) | 0.60486 (the Plan D file from 17 Sep); 104th of 300 in Phase 1 |
| Phase 2 score (1,000 ranking images) | 0.61683 |
| 1st place / 10th place final score | 0.68624 / 0.67721 |
| My best public-board score (about 51% of the test images, before 25 Sep) | 0.61008 (Plan D), 0.61001 (Plan C) |
| My best model on my own 300-image holdout | Combo, 0.672 |
| Same Combo on all 1,000 test images | 0.588 (0.596 after retraining on all labelled images) |
| Experiments planned | 110 experiment records, 227 run slots |
| Experiments actually tried | **90**, listed one by one in [docs/4-every-experiment-tried.md](docs/4-every-experiment-tried.md): 62 done, 16 failed, 4 partly, 8 prepared but never run |
| Money spent on vast.ai | about $29 (estimate, see [compute](docs/6-compute-kaggle-vastai.md)) |
| Best Phase 2 file (test half, all 1,000 test images) | Plan C with flip TTA, 0.608 |
| Final picks for Phase 2 | Plan C and combo_all, marked on 27 Sep. In the end the organizers used each team's best ranking-set score over all its Phase 2 files, so the marking did not change the result |

**Main lesson.** I chose models using one small holdout of 300 images. On it, almost every model scored about 0.67. On all 1,000 of Kaggle's test images the same models scored about 0.58 to 0.605, and the order changed: Plan B v2 went from last on my holdout to first on the real test, and Combo from first to third. I kept trusting the small holdout and explained the gap away. The real gap to the top teams was genuine: about 0.066 to 10th place in the final score. Details: [problems and mistakes](docs/7-problems-and-mistakes.md).

## Final result vs the top 10

Provisional ranking posted by the organizers on 27 Sep ([discussion 743830](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/743830)), not yet code-reviewed. Final = 0.5 x Phase 1 + 0.5 x Phase 2. Only the 68 teams that submitted in both phases are ranked.

| Rank | Team | Phase 1 | Phase 2 | Final |
|---:|---|---:|---:|---:|
| 1 | SAU | 0.67179 | 0.70068 | 0.68624 |
| 2 | neuralab | 0.66658 | 0.69115 | 0.67886 |
| 3 | Lumflux | 0.66599 | 0.69118 | 0.67859 |
| 4 | dw2026 | 0.66600 | 0.69111 | 0.67856 |
| 5 | deeprune | 0.66596 | 0.69111 | 0.67854 |
| 6 | LBchaser | 0.66596 | 0.69111 | 0.67853 |
| 7 | kernelHunter8 | 0.66590 | 0.69111 | 0.67850 |
| 8 | kaggle_player88 | 0.66573 | 0.69108 | 0.67840 |
| 9 | dashMatrix | 0.66466 | 0.69111 | 0.67788 |
| 10 | HSICAS | 0.66990 | 0.68452 | 0.67721 |
| **46** | **Asutosha (me)** | **0.60486** | **0.61683** | **0.61085** |

- My ranking-set score (0.617) was a little higher than my best test-image score (0.608). The top teams rose more, from about 0.666 to about 0.69, so the gap grew slightly in Phase 2.
- The organizers did not say which of my five Phase 2 files gave the 0.61683.
- The final ranking is due by 3 Oct 2026, 08:00 UTC, after the top 15 are code-reviewed.

## What is in this repo

| Path | What it is |
|---|---|
| [docs/1-competition.md](docs/1-competition.md) | The task, the metric, the two-phase scoring, the rules and how my reading of them changed |
| [docs/2-what-i-studied.md](docs/2-what-i-studied.md) | My handwritten notes, past competitions, papers and code I read, with links |
| [docs/3-plan-vs-what-was-done.md](docs/3-plan-vs-what-was-done.md) | The 110-experiment plan, why it was cut to a top 10 and then to tiers, and the status of every planned item |
| [docs/4-every-experiment-tried.md](docs/4-every-experiment-tried.md) | **All 90 experiments tried**, one row each, including 10-epoch screens, smoke tests, crashed runs and runs that never started |
| [docs/5-results-by-location.md](docs/5-results-by-location.md) | The same results in more detail (mAP50, mAP75, notes), grouped by where they ran |
| [docs/6-compute-kaggle-vastai.md](docs/6-compute-kaggle-vastai.md) | Kaggle notebooks, the Colab attempt, the three vast.ai boxes, costs and setup |
| [docs/7-problems-and-mistakes.md](docs/7-problems-and-mistakes.md) | Wrong decisions and technical problems, with what they cost |
| [docs/8-timeline.md](docs/8-timeline.md) | Day by day |
| [notes/](notes/) | Photos of my handwritten study notes (18 Sep) |
| [results/](results/) | Raw evidence: ledgers, evaluation JSON files, Cascade R-CNN logs, final retrain settings and per-round logs, submission snapshot |
| [plan/](plan/) | The machine-readable experiment registry and configuration matrix |
| [code/](code/) | Training runner, experiment modules, Kaggle notebooks, vast.ai scripts, tests |

Not included: competition data, model weights, prediction CSV files and chat logs.

## The main models in one table

These are only the full-length models. All 90 experiments, including the short 10-epoch screens (N, SP, H, S, M, P series), tails, smoke tests and failed runs, are in [docs/4-every-experiment-tried.md](docs/4-every-experiment-tried.md).

All are single models. Holdout = the same 300 labelled images for every row. Public = Kaggle's public board (about 51% of the 1,000 test images). Full test = all 1,000 test images (the board after 25 Sep).

| Model | What it is | Holdout mAP50-95 | Public 51% (full test) |
|---|---|---:|---:|
| Plan A | YOLO26m, 16 bands, 1024 px, 80 epochs | 0.655 | 0.590 (0.581) |
| Plan B v2 | RT-DETR-L, 16 bands, 640 px, 50 epochs, image stretched to square | 0.637 | 0.608 (0.603) |
| Plan D | Plan B v2 model, keep boxes down to 0.001 confidence, up to 300 per image | not measured | 0.610 (**0.605**) |
| Plan C | RT-DETR-L, image padded to square instead of stretched | 0.670 | 0.607 / 0.610 (0.599 / 0.604; **0.608** with flip TTA) |
| Combo | Plan C + RGB-mean 16-band stem + 1 spectral-augmentation copy + 1024 px | 0.672 | 0.599 (0.588) |
| Self-training | RT-DETR-L trained on real labels + Combo's confident predictions on the test images | 0.668 | 0.584 (0.579) |
| D-FINE-S / D-FINE-M | Other detector designs, ported to 16 bands | 0.670 / 0.672 | not submitted |
| RT-DETRv2 | Another detector design, ported to 16 bands | never scored on the holdout (only on its own scorer, 0.776, not comparable) | not submitted |
| Cascade R-CNN | MMDetection, ResNet-50 FPN, 16-band stem | 0.652 | not submitted |
| combo_all | Combo recipe retrained on all 2,997 labelled images | none (no holdout left) | 0.602 (0.596; 0.595 with flip TTA) |
| selftrain_all | Self-training recipe on all 2,997 labelled + 971 pseudo-labelled test images | none | 0.583 (0.575; 0.579 with flip TTA) |
| selftrain_v2 | All 2,997 labelled + 993 test images pseudo-labelled by combo_all | none | (0.582) |
| combo_st | Combo recipe on the same images as selftrain_v2 | none | (0.574) |

## Key links

- Rules and clarifications: [single model and TTA](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/727863), [annotations](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/737136), [pseudo-labels](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/741902), [self-training on test images](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/739853), [Phase 2 notice](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/742487), [Phase 1 results](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/743224)
- Leaderboard: https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/leaderboard
- Closest past competition, PBVS 2022 hyperspectral detection: [challenge report](https://openaccess.thecvf.com/content/CVPR2022W/PBVS/html/Rangnekar_Semi-Supervised_Hyperspectral_Object_Detection_Challenge_Results_-_PBVS_2022_CVPRW_2022_paper.html), [1st place method](https://openaccess.thecvf.com/content/CVPR2022W/PBVS/html/Yu_Pseudo-Label_Generation_and_Various_Data_Augmentation_for_Semi-Supervised_Hyperspectral_Object_CVPRW_2022_paper.html)
- Model code used: [Ultralytics RT-DETR](https://github.com/ultralytics/ultralytics), [D-FINE](https://github.com/Peterande/D-FINE), [RT-DETRv2](https://github.com/lyuwenyu/RT-DETR), [MMDetection Cascade R-CNN](https://github.com/open-mmlab/mmdetection/tree/main/configs/cascade_rcnn)
- GPU rental: https://vast.ai

The full link list is in [docs/2-what-i-studied.md](docs/2-what-i-studied.md).
