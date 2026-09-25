# Hyperspectral Object Detection Challenge 2026 (Track 1)

This is my record of the Kaggle competition: what I studied, what I planned, what I actually ran on Kaggle and on rented vast.ai GPUs, the results, the problems, and the mistakes.

Every number here comes from a saved result file, a run log, or a Kaggle submission. Where a number was never saved, the docs say so instead of guessing.

- Competition: https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026
- Kaggle profile: https://www.kaggle.com/itsasup (team "Asutosha", solo)
- Period covered: 11 Sep 2026 to 25 Sep 2026

## Bottom line

| Item | Result |
|---|---|
| Phase 1 final score (re-scored by the organizers on all 1,000 test images) | **0.60486, rank 104 of 300** |
| 1st place, Phase 1 | 0.67179 |
| My best public-board score (about 51% of the test images) | 0.61008 (Plan D), 0.61001 (Plan C) |
| My best model on my own 300-image holdout | Combo, 0.672 |
| Same Combo on the public board | 0.59852 (0.60242 after retraining on all labelled images) |
| Experiments planned | 110 experiment records, 227 run slots |
| Runs actually done | about 50, most of them short 10-epoch screens |
| Money spent on vast.ai | about $28 (estimate, see [compute](docs/5-compute-kaggle-vastai.md)) |
| Phase 2 (ranking set) | three prediction files ready, two more retrains still running on 25 Sep, nothing submitted yet |

**Main lesson.** I chose models using one small holdout of 300 images. On it, almost every model scored about 0.67. On Kaggle's test images the same models scored about 0.58 to 0.61, and the two tests ranked the models differently. I kept trusting the small holdout and explained the gap away. The real gap to the top teams (about 0.06) was genuine. Details: [problems and mistakes](docs/6-problems-and-mistakes.md).

## What is in this repo

| Path | What it is |
|---|---|
| [docs/1-competition.md](docs/1-competition.md) | The task, the metric, the two-phase scoring, the rules and how my reading of them changed |
| [docs/2-what-i-studied.md](docs/2-what-i-studied.md) | My handwritten notes, past competitions, papers and code I read, with links |
| [docs/3-plan-vs-what-was-done.md](docs/3-plan-vs-what-was-done.md) | The 110-experiment plan, why it was cut to a top 10 and then to tiers, and the status of every planned item |
| [docs/4-results.md](docs/4-results.md) | Every measured result, grouped by where it ran |
| [docs/5-compute-kaggle-vastai.md](docs/5-compute-kaggle-vastai.md) | Kaggle notebooks, the Colab attempt, the three vast.ai boxes, costs and setup |
| [docs/6-problems-and-mistakes.md](docs/6-problems-and-mistakes.md) | Wrong decisions and technical problems, with what they cost |
| [docs/7-timeline.md](docs/7-timeline.md) | Day by day |
| [notes/](notes/) | Photos of my handwritten study notes (18 Sep) |
| [results/](results/) | Raw evidence: ledgers, evaluation JSON files, Cascade R-CNN logs, submission snapshot |
| [plan/](plan/) | The machine-readable experiment registry and configuration matrix |
| [code/](code/) | Training runner, experiment modules, Kaggle notebooks, vast.ai scripts, tests |

Not included: competition data, model weights, prediction CSV files and chat logs.

## The models in one table

All are single models. Holdout = the same 300 labelled images for every row. Public = Kaggle's public board (about 51% of the 1,000 test images).

| Model | What it is | Holdout mAP50-95 | Public |
|---|---|---:|---:|
| Plan A | YOLO26m, 16 bands, 1024 px, 80 epochs | 0.655 | 0.590 |
| Plan B v2 | RT-DETR-L, 16 bands, 640 px, 50 epochs, image stretched to square | 0.637 | 0.608 |
| Plan D | Plan B v2 model, keep boxes down to 0.001 confidence, up to 300 per image | not measured | 0.610 |
| Plan C | RT-DETR-L, image padded to square instead of stretched | 0.670 | 0.607 / 0.610 |
| Combo | Plan C + RGB-mean 16-band stem + 1 spectral-augmentation copy + 1024 px | 0.672 | 0.599 |
| Self-training | RT-DETR-L trained on real labels + Combo's confident predictions on the test images | 0.668 | 0.584 |
| D-FINE-S / D-FINE-M / RT-DETRv2 | Other detector designs, ported to 16 bands | 0.670 / 0.672 / about 0.67 | not submitted |
| Cascade R-CNN | MMDetection, ResNet-50 FPN, 16-band stem | 0.652 | not submitted |
| combo_all | Combo recipe retrained on all 2,997 labelled images | none (no holdout left) | 0.602 |

## Key links

- Rules and clarifications: [single model and TTA](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/727863), [annotations](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/737136), [pseudo-labels](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/741902), [self-training on test images](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/739853), [Phase 2 notice](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/742487), [Phase 1 results](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/743224)
- Leaderboard: https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/leaderboard
- Closest past competition, PBVS 2022 hyperspectral detection: [challenge report](https://openaccess.thecvf.com/content/CVPR2022W/PBVS/html/Rangnekar_Semi-Supervised_Hyperspectral_Object_Detection_Challenge_Results_-_PBVS_2022_CVPRW_2022_paper.html), [1st place method](https://openaccess.thecvf.com/content/CVPR2022W/PBVS/html/Yu_Pseudo-Label_Generation_and_Various_Data_Augmentation_for_Semi-Supervised_Hyperspectral_Object_CVPRW_2022_paper.html)
- Model code used: [Ultralytics RT-DETR](https://github.com/ultralytics/ultralytics), [D-FINE](https://github.com/Peterande/D-FINE), [RT-DETRv2](https://github.com/lyuwenyu/RT-DETR), [MMDetection Cascade R-CNN](https://github.com/open-mmlab/mmdetection/tree/main/configs/cascade_rcnn)
- GPU rental: https://vast.ai

The full link list is in [docs/2-what-i-studied.md](docs/2-what-i-studied.md).
