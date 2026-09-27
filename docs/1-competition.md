# 1. The competition

## The task

Find objects in hyperspectral images and say what each one is.

- A normal photo has 3 colour channels (red, green, blue). These images have **16 bands** (460 to 600 nm), so each pixel carries 16 light readings instead of 3.
- The raw files are PNG mosaics. Each 4 x 4 block of pixels holds the 16 bands. Turning this into a height x width x 16 cube is called X2Cube.
- For every object, a prediction gives: class, confidence, and a box `(x1, y1, x2, y2)`.
- 18 classes: apple, apple_plastic, badminton, banana, banana_plastic, car, car_toy, charger_head, e-bike, egg, egg_plastic, egg_wood, orange, orange_plastic, people, rubik, stone_block, table_tennis.
- Data: 3,000 labelled training images (VOC XML boxes), 1,000 test images, and 1,000 "ranking" images released later.
- Three training images (1227, 1836, 1855) are skipped by the training code because their labels do not match the image. That leaves 2,997 usable labelled images.

Official page: https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026
It is Track 1 of the "2nd Hyperspectral Remote Sensing Data Processing and Application Challenge". The Hyperspectral Object **Tracking** challenge by the same host account is a separate event.

## The metric

mAP@[0.5:0.95], like a test grade from 0 to 1.

- A predicted box counts as correct if it overlaps the true box enough. Overlap is measured by IoU (1.0 = perfect, 0.5 = half overlapping).
- The score is averaged over 10 strictness levels, from IoU 0.50 to 0.95. So boxes that are roughly right are not enough; they must be tight.
- mAP50 (loose) and mAP75 (strict) are useful side numbers. In my runs mAP50 was about 0.95 while mAP75 was about 0.76 to 0.84. That gap showed that box tightness, not finding objects, was the main weakness.

## How the final score works

This changed during the competition (Phase 2 notice, [discussion 742487](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/742487)).

- **Final = 0.5 x Phase 1 + 0.5 x Phase 2.**
- Phase 1 = my best test-set submission made before the freeze, re-scored on all 1,000 test images. The public board only showed about 51% of them.
- Phase 2 = the ranking-set score of the submission I mark as final (up to 2), as announced. (In the provisional ranking the organizers actually used each team's best ranking-set score over all its Phase 2 files; see below.) A Phase 2 file must contain predictions for both the 1,000 test and the 1,000 ranking images, or the ranking half scores zero.
- Maximum 3 submissions per day.

Timeline (UTC), as revised by the notice:

| Date | Event |
|---|---|
| 25 Sep 08:00 | Phase 1 frozen, ranking images released |
| 25 Sep 08:00 to 09:00 | Switch-over hour, do not submit |
| 27 Sep 08:00 | Phase 2 closes |
| 30 Sep 08:00 | Top-10 teams send their code package |

My earlier notes had an older schedule (Phase 2 on 23 to 25 Sep). This was corrected on 22 and 24 Sep.

Phase 1 results post: [discussion 743224](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/743224). My frozen Phase 1 score is 0.60486, rank 104 of 300. It comes from my Plan D file (submitted 17 Sep), which is my best file when every submission is re-scored on all 1,000 test images. First place is 0.67179.

### What happened at the end (provisional ranking, [743830](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/743830))

- **Phase 2 score = each team's best ranking-set score over all its Phase 2 files.** The organizers said this does not depend on which files were marked as final. So marking finals did not change anyone's score.
- **309 teams entered, but only 68 were ranked**: the ones with valid submissions in both phases. 232 teams did only Phase 1 and 9 did only Phase 2; they were not ranked. My rank is 46 / 309.
- **The provisional top 15 (not 10) must send a code package** by 30 Sep 08:00 UTC. The organizers can also ask teams ranked 16th and below.
- **Final ranking:** by 3 Oct 2026, 08:00 UTC, after code review.

## Rules that shaped the work

| Rule | Source |
|---|---|
| Final submission = one trained model, one checkpoint. No ensembles, no mixing predictions of different models (WBF etc.) | [727863](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/727863), [739853](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/739853) |
| Test-time augmentation (flip, multi-scale, crops) with the same model is allowed | 727863, 742487 |
| Public pretrained weights are allowed if declared (name, source, license). No external labelled data | 741902 |
| Do not edit or hand-add boxes. Train on the labels as given. A visible object without a label counts as background in scoring too | [737136](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/737136) |
| Removing a corrupted image is allowed if listed | [741902](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/741902) |
| Pseudo-labels from my own teacher model, and self-training on the unlabelled **test** images, are allowed | 741902, 739853 |
| The **ranking** images are for prediction only: no training, no pseudo-labels, no BatchNorm or test-time adaptation on them | 742487 |

### How my reading of the rules changed

- Until 22 Sep my notes said "do not use pseudo-labeling or distillation until the organizer confirms". So pseudo-labeling was kept out of the plan.
- On 22 Sep the staff answered three questions I had asked (737136, 741902, 739853). Pseudo-labels and self-training on test images became allowed, and self-training was added to the plan the same day.
- The Phase 2 notice (also 22 Sep) narrowed this: the new ranking images may never be trained on. At first this was misread as banning the methods themselves. It only bans using those images.
- Hand-fixing missing labels (planned as experiment A1) was ruled out by 737136 and 741902.

## Leaderboard snapshots I recorded

| Date | My best public score | My public rank | Leader | Note |
|---|---:|---:|---:|---|
| 13 Sep | 0.60836 | 46 | 0.67943 | about 11 days left |
| 22 Sep | 0.61008 | 93 | not recorded | from the discussion thread |
| 23 Sep | 0.61008 | 99 | 0.68118 | top-10 cut-off 0.66958 |
| 25 Sep (final Phase 1) | 0.60486 | 104 of 300 | 0.67179 | re-scored on all 1,000 test images |
| 27 Sep (provisional final) | final 0.61085 (Phase 2: 0.61683) | **46 / 309** | 0.68624 | only teams with both phases ranked; [743830](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/743830) |
