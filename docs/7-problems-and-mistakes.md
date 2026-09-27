# 7. Problems and mistakes

## Part 1: decisions that went wrong

### 1. Trusting one 300-image holdout (the biggest mistake)

**What happened.** Almost every model choice was made on the same 300 held-out training images. The planned 3-fold cross-validation was never finished (only fold 0 was used). On this holdout nearly everything scored about 0.67. On Kaggle's test images the same models scored about 0.58 to 0.61.

The two tests also ranked models differently:

| Model | Holdout (300 images) | Public (about 510 test images) | Full test (all 1,000, scored after 25 Sep) |
|---|---:|---:|---:|
| Combo | 0.672 (1st) | 0.599 | 0.588 (3rd) |
| Plan C | 0.670 (2nd) | 0.607 | 0.599 (2nd) |
| Self-training | 0.668 (3rd) | 0.584 | 0.579 (5th) |
| Plan A YOLO26m | 0.655 (4th) | 0.590 | 0.581 (4th) |
| Plan B v2 | 0.637 (last) | 0.608 | 0.603 (1st) |
| Plan D (Plan B v2 model, low confidence floor) | - | 0.610 | 0.605 (my frozen Phase 1 score) |

The full-test scores arrived after the decisions were made, but they confirm the problem: the holdout's order was wrong. It also means the square-padding "win" (+0.035 on the holdout) did not hold on the real test (Plan C 0.599 vs Plan B v2 0.603).

**Why it was wrong.**
- The public board uses about 51% of 1,000 test images, about 510 images. That is a larger sample than my 300. I set a rule to ignore the public board because it was "only 51%", but my holdout was even smaller.
- The holdout came from the training images. The test images may be different from them in some way (not proven), so a training-image holdout can over-state the real score.
- After being used for dozens of decisions, the holdout was no longer an untouched test. This was noted on 19 Sep but not acted on.
- When the measured models all landed at about 0.65 to 0.67, I read it as "the ceiling of this dataset". The frozen Phase 1 results show top teams at about 0.67 **on the real test**, while I was at 0.605. The ceiling was mine, not the dataset's.

**Cost.** Models were picked and tuned on a test that did not reflect the real images. Most of the roughly $29 spent on vast.ai went into experiments judged by this holdout.

**Lesson.** When a small local test and a bigger real test disagree, the disagreement is the warning. Use more folds or a larger validation set, and take "should we change tactics?" seriously instead of defending the plan.

### 2. Submitting a 1-epoch smoke test

Plan B v1 was a 1-epoch pipeline check (validation 0.28). It was submitted and scored 0.24195. This broke my own rule of not submitting anything with a holdout below 0.50. The runner was changed afterwards so smoke runs cannot create a submission file.

### 3. Model files lost with the first GPU box

On 23 Sep only the prediction CSV files were copied from box 1 before it was destroyed. The trained Combo and Self-training models stayed on the box. Phase 2 needs a model to predict the new ranking images, so on 24 Sep both had to be retrained on a new box (about $4, and the original models are gone for good).

### 4. Picking the "FINAL Phase 1" pair by reasoning, not numbers

Combo and Self-training were tagged as the final Phase 1 picks. Self-training was 5th on the holdout (0.668), behind Combo, D-FINE-M, D-FINE-S and Plan C. It was picked because "it learned from the test images". That was a reasonable bet, not a measured one, and its public score (0.584) was the lowest of the retrained models.

### 5. 10-epoch screens as the gate

Short screens were cheap, but they can hide ideas that start slowly, and a +0.006 difference is close to noise. The three best screens were combined into one "Combo" without testing other combinations.

### 6. Budget tracking was wrong for a while

On box 1 the running estimate said about $5 spent when about $10.70 had actually been spent. It was corrected only when I read the real balance from the vast.ai page. After that, a budget guard on the box (anchored to the real balance) alerted and stopped the box near $2.

### 7. Old schedule in the notes

Notes said Phase 2 would be 23 to 25 Sep. The organizers had moved it to 25 to 27 Sep (notice 742487). This caused confusion about when the ranking images would appear.

### 8. The last retrain of Combo made it worse

combo_st (Combo recipe plus 993 pseudo-labelled test images) scored 0.574 on the full test, against 0.596 for combo_all without them. The same pseudo-labels helped the self-training model (+0.006). Why Combo got worse is not known. One untested guess is that its spectral-augmentation copy doubled the effect of wrong pseudo-labels.

### 9. Misleading numbers shown next to real ones

After retraining on all labelled images (24 Sep), the models were scored on images they had trained on (0.795 and 0.751). These numbers mean nothing and should not have been put in the same table as real scores.

## What the top teams probably did differently

This is a guess from the file names on the Phase 1 results table ([743224](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/743224)), not proof:

- Much bigger models (names mention YOLO26x, Co-DINO with Swin, DINOv3, RF-DETR, DEIM). I used RT-DETR-L.
- Bigger images (1536, 1720 px). I used 640 and 1024.
- More test-time views (4 to 8 flips and sizes). I used 2.
- Work aimed at box precision (box loss, box refinement). I knew this was my weak spot but did not fix it.
- Some names also suggest several models combined, which is against the rules; the organizers replace those scores with the team's best single-model score.

The provisional final ranking confirmed the gap: my final score was 0.61085 (46th of 68), against 0.67721 for 10th place and 0.68624 for 1st. The top 10 all scored about 0.684 to 0.701 on the ranking images, against my 0.617. My gap of about 0.066 looks mostly like model size and compute, not one missing setting. Lesson for next time: try one big modern model at a large image size early, before weeks of tuning a medium one.

## Part 2: technical problems

| Problem | Effect | Fix |
|---|---|---|
| Localization tail destroyed the model: score fell from 0.689 to 0.031 | L1 results invalid on Kaggle | Cause: the fine-tune kept Ultralytics' default warm-up bias learning rate of 0.1, about 1,000 times the target 1e-4, which blew up an already-trained model. Set `warmup_epochs=0`, `warmup_bias_lr=0`, `warmup_momentum=0`. After the fix: 0.689 to 0.698 |
| Ultralytics `augment=True` (TTA) silently does nothing for RT-DETR | "Combo + TTA" was byte-identical to Combo, so the reported "no gain" was not a real test | Wrote manual flip TTA on 25 Sep ([code/vastai_phase2/mac_predict_tta.py](../code/vastai_phase2/mac_predict_tta.py)) |
| D-FINE and RT-DETRv2 scored by their own tools (0.78) vs Ultralytics (0.70) | Looked like a big win | D-FINE-S and D-FINE-M re-scored on the same holdout: 0.670 / 0.672, a tie. RT-DETRv2 was never re-scored, but was still called a tie at the time |
| A metrics script printed D-FINE's Average Recall (0.751) as its AP | False "breakthrough" | Parser fixed; real AP was 0.670 and 0.672 |
| Kaggle weekly GPU quota ran out (21 Sep) | Could not run more on Kaggle | Moved to vast.ai |
| Kaggle API blocks Colab and vast.ai addresses (401/403) | No direct data download | Download on the Mac, then upload or use a signed link |
| Very slow transfers (Kaggle CLI 30 to 40 KB/s, Mac upload 3 KB/s to 1.4 MB/s) | Hours lost | Signed download link on the box; 8 parallel upload streams |
| mmcv would not build on PyTorch 2.14 | Cascade R-CNN could not run on box 1 | Separate environment with PyTorch 2.1.2 on box 2 |
| MMDetection only accepts 1- or 3-channel mean/std | 16-band input rejected | Normalize the 16 bands while loading the image |
| D-FINE port: missing packages, a debug visualizer that cannot save more than 4 channels, a profiler hard-coded to 3 channels | Several crashes before training | Installed the missing packages, disabled the visualizer, patched 3 to 16 |
| RF-DETR's standard loader converts images to RGB | 16 bands would be lost | Not used; RT-DETR (accepts the channel count) used instead |
| Self-training images were symlinks; the code resolved them to the original file names | "Training image and label IDs do not match" | Used real copies instead of symlinks |
| Folder search matched `Annotations/VIS` instead of the image folder | Combo crashed at start | Excluded `Annotations` from the search |
| Sliced inference (P4) mapped boxes with the wrong offset after square padding | Scored 0.0000 | Dropped for budget |
| NMS sweep built no test split and filled the disk with a 24 GB cache | Run failed, disk at 88% | Dropped, cache deleted |
| A failed run wrote a "done" marker | Would have auto-stopped the box during other training | Marker removed, auto-stop re-armed |
| Mac-side watcher stopped while the laptop lid was closed | Box 2 idled about 30 minutes | Put the stop trigger on the box itself |
| Plan D's first run landed on a P100 GPU | CUDA "no kernel image" error | Reran on CPU |
| Saved Plan C confusion matrices were all zeros | Could not read false positives / negatives | Treated as unavailable |
| Ranking images not downloadable at release time | Delay on 25 Sep | Downloaded by hand from the Kaggle page |
| Plan C's Kaggle session stopped after epoch 35 of 50 | Run incomplete | Resumed from the saved checkpoint to finish |
| Kaggle allows only 2 GPU notebooks at once, 30 GPU-hours per week | Screens ran in pairs; quota gone by 21 Sep | Moved to vast.ai |
| D-FINE smoke test at 128 px | Failed (position encoding needs 640 px) | Ran at 640 px |
| RT-DETRv2 Hugging Face version rejected 16-channel input | Smoke test failed on Colab | Used the official RT-DETR repo on vast.ai instead |
| Colab: `/kaggle/input` is read-only, download errors were hidden, `--unzip` flag not supported | Several failed Colab tries before the real block (403/401) was seen | Colab dropped |
