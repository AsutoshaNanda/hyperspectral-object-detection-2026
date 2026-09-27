# 8. Timeline

Times are UTC where known.

| Date | What happened |
|---|---|
| Before 11 Sep | Research on the task and past competitions. Training runner written (`submission01_yolo16M.py`). First YOLO26m notebook run hit an invalid CUDA device error |
| 11 Sep | Review of the runner: it made a holdout but never scored it, reruns deleted old data, split balancing could silently fall back to random. Fixed. Plan B v1 (RT-DETR-L, 1 epoch) submitted by mistake: 0.24195. Rule added: no submission unless holdout is at least 0.50. YOLO recovery notebook (80 epochs) and Plan B v2 (50 epochs) started on Kaggle |
| 11 to 13 Sep | Plan A YOLO26m submitted 11 Sep 19:47: holdout 0.65471, public 0.58967. Plan B v2 RT-DETR-L submitted 12 Sep 07:19: holdout 0.63706, public 0.60836. The two tests disagreed on which was better. Found the big mAP50 vs mAP75 gap (box tightness) |
| 13 Sep | Score-improvement plan written. Public rank 46, leader 0.67943 |
| 14 Sep | Plan D (low confidence floor, same model) failed on a P100, rerun on CPU |
| 17 Sep 12:57 | Plan D submitted: public 0.61008. Plan B v2 resubmitted twice. (Plan D later became my frozen Phase 1 score, 0.60486 on all 1,000 test images) |
| 18 Sep 14:08 and 15:00 | Plan C (square padding) submitted: holdout 0.66981, public 0.60691; with low confidence floor 0.61001 |
| 18 Sep | Handwritten study notes ([notes/](../notes/)) |
| 19 Sep | 5-day roadmap: 110 experiment records, 227 run slots. Folds built (V1). Size diagnostics (D1) measured. Execution tracker and code modules written |
| 19 to 20 Sep | Fold-0 control (50 epochs): 0.68923. TTA and sliced-inference screen: nothing passed. SAM1: AUC about 0.71. Colab smoke tests: D-FINE-S passed at 640 px (random input layer), RT-DETRv2 (Hugging Face) failed. Folds 1 and 2 launched; fold 1 finished (score not saved), fold 2's outcome not recorded. Plan cut to a top 30, then a rolling top 10 |
| 21 to 22 Sep (Kaggle) | 10-epoch screens: H1b 0.6539, S1b 0.6420, P1 0.6343, M1d 0.6273, N1 0.6223 vs control 0.6201. D-FINE-S crashed at start (3-channel stats step), patched. 10-epoch localization tail collapsed to 0.031, 12-epoch tail had the same bug. Cascade R-CNN notebook built. Kaggle weekly GPU quota used up |
| 22 Sep early morning | Cascade R-CNN on Colab: 7 tries, never reached training. Kaggle blocks Colab (403/401). Decided to rent a GPU on vast.ai |
| 22 Sep morning | Organizer answers (737136, 741902, 739853): pseudo-labels and self-training on test images allowed, no hand-fixed labels. Phase 2 notice (742487): new scoring and dates, ranking images inference-only |
| 22 Sep 11:02 | vast.ai box 1 rented (2 x RTX 3090). Kaggle blocks the box too; data uploaded from the Mac |
| 22 Sep | Combo started. Localization-tail bug found (warm-up learning-rate spike) and fixed. D-FINE-S ported to 16 bands. Tier B/C screens run. 18:04 Combo holdout 0.67216 (beat Plan C by 0.002). Combo submitted as a check: public 0.59852. Overnight queue with auto-stop set up |
| 22 to 23 Sep night | 50-epoch D-FINE-S, D-FINE-M, RT-DETRv2 and SP2 runs. Their own scorer said about 0.78 |
| 23 Sep morning | Fair holdout check: D-FINE-S 0.670, D-FINE-M 0.672 (ties). Budget found to be under-counted ($10.70 spent, $9.30 left). Full-data retrain cut. Cascade R-CNN could not install. Self-training data built (symlink bug fixed) |
| 23 Sep day | Combo + L1 tail: no gain. Combo + TTA: no gain (later found to be a no-op). Self-training: 0.66837. P2 patch training done, P4 failed (bug). Public check showed Combo 0.599 vs Plan C 0.607 |
| 23 Sep about 12:00 | Metrics check: D-FINE "0.751" was Average Recall, not AP. Public rank 99, leader 0.68118 |
| 23 Sep 13:30 to 16:05 | Box restarted for extra runs. Full-data retrain reached epoch 15 of 50. NMS sweep failed. Combo and Self-training submitted, tagged "FINAL Phase 1" (self-training public 0.58423). Box paused, then destroyed. Model files were not copied |
| 24 Sep | Found the model files were lost. Box 2 rented (2 x RTX 3090, $0.324/h). combo_all and selftrain_all retrained on all 2,997 labelled images. Cascade R-CNN run in its own PyTorch 2.1 environment |
| 25 Sep 04:05 | Box 2 stopped. Cascade R-CNN holdout 0.652 (lost). combo_all public 0.60242, selftrain_all 0.58251, Plan C resubmitted 0.61001 |
| 25 Sep 05:10 | Found that Ultralytics TTA does nothing for RT-DETR. Manual flip TTA written and checked |
| 25 Sep 08:00 | Phase 1 frozen. Ranking images released (downloaded by hand) |
| 25 Sep 09:34 | Three Phase 2 files ready (combo_all, selftrain_all, Plan C; test + ranking; flip TTA) |
| 25 Sep 10:41 | Box 3 rented (2 x RTX 3090, Taiwan). Lightning AI tried and dropped. selftrain_v2 and combo_st started on all labelled images + 993 pseudo-labelled test images |
| 25 Sep 10:55 | Read the organizers' Phase 1 results (743224): 0.60486, rank 104 of 300. First place 0.67179 |
| 25 Sep about 20:20 | selftrain_v2 finished, copied to the Mac with its full logs; Phase 2 file made |
| 25 Sep 21:03 | Kaggle's board now scores all 1,000 test images: combo_all 0.596, selftrain_all 0.575 (both lower than their 51% scores) |
| 26 Sep 04:51 | combo_st finished, copied with its logs; box 3 stopped |
| 26 Sep 05:07 to 05:14 | Submitted: selftrain_v2 0.58155, combo_st 0.57443, combo_all Phase 2 with flip TTA 0.59517 (all on the full test) |
| 26 Sep | Looked at the Phase 1 results table: several top file names suggest bigger models, bigger images and more TTA views (a guess, not proof) |
| 27 Sep 04:40 and 04:42 | Submitted the last two Phase 2 files: selftrain_all 0.57949, Plan C **0.60798** (best Phase 2 file). Kaggle's list now shows every file re-scored on all 1,000 test images |
| 27 Sep 04:48 | Marked the two finals: Plan C Phase 2 (0.608) and combo_all Phase 2 (0.595) |
| 27 Sep 08:00 | Phase 2 closed |
| 27 Sep 08:05 | This repo made public, after the close |
| 27 Sep, after 09:17 | Organizers posted the provisional final ranking ([743830](https://www.kaggle.com/competitions/hyperspectral-object-detection-challenge-2026/discussion/743830)): Phase 2 0.61683, final **0.61085, rank 46 / 309 teams**. 1st 0.68624, 10th 0.67721 |
| 30 Sep 08:00 (scheduled) | Code packages due from the provisional top 15 (not required at rank 46) |
| 3 Oct 08:00 at the latest (scheduled) | Final ranking after code review |
