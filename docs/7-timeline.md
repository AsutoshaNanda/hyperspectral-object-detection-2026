# 7. Timeline

Times are UTC where known.

| Date | What happened |
|---|---|
| Before 11 Sep | Research on the task and past competitions. Training runner written (`submission01_yolo16M.py`). First YOLO26m notebook run hit an invalid CUDA device error |
| 11 Sep | Review of the runner: it made a holdout but never scored it, reruns deleted old data, split balancing could silently fall back to random. Fixed. Plan B v1 (RT-DETR-L, 1 epoch) submitted by mistake: 0.24195. Rule added: no submission unless holdout is at least 0.50. YOLO recovery notebook (80 epochs) and Plan B v2 (50 epochs) started on Kaggle |
| 11 to 13 Sep | Plan A YOLO26m: holdout 0.65471, public 0.58967. Plan B v2 RT-DETR-L: holdout 0.63706, public 0.60836. The two tests disagreed on which was better. Found the big mAP50 vs mAP75 gap (box tightness) |
| 13 Sep | Score-improvement plan written. Public rank 46, leader 0.67943 |
| 14 Sep | Plan D (low confidence floor, same model) failed on a P100, rerun on CPU. Later public 0.61008 |
| Before 19 Sep | Plan C (square padding): holdout 0.66981, public 0.60691; with low confidence floor 0.61001 |
| 18 Sep | Handwritten study notes ([notes/](../notes/)) |
| 19 Sep | 5-day roadmap: 110 experiment records, 227 run slots. Folds built (V1). Size diagnostics (D1) measured. Execution tracker and code modules written |
| 19 to 20 Sep | Fold-0 control (50 epochs): 0.68923. TTA and sliced-inference screen: nothing passed. SAM1: AUC about 0.71. Plan cut to a top 30, then a rolling top 10 |
| 21 Sep | 10-epoch screens on Kaggle: H1b 0.6539, S1b 0.6420, P1 0.6343, M1d 0.6273, N1 0.6223 vs control 0.6201. Localization tail collapsed to 0.031 (bug). Kaggle GPU quota used up. Colab blocked by Kaggle (403/401) |
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
| 26 to 27 Sep | Planned: finish box 3 runs, submit Phase 2 files, mark the final two before 27 Sep 08:00 UTC |
