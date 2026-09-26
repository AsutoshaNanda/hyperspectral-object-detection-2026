# Raw result files

| File | What it holds |
|---|---|
| [kaggle_plans_ledger.jsonl](kaggle_plans_ledger.jsonl) | Plan B v1, Plan A, Plan B v2, Plan D records: metrics, hashes, submission references, decisions |
| [roadmap_ledger.jsonl](roadmap_ledger.jsonl) | Records from the 19 Sep roadmap tracker (Plan C audit, V1 folds, D1 size AP) |
| [kaggle_results.json](kaggle_results.json) | Summary of the Kaggle runs of 10 or more epochs, including the 10-epoch screens |
| [kaggle_submissions_2026-09-19.json](kaggle_submissions_2026-09-19.json) | Snapshot of the Kaggle submissions page on 19 Sep |
| [plan_c/](plan_c/) | Plan C validation and holdout evaluation (per class), run settings, submission gate |
| [plan_c_audit.json](plan_c_audit.json), [plan_c_class_comparison.csv](plan_c_class_comparison.csv) | Plan C vs Plan B v2 per-class comparison |
| [d1_size_ap/](d1_size_ap/) | Accuracy by object size for YOLO26m and RT-DETR-L, and the fold summary |
| [cascade_rcnn/](cascade_rcnn/) | Cascade R-CNN holdout score, the scorer check against Ultralytics, and the run log |
| [final_retrains/](final_retrains/) | Settings and model choice for combo_all, selftrain_all, selftrain_v2, combo_st; round-by-round logs for the last two. Those per-round scores are on images the models trained on, so they only show training health |

Results from the vast.ai boxes that were not saved as files are in [docs/5-results-by-location.md](../docs/5-results-by-location.md), taken from the run logs at the time. Where a value was not saved, the docs say so.
