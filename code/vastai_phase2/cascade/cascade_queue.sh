#!/usr/bin/env bash
# Box-side queue: wait for Self-training to free GPU 1, then (1) ruler check, (2) Cascade R-CNN
# train 24 epochs + holdout predictions, (3) score holdout with the Ultralytics-identical metric.
set -uo pipefail
cd /workspace/cascade
LOG=/workspace/cascade/queue.log
say(){ echo "[$(date -u +%H:%M)] $*" | tee -a "$LOG"; }
say "waiting for SELFTRAIN_ALL_DONE"
until grep -aq SELFTRAIN_ALL_DONE /workspace/selftrain_all/run.log 2>/dev/null; do sleep 60; done
say "self-training done; ruler check on GPU1"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace /venv/main/bin/python ruler_check.py selftrain_all > ruler.log 2>&1
tail -6 ruler.log | tee -a "$LOG"
say "START cascade training (GPU1)"
CUDA_VISIBLE_DEVICES=1 /venv/mmdet/bin/python run_cascade.py > cascade_train.log 2>&1
say "cascade exit rc=$?"
if [ -f holdout_preds.bbox.json ]; then
  /venv/main/bin/python eval_ultra.py data/test.json holdout_preds.bbox.json > cascade_holdout_ultra.json 2>&1
  say "CASCADE HOLDOUT (ultralytics ruler): $(tr -d '\n ' < cascade_holdout_ultra.json)"
  say "CASCADE HOLDOUT (pycocotools): $(grep -aoE 'coco/bbox_mAP: [0-9.]+' cascade_train.log | tail -1)"
fi
say "CASCADE_QUEUE_DONE"
