#!/usr/bin/env bash
# Phase-2 rebuild of Combo (H1b+S1b+P1: rgb_mean stem + 1 spectral-aug copy + imgsz 1024),
# RT-DETR-L 50 epochs square_pad -- identical recipe to the 0.672-holdout Combo,
# but trained on ALL 2,997 labelled images (--train-on-all). GPU 0.
set -uo pipefail
export PATH=/venv/main/bin:$PATH PYTHONPATH=/workspace CUDA_VISIBLE_DEVICES=0
cd /workspace
DATA=/workspace/data
CUBES=$(find $DATA -type d -path '*data_train*/VIS' -not -path '*Annotations*' | head -1)
TEST=$(find $DATA  -type d -path '*data_test*/VIS'  -not -path '*Annotations*' | head -1)
XML=$(find $DATA   -type d -path '*Annotations*/VIS' | head -1)
CLASS=$(find $DATA -name class.txt | head -1)
mkdir -p /workspace/combo_all
python hsi_runner.py --train-cubes "$CUBES" --test-cubes "$TEST" --train-xml "$XML" --class-file "$CLASS" \
  --workdir /workspace/combo_all --cache-dir /workspace/tmp_combo_all \
  --split-manifest /workspace/folds/planC_split_manifest.json --train-on-all \
  --family rtdetr --model rtdetr-l.pt --epochs 50 --imgsz 1024 --batch 2 --workers 8 --device 0 --seed 42 \
  --input-policy square_pad --stem-init rgb_mean --spectral-aug-copies 1 \
  --run-name combo_all --conf 0.001 --iou 0.65 --max-det 300 \
  2>&1 | tee /workspace/combo_all/run.log
echo "COMBO_ALL_DONE rc=${PIPESTATUS[0]}" | tee -a /workspace/combo_all/run.log
