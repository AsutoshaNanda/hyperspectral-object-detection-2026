#!/usr/bin/env bash
# Phase-2 rebuild of Self-training: fresh RT-DETR-L student, same recipe as the 0.668-holdout run
# (rgb_mean stem, imgsz 1024, no spectral-aug copies, 50 epochs), trained on ALL 2,997 real labelled
# images + pseudo-labelled competition test images (teacher = original Combo, conf >= 0.5). GPU 1.
set -uo pipefail
export PATH=/venv/main/bin:$PATH PYTHONPATH=/workspace CUDA_VISIBLE_DEVICES=1
cd /workspace
DATA=/workspace/data
TEST=$(find $DATA -type d -path '*data_test*/VIS' -not -path '*Annotations*' | head -1)
CLASS=$(find $DATA -name class.txt | head -1)
mkdir -p /workspace/selftrain_all
python /workspace/build_selftrain_data.py 2>&1 | tee /workspace/selftrain_all/run.log
python hsi_runner.py --train-cubes /workspace/st_cubes --test-cubes "$TEST" --train-xml /workspace/st_xml --class-file "$CLASS" \
  --workdir /workspace/selftrain_all --cache-dir /workspace/tmp_selftrain_all \
  --split-manifest /workspace/folds/st_all_manifest.json --train-on-all \
  --family rtdetr --model rtdetr-l.pt --epochs 50 --imgsz 1024 --batch 2 --workers 8 --device 0 --seed 42 \
  --input-policy square_pad --stem-init rgb_mean --spectral-aug-copies 0 \
  --run-name selftrain_all --conf 0.001 --iou 0.65 --max-det 300 \
  2>&1 | tee -a /workspace/selftrain_all/run.log
echo "SELFTRAIN_ALL_DONE rc=${PIPESTATUS[0]}" | tee -a /workspace/selftrain_all/run.log
