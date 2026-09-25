#!/usr/bin/env bash
# Phase-2 v2 (Self-training round 2: same recipe as selftrain_all, stronger teacher): trained on ALL 2,997 labelled images + pseudo-labelled competition TEST images.
# Teacher = combo_all (trained on all data) with flip-TTA, conf >= 0.5 (3,696 boxes / 993 test imgs).
# Ranking set is NEVER used for training. GPU 0.
set -uo pipefail
export PATH=/venv/main/bin:$PATH PYTHONPATH=/workspace CUDA_VISIBLE_DEVICES=0
cd /workspace
DATA=/workspace/data
TEST=$(find $DATA -type d -path '*data_test*/VIS' -not -path '*Annotations*' | head -1)
CLASS=$(find $DATA -name class.txt | head -1)
mkdir -p /workspace/selftrain_v2
export TEACHER_CSV=/workspace/teacher/teacher_comboall_conf05.csv OUT_MANIFEST=/workspace/folds/selftrain_v2_manifest.json \
       ST_CUBES=/workspace/selftrain_v2_cubes ST_XML=/workspace/selftrain_v2_xml
python /workspace/build_selftrain_data.py 2>&1 | tee /workspace/selftrain_v2/run.log
python hsi_runner.py --train-cubes /workspace/selftrain_v2_cubes --test-cubes "$TEST" --train-xml /workspace/selftrain_v2_xml --class-file "$CLASS" \
  --workdir /workspace/selftrain_v2 --cache-dir /workspace/tmp_selftrain_v2 \
  --split-manifest /workspace/folds/selftrain_v2_manifest.json --train-on-all \
  --family rtdetr --model rtdetr-l.pt --epochs 50 --imgsz 1024 --batch 2 --workers 8 --device 0 --seed 42 \
  --input-policy square_pad --stem-init rgb_mean --spectral-aug-copies 0 \
  --run-name selftrain_v2 --conf 0.001 --iou 0.65 --max-det 300 \
  2>&1 | tee -a /workspace/selftrain_v2/run.log
echo "SELFTRAIN_V2_DONE rc=${PIPESTATUS[0]}" | tee -a /workspace/selftrain_v2/run.log
