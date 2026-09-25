#!/usr/bin/env bash
# Phase-2 v2 (Combo recipe + self-training pseudo-labels): trained on ALL 2,997 labelled images + pseudo-labelled competition TEST images.
# Teacher = combo_all (trained on all data) with flip-TTA, conf >= 0.5 (3,696 boxes / 993 test imgs).
# Ranking set is NEVER used for training. GPU 1.
set -uo pipefail
export PATH=/venv/main/bin:$PATH PYTHONPATH=/workspace CUDA_VISIBLE_DEVICES=1
cd /workspace
DATA=/workspace/data
TEST=$(find $DATA -type d -path '*data_test*/VIS' -not -path '*Annotations*' | head -1)
CLASS=$(find $DATA -name class.txt | head -1)
mkdir -p /workspace/combo_st
export TEACHER_CSV=/workspace/teacher/teacher_comboall_conf05.csv OUT_MANIFEST=/workspace/folds/combo_st_manifest.json \
       ST_CUBES=/workspace/combo_st_cubes ST_XML=/workspace/combo_st_xml
python /workspace/build_selftrain_data.py 2>&1 | tee /workspace/combo_st/run.log
python hsi_runner.py --train-cubes /workspace/combo_st_cubes --test-cubes "$TEST" --train-xml /workspace/combo_st_xml --class-file "$CLASS" \
  --workdir /workspace/combo_st --cache-dir /workspace/tmp_combo_st \
  --split-manifest /workspace/folds/combo_st_manifest.json --train-on-all \
  --family rtdetr --model rtdetr-l.pt --epochs 50 --imgsz 1024 --batch 2 --workers 8 --device 0 --seed 42 \
  --input-policy square_pad --stem-init rgb_mean --spectral-aug-copies 1 \
  --run-name combo_st --conf 0.001 --iou 0.65 --max-det 300 \
  2>&1 | tee -a /workspace/combo_st/run.log
echo "COMBO_ST_DONE rc=${PIPESTATUS[0]}" | tee -a /workspace/combo_st/run.log
