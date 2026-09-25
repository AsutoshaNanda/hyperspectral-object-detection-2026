# Final training instructions: Combo + Self-training on full data

**Deadline:** Phase 2 closes **2026-09-27 08:00 UTC**. The ranking images are for **prediction only**
and must never be used for training. The final submission must come from **one model** (no blending).

Backup files (already checked, ready to submit):
`Submission/FINAL_phase2/combo_all/phase2_tta/submission.csv` and
`Submission/FINAL_phase2/selftrain_all/phase2_tta/submission.csv`

## What gets trained

| Run | Recipe | Training data | GPU | Est. time |
|---|---|---|---|---|
| `selftrain_v2` | Self-training recipe (rgb_mean stem, imgsz 1024, 50 epochs, RT-DETR-L) | all 2,997 labelled + 993 self-labelled test images | 0 | ~8 h |
| `combo_st` | Combo recipe (same + 1 spectral-aug copy) | all 2,997 labelled + 993 self-labelled test images | 1 | ~15 h |

Self-labels = `Experiments/phase2/teacher_comboall_conf05.csv`: combo_all (trained on all data) flip-TTA
predictions on **test images only**, confidence >= 0.5 (3,696 boxes on 993 test images).

## Step 1: Start the GPU box
cloud.vast.ai → Instances → **Start** instance `<INSTANCE_ID>` (2x RTX 3090, ~$0.32/h). Keep >= $6 credit.
Copy its SSH command from the console. The port/IP may change after a restart, so set them to match:

```bash
HOST=root@<BOX_IP>; PORT=<SSH_PORT>
```

## Step 2: Upload the 4 small files (from the project folder on the Mac)

```bash
cd "<project-folder>/Experiments/phase2"
scp -P $PORT build_selftrain_data.py run_selftrain_v2.sh run_combo_st.sh teacher_comboall_conf05.csv.gz $HOST:/workspace/
ssh -p $PORT $HOST 'cd /workspace && mkdir -p teacher && gunzip -c teacher_comboall_conf05.csv.gz > teacher/teacher_comboall_conf05.csv && wc -l teacher/teacher_comboall_conf05.csv && ls data folds rtdetr-l.pt hsi_runner.py'
```
Expected: `3697` lines. `data`, `folds/planC_split_manifest.json`, `rtdetr-l.pt` and `hsi_runner.py` must all exist.

## Step 3: Launch both runs (they keep going if you disconnect)

```bash
ssh -p $PORT $HOST 'cd /workspace && nohup bash run_selftrain_v2.sh > selftrain_v2.out 2>&1 & nohup bash run_combo_st.sh > combo_st.out 2>&1 & sleep 120; head -3 selftrain_v2/run.log combo_st/run.log; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv'
```
The first line of each log should say `real labelled: 2997 | pseudo test images: 993`. Both GPUs should be busy.

## Step 4: Check progress (any time)

```bash
ssh -p $PORT $HOST 'tail -c 400 /workspace/selftrain_v2/run.log; echo; tail -c 400 /workspace/combo_st/run.log'
```
Finished when the logs end with `SELFTRAIN_V2_DONE rc=0` / `COMBO_ST_DONE rc=0`.
If combo_st is not done by **~04:00 UTC Sep 27**, stop waiting and use selftrain_v2 plus the backups.

## Step 5: Copy each finished model to the Mac (repeat with `N=combo_st`)

```bash
N=selftrain_v2
D="<project-folder>/Submission/FINAL_phase2/$N"; mkdir -p "$D"
CK=$(ssh -p $PORT $HOST "python3 -c \"import json;print(json.load(open('/workspace/$N/selection.json'))['selected_checkpoint'])\"")
for f in selection.json data_audit.json run_config.json inference_geometry.json split_manifest.json dataset.yaml; do scp -P $PORT $HOST:/workspace/$N/$f "$D/"; done
scp -P $PORT "$HOST:$CK" "$D/best.pt"
ssh -p $PORT $HOST "sha256sum '$CK'"; shasum -a 256 "$D/best.pt"
```
The two sha256 values must match.

## Step 6: Stop the box (to stop the charges)
Once **both** models are copied: vast.ai console → **Stop** instance `<INSTANCE_ID>`.

## Step 7: Predict test + ranking with flip-TTA on the Mac (~6 min per model)

```bash
cd "<project-folder>"
source .venv_phase2/bin/activate
N=selftrain_v2
PRED_CONF=0.001 PRED_MAX_DET=300 python Experiments/phase2/mac_predict_tta.py \
  "Submission/FINAL_phase2/$N" "Submission/FINAL_phase2/$N/phase2_tta" \
  "Resources/hod_data/data_test/data_test/VIS" "Resources/hod_data/ranking_unzip/data_ranking/VIS"
```

## Step 8: Check the file before submitting

```bash
N=selftrain_v2 python - <<'EOF'
import os, pandas as pd, pathlib
N=os.environ["N"]; R="Resources/hod_data"
s=pd.read_csv(f"Submission/FINAL_phase2/{N}/phase2_tta/submission.csv")
ids=lambda p:{int(x.stem) for x in pathlib.Path(p).glob("*.png")}
test, rank = ids(f"{R}/data_test/data_test/VIS"), ids(f"{R}/ranking_unzip/data_ranking/VIS")
assert list(s.columns)==["id","image_id","class_id","confidence","x1","y1","x2","y2"]
assert set(s.image_id) <= test|rank, "unknown image ids"
print("test imgs", len(set(s.image_id)&test), "/1000 | ranking imgs", len(set(s.image_id)&rank), "/1000")
print("classes", s.class_id.min(), "-", s.class_id.max(), "| conf", s.confidence.min(), "-", s.confidence.max(),
      "| max boxes/img", s.groupby("image_id").size().max(), "| rows", len(s))
EOF
```
Needs: both counts close to 1000, classes 0-17, confidence 0-1, max boxes per image <= 300.

## Step 9: Submit and pick finals (by 08:00 UTC Sep 27; 3 submissions/day)
- Submit each new file on the Kaggle Submit page. Its public score = score on the **full 1,000 test images**.
  Use that to compare (the reliable measure).
- On **My Submissions**, mark your **2 final picks** (the best full-test scores among Phase 2 files).
  If you don't, Kaggle may auto-pick a Phase 1 file that scores 0 on the ranking set.

## If code review is ever requested (top 10 only)
Include: `hsi_runner.py`, `build_selftrain_data.py`, the run script, `teacher_comboall_conf05.csv`, `best.pt`,
`mac_predict_tta.py` + `tta/`, the submission sha256, and a README saying: single RT-DETR-L model, public
rtdetr-l.pt (Ultralytics, COCO) pretrained weights, pseudo-labels from combo_all on test images only
(conf >= 0.5, 1 round), ranking images used for inference only.
