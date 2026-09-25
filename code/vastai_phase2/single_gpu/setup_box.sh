#!/usr/bin/env bash
# Fresh single-GPU vast box setup for combo_st. Usage: bash setup_box.sh "<signed dataset URL>"
set -euo pipefail
cd /workspace
export PATH=/venv/main/bin:$PATH
pip install -q ultralytics==8.4.147 pandas scikit-learn pyyaml opencv-python-headless 2>&1 | tail -2
python -c "import torch,ultralytics;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'ultralytics',ultralytics.__version__)"
mkdir -p data folds teacher
mv planC_split_manifest.json folds/
gunzip -c teacher_comboall_conf05.csv.gz > teacher/teacher_comboall_conf05.csv
[ -s rtdetr-l.pt ] || curl -sSL -o rtdetr-l.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/rtdetr-l.pt
curl -sSL -o data/comp.zip "$1"
cd data && unzip -q comp.zip && rm comp.zip && cd ..
echo "train pngs: $(find data -path '*data_train*/VIS/*.png' -not -path '*Annotations*' | wc -l) | xmls: $(find data -path '*Annotations*' -name '*.xml' | wc -l) | test pngs: $(find data -path '*data_test*/VIS/*.png' | wc -l)"
sha256sum rtdetr-l.pt folds/planC_split_manifest.json | cut -c1-16
wc -l teacher/teacher_comboall_conf05.csv
echo SETUP_DONE
