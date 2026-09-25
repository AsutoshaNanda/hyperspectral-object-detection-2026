"""Build the self-training dataset: all real labelled images + pseudo-labelled competition TEST images.

Teacher = the original Combo (holdout 0.672); its test predictions are FINAL_combo_holdout0.672.csv.
Pseudo-label rule (same as the 0.668 self-train run): keep boxes with confidence >= 0.5.
Ranking-set images are NEVER used here (organizer rule: ranking set is inference-only).
Files are hard-linked/copied (not symlinked) so the runner keeps the pt_<id> stem names.
"""
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/workspace")
import hsi_runner as R  # noqa: E402

DATA = Path("/workspace/data")
TEACHER_CSV = Path(os.environ.get("TEACHER_CSV", "/workspace/teacher/teacher_combo_conf05.csv"))
MANIFEST = Path("/workspace/folds/planC_split_manifest.json")
OUT_MANIFEST = Path(os.environ.get("OUT_MANIFEST", "/workspace/folds/st_all_manifest.json"))
ST_CUBES = Path(os.environ.get("ST_CUBES", "/workspace/st_cubes"))
ST_XML = Path(os.environ.get("ST_XML", "/workspace/st_xml"))
THR = 0.5

cubes = [p for p in DATA.rglob("VIS") if "data_train" in str(p) and "Annotations" not in str(p)][0]
testc = [p for p in DATA.rglob("VIS") if "data_test" in str(p) and "Annotations" not in str(p)][0]
xmlr = [p for p in DATA.rglob("VIS") if "Annotations" in str(p)][0]
classf = next(DATA.rglob("class.txt"))
names = R.read_classes(str(classf))

for d in (ST_CUBES, ST_XML):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)


def place(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy(src, dst)


man = json.loads(MANIFEST.read_text())
real = [str(s) for split in ("train", "val", "test") for s in man[split]]
for stem in real:
    place(cubes / f"{stem}.png", ST_CUBES / f"{stem}.png")
    place(xmlr / f"{stem}.xml", ST_XML / f"{stem}.xml")

sub = pd.read_csv(TEACHER_CSV)
sub = sub[sub.confidence >= THR]
pseudo = []
for png in sorted(testc.glob("*.png")):
    stem = png.stem
    rows = sub[sub.image_id == int(stem)] if stem.isdigit() else sub[sub.image_id == stem]
    if len(rows) == 0:
        continue
    h, w = R.read_cube(str(png)).shape[:2]
    ns = f"pt_{stem}"
    place(png, ST_CUBES / f"{ns}.png")
    ann = ET.Element("annotation")
    size = ET.SubElement(ann, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    for _, r in rows.iterrows():
        obj = ET.SubElement(ann, "object")
        ET.SubElement(obj, "name").text = names[int(r.class_id)]
        box = ET.SubElement(obj, "bndbox")
        for key, value in (("xmin", r.x1), ("ymin", r.y1), ("xmax", r.x2), ("ymax", r.y2)):
            ET.SubElement(box, key).text = str(float(value))
    ET.ElementTree(ann).write(ST_XML / f"{ns}.xml")
    pseudo.append(ns)

combined = {"train": sorted(map(str, man["train"])) + pseudo, "val": man["val"], "test": man["test"]}
OUT_MANIFEST.write_text(json.dumps(combined))

# Validate with the runner's own readers.
c2i = {n: i for i, n in enumerate(names)}
cube_stems = set(R.unique_stems(R.collect_cubes(str(ST_CUBES))))
xml_stems = set(R.annotation_map(str(ST_XML)))
assert cube_stems == xml_stems, (len(cube_stems - xml_stems), len(xml_stems - cube_stems))
w, h, parsed = R.parse_voc(str(ST_XML / f"{pseudo[0]}.xml"), c2i)
print(f"real labelled: {len(real)} | pseudo test images: {len(pseudo)} "
      f"({int(sub.image_id.isin([int(p[3:]) for p in pseudo]).sum())} boxes @conf>={THR}) | "
      f"manifest train {len(combined['train'])} val {len(combined['val'])} test {len(combined['test'])}")
print(f"sample pseudo {pseudo[0]}: {w}x{h}, {len(parsed)} boxes; cubes==xml stems: {len(cube_stems)}")
