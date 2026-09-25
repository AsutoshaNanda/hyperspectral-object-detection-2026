import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Experiments/TOP10_REMAINING_CODE.md"
FILES = [
    "Experiments/TOP10_EXECUTION.md",
    "Submission/submission01_yolo16M.py",
    "Experiments/build_folds.py",
    "Experiments/generate_baseline_cv_notebooks.py",
    "Experiments/generate_localization_tail_notebooks.py",
    "Experiments/generate_top10_screen_notebooks.py",
    "Experiments/architecture_dataset.py",
    "Experiments/generate_architecture_screen_notebooks.py",
    "Experiments/preprocessing/normalization.py",
    "Experiments/preprocessing/pipeline.py",
    "Experiments/preprocessing/mnf.py",
    "Experiments/preprocessing/spectral_filters.py",
    "Experiments/preprocessing/spectral_augmentation.py",
    "Experiments/evaluation/evaluate_size_ap.py",
    "Experiments/evaluation/evaluate_material_pairs.py",
    "Experiments/models/__init__.py",
    "Experiments/models/catalog.py",
    "Experiments/models/adaptation.py",
    "Experiments/models/loaders.py",
    "Experiments/models/dfine.py",
    "Experiments/models/smoke.py",
    "Experiments/models/VERIFIED_SOURCES.md",
    "Experiments/run_experiment.py",
    "Experiments/orchestration/__init__.py",
    "Experiments/orchestration/ledger.py",
    "Experiments/orchestration/state.py",
    "Experiments/orchestration/queue.py",
    "Experiments/orchestration/ranking.py",
    "Experiments/orchestration/report.py",
    "Experiments/build_registry.py",
    "tests/test_submission_runner.py",
    "tests/test_spectral_pipeline.py",
    "tests/test_spectral_preprocessing.py",
    "tests/test_size_evaluation.py",
    "tests/test_build_folds.py",
    "tests/test_experiment_orchestration.py",
    "Experiments/models/tests/test_architecture_adapters.py",
    "tests/test_top10_notebooks.py",
]


def main():
    sources = [(name, (ROOT / name).read_text()) for name in FILES]
    header = f"""# Remaining top-10 source code and execution handoff

Exported: {datetime.now(timezone.utc).isoformat()}

This is a complete source snapshot of the existing files needed for the current
top-10 candidate batch. Code below is unchanged. This document is not itself a
Python script. Save each block to its named relative path under a project directory.
Original source files and their hashes remain the execution authority.

## Scope and status

The active candidate batch is L1b, L1c, L1d, S1b, N1, H1b, M1d, P1, DF1 and E1a.
Matched controls and verification folds are additional runs, not extra candidates.
N3 appears in the screen generator as a prepared backlog option; it is not part
of the active ten. No generator or synthetic smoke check is a competition result.

Notebook creation is complete for all ten active candidates. Three localization
tails, five current-detector screens, DF1 D-FINE-S, and E1a RT-DETRv2-R18 each
have a private Kaggle notebook directory and metadata. DF1 and E1a now include
fixed-fold COCO conversion, 16-band tensor loading, 18-class configuration,
RGB-mean stem expansion with 3/16 correction, official-source training, and
evidence output. Local validation compiles every code cell and checks metadata.

This is notebook-level completion only. DF1/E1a remote runtime smoke, dependency
installation, training, and competition measurements remain unverified until a
GPU notebook executes. No notebook result is claimed from static validation.

### Notebook creation checkpoint

| Candidate | Local notebook | Creation state |
|---|---|---|
| L1b | `Experiments/kaggle/localization_tails/hsi-l1-tail-10e/hsi-l1-tail-10e.ipynb` | created, statically validated |
| L1c | `Experiments/kaggle/localization_tails/hsi-l1-tail-12e/hsi-l1-tail-12e.ipynb` | created, statically validated |
| L1d | `Experiments/kaggle/localization_tails/hsi-l1-tail-15e/hsi-l1-tail-15e.ipynb` | created, statically validated |
| S1b | `Experiments/kaggle/top10_screens/s1b-rgb-mean-f0/hsi-s1b-rgb-mean-f0.ipynb` | created, statically validated |
| N1 | `Experiments/kaggle/top10_screens/n1-zscore-f0/hsi-n1-zscore-f0.ipynb` | created, statically validated |
| H1b | `Experiments/kaggle/top10_screens/h1b-spectral-copy-f0/hsi-h1b-spectral-copy-f0.ipynb` | created, statically validated |
| M1d | `Experiments/kaggle/top10_screens/m1d-multiscale010-f0/hsi-m1d-multiscale010-f0.ipynb` | created, statically validated |
| P1 | `Experiments/kaggle/top10_screens/p1-img1024-f0/hsi-p1-img1024-f0.ipynb` | created, statically validated |
| DF1 | `Experiments/kaggle/architecture_screens/df1-dfine-s-f0/hsi-df1-dfine-s-f0.ipynb` | created, static only; GPU runtime pending |
| E1a | `Experiments/kaggle/architecture_screens/e1a-rtdetrv2-r18-f0/hsi-e1a-rtdetrv2-r18-f0.ipynb` | created, static only; GPU runtime pending |

Known outstanding checks:

- D-FINE and RT-DETRv2 architecture notebooks use official repositories at the
  commit resolved during execution. Repository downloads and source patches are
  asserted, but remain runtime-sensitive until executed on Kaggle.
- Architecture notebooks use `.npy` 16-band tensors and patch official COCO
  loaders. Their first remote run is a hard smoke gate before any result counts.
- The architecture smoke runner uses synthetic tensors and currently does not
  perform a complete competition smoke epoch. Public COCO heads are not yet
  guaranteed to be replaced with 18 competition classes.
- Low-LR tails now explicitly use AdamW because optimizer=auto may ignore lr0.
- N1 uses training-fitted signed encoding and quantization for the current TIFF
  loader. This is z-score plus a declared storage transform, not untouched
  floating-point z-scores at the detector input.
- Historical all-zero confusion matrices are unavailable diagnostics. New
  runner exports null instead; independent saved-prediction evaluation is needed.
- Original-coordinate size AP is a COCO evaluation and must not be mixed with
  Ultralytics AP numbers as though their definitions were identical.
- The old holdout is previously inspected. It cannot become untouched by renaming.
- Final model must be one trained checkpoint. No model ensembling.

## Execution commands

Use an isolated Python environment with the pinned experiment dependencies.
Do not install over a working CUDA runtime merely to match a local CPU setup.
Kaggle notebook metadata names private resources in the authorized account;
these resources require that account's access and are not bundled here.

```bash
python Experiments/generate_baseline_cv_notebooks.py
python Experiments/generate_localization_tail_notebooks.py
python Experiments/generate_top10_screen_notebooks.py
python Experiments/generate_architecture_screen_notebooks.py
python -m pytest tests/test_submission_runner.py tests/test_spectral_pipeline.py tests/test_build_folds.py Experiments/models/tests tests/test_top10_notebooks.py -q
```

After inspecting a generated private notebook and confirming its dependencies,
launch one chosen run with `kaggle kernels push -p PATH_TO_RUN_DIRECTORY`.
Check the returned version and actual kernel status. A successful CLI exit code
alone does not prove that a GPU job was accepted. Download each completed version
to a fresh evidence directory. Never use `kaggle competitions submit` under the
current instructions. Do not start unbounded background dispatch or paid compute.

The registry builder requires the original roadmap at
`Resources/Hyperspectral_5_Day_Experiment_Roadmap.md`. It must not overwrite a
registry containing execution progress. The existing ledger and state remain
authoritative; this document contains no fabricated result rows.

## Source manifest

| Relative file | Lines | SHA-256 |
|---|---:|---|
"""
    manifest = []
    for name, source in sources:
        digest = hashlib.sha256(source.encode()).hexdigest()
        manifest.append({"path": name, "sha256": digest, "lines": len(source.splitlines())})
        header += f"| {name} | {len(source.splitlines())} | {digest} |\n"
    blocks = [header]
    for name, source in sources:
        language = "python" if name.endswith(".py") else "markdown"
        fence = "`" * max(4, max((len(part) for part in source.splitlines() if part and set(part) == {"`"}), default=0) + 1)
        ending = "" if source.endswith("\n") or not source else "\n"
        blocks.append(f"\n## {name}\n\n<!-- BEGIN FILE: {name} -->\n{fence}{language}\n{source}{ending}{fence}\n<!-- END FILE: {name} -->\n")
    OUTPUT.write_text("".join(blocks))
    OUTPUT.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"file": str(OUTPUT), "source_files": len(sources), "bytes": OUTPUT.stat().st_size,
                      "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
