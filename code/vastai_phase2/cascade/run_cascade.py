"""Train Cascade R-CNN 16-band, then evaluate the best checkpoint on the 300-image holdout.
Usage: python run_cascade.py [--smoke]   (run in /venv/mmdet, cwd /workspace/cascade)
"""
import sys

sys.path.insert(0, "/workspace/cascade")
from mmengine.config import Config  # noqa: E402
from mmengine.runner import Runner  # noqa: E402

smoke = "--smoke" in sys.argv
cfg = Config.fromfile("/workspace/cascade/cascade_hsi16.py")
if smoke:
    cfg.work_dir = "/workspace/cascade/smoke"
    cfg.train_dataloader.dataset.indices = 40
    cfg.val_dataloader.dataset.indices = 8
    cfg.test_dataloader.dataset.indices = 8
    cfg.train_cfg.max_epochs = 1
    cfg.train_cfg.val_interval = 1
    cfg.default_hooks.checkpoint = dict(type="CheckpointHook", interval=1)
    cfg.test_evaluator.outfile_prefix = "/workspace/cascade/smoke/holdout_preds"
    cfg.default_hooks.logger = dict(type="LoggerHook", interval=5)
runner = Runner.from_cfg(cfg)
runner.train()
if not smoke:
    import glob
    import os
    cands = glob.glob("/workspace/cascade/work/best_*.pth") or glob.glob("/workspace/cascade/work/epoch_*.pth")
    best = max(cands, key=os.path.getmtime)  # fallback: latest epoch checkpoint if no best_* file
    runner.load_checkpoint(best)
    print("BEST_CHECKPOINT", best)
runner.test()
print("CASCADE_RUN_DONE")
