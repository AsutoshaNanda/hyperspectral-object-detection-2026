"""MMDetection plug-in: load prepared 16-band uint8 .npy images (HxWx16)."""
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadNpyImage(BaseTransform):
    """Loads HxWx16 uint8 and standardises each band with TRAIN mean/std (MMDet's data
    preprocessor only accepts 1 or 3 channel mean/std, so normalisation happens here)."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def transform(self, results):
        img = np.load(results["img_path"])
        assert img.ndim == 3 and img.shape[-1] == 16, img.shape
        img = (img.astype(np.float32) - self.mean) / self.std
        results["img"] = img
        results["img_shape"] = img.shape[:2]
        results["ori_shape"] = img.shape[:2]
        return results
