import os
from pathlib import Path

import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from PIL import Image


def read_exr_depth_as_pil(path):
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # float32 if EXR
    if arr is None:
        raise ValueError("Failed to read EXR. Your OpenCV build may lack OpenEXR support.")
    if arr.ndim == 3:
        arr = arr[..., 0]  # pick first channel if needed
    return Image.fromarray(arr.astype(np.float32), mode='F')
