from pathlib import Path
import numpy as np
import pandas as pd
from mmdet.datasets import DATASETS
from torch.utils.data import Dataset
from mmdet.datasets.pipelines import Compose

# Ensure allo pipeline components are registered before Compose(pipeline) runs
from .pipelines import map_labels  # noqa: F401
from .pipelines import filter_depth  # noqa: F401
from .pipelines import downsample_voxels  # noqa: F401
from .pipelines import loading_allo_imgs  # noqa: F401
from .allo_exclude import filter_excluded


@DATASETS.register_module()
class ALLODataset(Dataset):
    """ALLO orbital scenes: RGB + segmentation + GT/mono depth + voxel occupancy.

    Raw labels use 9 classes with the anomaly at index 1 (John's convention). In OUR
    data the injected anomaly instead lives at raw index 7 -- the swap is done by
    ``AnomalyMapLabels(anomaly_raw_index=7)`` in the pipeline, for BOTH gt_occ and
    gt_semantics, so this loader hands raw labels through untouched.
    """
    CLASSES = (
        "background", "robotic_arms", "solar_arrays", "pressurized_modules",
        "airlock_docking_ports", "truss_other", "modules_other", "celestial_bodies", "anomaly"
    )
    # Default subdir names under each scene (override in config if your layout differs)
    DEFAULT_IMG_SUBDIR = "images"
    DEFAULT_SEG_SUBDIR = "segmentation_mask"
    DEFAULT_DEPTH_SUBDIR = "depth"
    DEFAULT_VOXEL_SUBDIR = "voxels"
    DEFAULT_IMG_GLOB = "*normal.png"

    def __init__(
        self,
        data_root,
        depth_root,
        pipeline,
        split,
        occ_size,
        pc_range,
        test_mode=False,
        load_continuous=False,
        mini_split=False,
        img_subdir=None,
        seg_subdir=None,
        depth_subdir=None,
        voxel_subdir=None,
        img_glob=None,
        **kwargs
    ):
        super().__init__()

        self.load_continuous = load_continuous
        self.img_subdir = img_subdir or self.DEFAULT_IMG_SUBDIR
        self.seg_subdir = seg_subdir or self.DEFAULT_SEG_SUBDIR
        self.depth_subdir = depth_subdir or self.DEFAULT_DEPTH_SUBDIR
        self.voxel_subdir = voxel_subdir or self.DEFAULT_VOXEL_SUBDIR
        self.img_glob = img_glob or self.DEFAULT_IMG_GLOB

        self.data_root = Path(data_root)
        self.depth_root = depth_root
        self.test_mode = test_mode
        self.data_infos = self.load_annotations()

        if mini_split:
            self.data_infos = self.data_infos.sample(n=100, random_state=42)

        self.occ_size = occ_size
        self.pc_range = pc_range
        #* Intrinsics for 1344x672 from the provided (1280x720, fx=fy=888.89, cx=640, cy=360)
        sx, sy = 1344/1280, 672/720
        fx, fy = 888.89*sx, 888.89*sy                  # ≈ 933.33, 829.63
        cx, cy = 640*sx, 360*sy                        # 672.0, 336.0
        self.cam_intrinsics = np.identity(4, dtype=np.float32)
        self.cam_intrinsics[:3, :4] = np.array([
            [fx, 0., cx, 0.],
            [0., fy, cy, 0.],
            [0., 0.,  1., 0.],
        ], dtype=np.float32)
        #* Transform from LiDAR to Camera coordinates (flipped y-axis)
        self.lidar2cam_rts = np.array([
            [0., -1.,  0., 0.],   # X_cam = -Y_sk
            [0.,  0.,  1., 0.],   # Y_cam =  Z_sk
            [1.,  0.,  0., 0.],   # Z_cam =  X_sk
            [0.,  0.,  0., 1.],
        ], dtype=np.float32)
        self.lidar2img_rts = self.cam_intrinsics @ self.lidar2cam_rts

        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        self._set_group_flag()

    def _set_group_flag(self):
        """Set flag according to image aspect ratio."""
        self.flag = np.zeros(len(self.data_infos), dtype=np.uint8)

    def __len__(self):
        return len(self.data_infos)

    def prepare_train_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            print('found None in training data')
            return None

        example = self.pipeline(input_dict)
        return example

    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            print('found None in test data')
            return None

        example = self.pipeline(input_dict)
        return example

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def get_data_info(self, index):
        info = self.data_infos.iloc[index]

        input_dict = dict(
            occ_size = np.array(self.occ_size),
            pc_range = np.array(self.pc_range),
        )

        focal_length = None
        baseline = None

        input_dict.update(
            dict(
                img_filename=info['img_path'],
                seg_filename=info['seg_path'],
                stereo_depth_path=info['stereo_depth_path'],
                depth_filename=info['gt_depth_path'],
                voxel_path=info['voxel_path'],
                lidar2img=self.lidar2img_rts,
                cam_intrinsic=self.cam_intrinsics,
                lidar2cam=self.lidar2cam_rts,
                focal_length=focal_length,
                baseline=baseline
            ))
        # gt_occ is None for test-set
        input_dict['gt_occ'] = self.get_ann_info(index, key='voxel_path')
        return input_dict

    def load_annotations(self):
        """
        Load annotations from the data root and store them in a Pandas DataFrame.

        Expected allo_3d folder structure (under data_root):
            allo_3d/
                train/                          # or test/
                    <scene>/images/*normal.png   # RGB images (required)
                    <scene>/segmentation_mask/   # same names, .png segmentation
                    <scene>/depth/               # same base names, .exr depth
                    <scene>/voxels/              # same base names, .npz with 'labels' key
                mono_depth/
                    train/                       # or test/; .npy mono depth per frame

        Paths are derived from each image path: images -> segmentation_mask, depth, voxels;
        stereo_depth_path uses <depth_root>/<split>/... and .npy.
        """
        split = "train" if not self.test_mode else "test"
        split_path = self.data_root / split
        columns = [
            "img_path", "seg_path", "gt_depth_path", "stereo_depth_path", "voxel_path",
        ]
        frames = sorted(split_path.glob(f"**/{self.img_subdir}/{self.img_glob}"))
        depth_root = Path(self.depth_root)
        self.data_infos = pd.DataFrame(
            [
                {
                    "img_path": str(f),
                    "seg_path": str(f).replace(self.img_subdir, self.seg_subdir),
                    "gt_depth_path": str(f).replace(self.img_subdir, self.depth_subdir).replace(".png", ".exr"),
                    "stereo_depth_path": str(depth_root / Path(f).relative_to(self.data_root).with_suffix(".npy")),
                    "voxel_path": str(f).replace(self.img_subdir, self.voxel_subdir).replace(".png", ".npz"),
                }
                for f in frames
            ],
            columns=columns,
        )
        self.data_infos = filter_excluded(self.data_infos, self.data_root)
        return self.data_infos

    def get_ann_info(self, index, key='voxel_path'):
        info = self.data_infos.iloc[index][key]
        return None if info is None else np.load(info)['labels']
