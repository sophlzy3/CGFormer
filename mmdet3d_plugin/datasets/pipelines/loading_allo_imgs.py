import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as TF
from mmdet.datasets.builder import PIPELINES
from .utils import read_exr_depth_as_pil


@PIPELINES.register_module()
class LoadMultiViewImageFromFilesWithSegDepth(object):
    """Load an ALLO frame: RGB image, segmentation mask, GT (.exr) depth, mono depth.

    Unlike the SemanticKITTI loader, the geometric augmentation (resize/crop/flip/
    rotate) is applied identically to the image, the segmentation mask, the GT depth
    and the mono-depth prior, so all four stay pixel-aligned.
    """
    def __init__(self,
            data_config,
            is_train=False,
            img_norm_cfg=None,
            color_jitter=(0.4, 0.4, 0.4),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ):
        super().__init__()

        self.is_train = is_train
        self.data_config = data_config
        self.img_norm_cfg = img_norm_cfg

        self.color_jitter = (
            TF.ColorJitter(*color_jitter) if color_jitter else None
        )
        self.normalize_img = TF.Compose(
            [
                TF.ToTensor(),
                TF.Normalize(
                    mean=mean, std=std
                ),
            ]
        )
        self.ToTensor = TF.ToTensor()

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']

        if self.is_train:
            resize = float(fW)/float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            flip_v = self.data_config.get('flip_v', False) and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])

        else:
            resize = float(fW) / float(W)
            resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale

            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            flip_v = False
            rotate = 0

        return resize, resize_dims, crop, flip, flip_v, rotate

    def img_transform(self, img, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, flip_v, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, flip_v, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        # Keep geometry transforms aligned with PIL flips
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        if flip_v:
            A = torch.Tensor([[1, 0], [0, -1]])
            b = torch.Tensor([0, crop[3] - crop[1]])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, flip_v, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        if flip_v:
            img = img.transpose(method=Image.FLIP_TOP_BOTTOM)
        img = img.rotate(rotate)

        return img

    def get_inputs(self, results, flip=None, scale=None):
        img_filenames = results['img_filename']
        seg_filenames = results['seg_filename']
        depth_filenames = results['depth_filename']

        # intrins
        intrin = torch.Tensor(results['cam_intrinsic'])

        # extrins
        lidar2cam = torch.Tensor(results['lidar2cam'])
        if lidar2cam.shape[-2:] == (3, 4):
            lidar2cam = torch.cat([lidar2cam, lidar2cam.new_tensor([[0., 0., 0., 1.]])], dim=0)
        cam2lidar = lidar2cam.inverse()
        rot = cam2lidar[:3, :3]
        tran = cam2lidar[:3, 3]

        if not isinstance(img_filenames, list):
            img_filenames = [img_filenames]
        if not isinstance(seg_filenames, list):
            seg_filenames = [seg_filenames]
        if not isinstance(depth_filenames, list):
            depth_filenames = [depth_filenames]

        focal_length = results['focal_length']
        baseline = results.get('baseline', None)
        data_lists = []
        raw_img_list = []
        seg_list = []
        depth_list = []
        for i, item in enumerate(zip(img_filenames, seg_filenames, depth_filenames)):
            img_filename, seg_filename, depth_filename = item
            img = Image.open(img_filename).convert('RGB')
            seg = Image.open(seg_filename).convert('L')
            depth = read_exr_depth_as_pil(depth_filename)

            # perform image-view augmentation
            post_rot = torch.eye(2)
            post_trans = torch.zeros(2)

            if i == 0:
                img_augs = self.sample_augmentation(H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, flip_v, rotate = img_augs

            img, post_rot2, post_tran2 = self.img_transform(
                img, post_rot, post_trans, resize=resize,
                resize_dims=resize_dims, crop=crop, flip=flip, flip_v=flip_v, rotate=rotate
            )
            seg = self.img_transform_core(seg, resize_dims=resize_dims,
                    crop=crop, flip=flip, flip_v=flip_v, rotate=rotate)
            depth = self.img_transform_core(depth, resize_dims=resize_dims,
                    crop=crop, flip=flip, flip_v=flip_v, rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            # output
            canvas = np.array(img)

            if self.color_jitter and self.is_train:
                img = self.color_jitter(img)

            img = self.normalize_img(img)

            result = [img, rot, tran, intrin, post_rot, post_tran, cam2lidar]
            result = [x[None] for x in result]

            data_lists.append(result)
            raw_img_list.append(canvas)

            # Ensure writable numpy array before torch.from_numpy to avoid warning
            seg = torch.from_numpy(np.asarray(seg).copy())
            depth = self.ToTensor(depth).squeeze(0)
            seg_list.append(seg)
            depth_list.append(depth)

        results['gt_semantics'] = torch.stack(seg_list, dim=0)
        results['gt_depths'] = torch.stack(depth_list, dim=0)
        results['img_aug_flip_h'] = flip
        results['img_aug_flip_v'] = flip_v

        # Load depth prediction
        # It's actually mono depth but called stereo depth for consistency with the original code
        stereo_depth_path = results['stereo_depth_path']
        if not os.path.exists(stereo_depth_path):
            raise FileNotFoundError(
                f"Mono depth file not found: {stereo_depth_path}\n"
                "Check data_root/depth_root in the config (ALLO_3D_ROOT / ALLO_MONODEPTH_ROOT) "
                "and that the job symlinked mono_depth -> mono_depth_normal in the staged dataset."
            )
        stereo_depth = np.load(stereo_depth_path)
        stereo_depth = Image.fromarray(stereo_depth)
        resize, resize_dims, crop, flip, flip_v, rotate = img_augs
        stereo_depth = self.img_transform_core(stereo_depth, resize_dims=resize_dims,
                crop=crop, flip=flip, flip_v=flip_v, rotate=rotate)
        results['stereo_depth'] = self.ToTensor(stereo_depth)

        num = len(data_lists[0])
        result_list = []
        for i in range(num):
            result_list.append(torch.cat([x[i] for x in data_lists], dim=0))

        if focal_length is not None:
            results['focal_length'] = torch.tensor(focal_length, dtype=torch.float32)
        else:
            results['focal_length'] = None
        if baseline is not None:
            results['baseline'] = torch.tensor(baseline, dtype=torch.float32)
        else:
            results['baseline'] = None
        results['raw_img'] = raw_img_list

        return result_list

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)

        return results
