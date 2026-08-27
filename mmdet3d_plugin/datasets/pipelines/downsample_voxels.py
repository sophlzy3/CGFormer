import torch
import numpy as np
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module(name='DownsampleVoxels')
class DownsampleVoxels(object):
    def __init__(self, occ_size, empty_idx=0, ignore_index=255):
        super().__init__()
        self.occ_size = occ_size
        self.empty_idx = empty_idx
        self.ignore_index = ignore_index

    def __call__(self, results):
        gt_occ = results['gt_occ']  # (X, Y, Z)
        assert isinstance(gt_occ, torch.Tensor), "gt_occ must be a torch.Tensor"
        assert gt_occ.ndim == 3, f"gt_occ must have shape (X, Y, Z), got {tuple(gt_occ.shape)}"

        target_x, target_y, target_z = self.occ_size
        x, y, z = gt_occ.shape

        # No change needed
        if (x, y, z) == (target_x, target_y, target_z):
            results['occ_size'] = np.array(self.occ_size)
            return results

        # Only downsampling is supported (no upsampling). Each axis must divide evenly.
        assert target_x <= x and x % target_x == 0, \
            f"X dimension must be divisible by target and not larger. Got X={x}, target_x={target_x}"
        assert target_y <= y and y % target_y == 0, \
            f"Y dimension must be divisible by target and not larger. Got Y={y}, target_y={target_y}"
        assert target_z <= z and z % target_z == 0, \
            f"Z dimension must be divisible by target and not larger. Got Z={z}, target_z={target_z}"

        gt_occ_ds = gt_occ

        factor_x = x // target_x
        factor_y = y // target_y
        factor_z = z // target_z

        if factor_x > 1:
            gt_occ_ds = self.downsample_along_axis(gt_occ_ds, factor_x, axis=0)
        if factor_y > 1:
            gt_occ_ds = self.downsample_along_axis(gt_occ_ds, factor_y, axis=1)
        if factor_z > 1:
            gt_occ_ds = self.downsample_along_axis(gt_occ_ds, factor_z, axis=2)

        results['gt_occ'] = gt_occ_ds
        results['occ_size'] = np.array(self.occ_size)
        return results

    def _merge_pair(self, a, b):
        # a, b: (...,) int64
        out = torch.empty_like(a)

        same = (a == b)
        out[same] = a[same]

        a_ign = (a == self.ignore_index)
        b_ign = (b == self.ignore_index)

        # one ignore, take the other
        only_b = a_ign & (~b_ign)
        out[only_b] = b[only_b]

        only_a = b_ign & (~a_ign)
        out[only_a] = a[only_a]

        # both ignore
        both_ign = a_ign & b_ign
        out[both_ign] = self.ignore_index

        # remaining: both non-ignore and different
        remaining = ~(same | only_b | only_a | both_ign)
        if remaining.any():
            a_emp = (a == self.empty_idx)
            b_emp = (b == self.empty_idx)

            # prefer non-empty over empty
            choose_b = remaining & a_emp & (~b_emp)
            out[choose_b] = b[choose_b]

            choose_a = remaining & b_emp & (~a_emp)
            out[choose_a] = a[choose_a]

            # both empty or both non-empty and different: pick min for determinism
            rem2 = remaining & ~(choose_a | choose_b)
            if rem2.any():
                out[rem2] = torch.minimum(a[rem2], b[rem2])

        return out

    def downsample_along_axis(self, labels, factor, axis):
        out = labels
        f = factor
        while f > 1:
            assert (f % 2) == 0, f"Only even downsample factors supported along axis {axis}"
            idx_even = [slice(None), slice(None), slice(None)]
            idx_odd = [slice(None), slice(None), slice(None)]
            idx_even[axis] = slice(0, None, 2)
            idx_odd[axis] = slice(1, None, 2)
            a = out[tuple(idx_even)]
            b = out[tuple(idx_odd)]
            out = self._merge_pair(a, b)
            f //= 2
        return out
