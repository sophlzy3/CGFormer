import torch
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module(name='FilterDepth')
class FilterDepth(object):
    def __init__(self, min_depth=0, max_depth=100, background_index=0):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.background_index = background_index

    def __call__(self, results):
        gt_depths = results['gt_depths']
        gt_semantics = results['gt_semantics']
        assert gt_depths is not None, "gt_depths is required"
        assert gt_semantics is not None, "gt_semantics is required"

        # set depth to max depth if segmentation mask is 0 (background)
        fg_mask = gt_semantics != self.background_index
        gt_depths[~fg_mask] = self.max_depth

        results['gt_depths'] = gt_depths
        return results
