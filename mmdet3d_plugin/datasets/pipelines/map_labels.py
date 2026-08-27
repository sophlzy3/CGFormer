import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module(name='AnomalyMapLabels')
class AnomalyMapLabels(object):
    def __init__(self, num_classes, with_seg=True, anomaly_index=1, ignore_index=255, with_anomalies=False,
                 anomaly_raw_index=None):
        super().__init__()
        self.num_classes = num_classes
        self.with_seg = with_seg
        self.anomaly_index = anomaly_index
        self.with_anomalies = with_anomalies
        self.ignore_index = ignore_index
        # OUR ALLO stores the injected anomaly at a non-standard raw index (7), not the
        # canonical anomaly_index (1). When set, swap anomaly_raw_index <-> anomaly_index
        # up front so the remap below (and class_names) stay correct. Applied to BOTH gt_occ
        # and gt_semantics (both routed through remap_labels), keeping voxel+seg consistent.
        # Leave None (default) for John's data, which already uses the canonical convention.
        self.anomaly_raw_index = anomaly_raw_index

    def remap_labels(self, label_tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(label_tensor, torch.Tensor) and isinstance(label_tensor, np.ndarray):
            label_tensor = torch.from_numpy(label_tensor)
        label_tensor = label_tensor.long()

        if self.anomaly_raw_index is not None and self.anomaly_raw_index != self.anomaly_index:
            swapped = label_tensor.clone()
            swapped[label_tensor == self.anomaly_raw_index] = self.anomaly_index
            swapped[label_tensor == self.anomaly_index] = self.anomaly_raw_index
            label_tensor = swapped

        out = label_tensor.clone()

        # preserve ignore labels
        ignore_mask = (label_tensor == self.ignore_index)

        # shift classes greater than the anomaly index down by 1
        greater_mask = (label_tensor > self.anomaly_index) & (~ignore_mask)
        out[greater_mask] = out[greater_mask] - 1

        # map anomaly to desired target
        anomaly_mask = (label_tensor == self.anomaly_index)
        if self.with_anomalies:
            out[anomaly_mask] = self.num_classes
        else:
            out[anomaly_mask] = self.ignore_index

        # restore ignore labels explicitly
        out[ignore_mask] = self.ignore_index
        return out

    def __call__(self, results):
        """
        Remaps segmentation labels (with or without anomalies) to a consecutive range of labels.
        """
        gt_occ = results['gt_occ']
        gt_semantics = results.get('gt_semantics', None)

        if gt_semantics is None and self.with_seg:
            raise ValueError("gt_semantics is required when with_seg is True")

        results['gt_occ'] = self.remap_labels(gt_occ)
        if self.with_seg and gt_semantics is not None:
            results['gt_semantics'] = self.remap_labels(gt_semantics)

        return results


@PIPELINES.register_module(name='IgnoreLabels')
class IgnoreLabels(object):
    def __init__(self, indices_to_ignore, ignore_index=255):
        super().__init__()
        self.indices_to_ignore = indices_to_ignore
        self.ignore_index = ignore_index

    def __call__(self, results):
        """
        Ignores specified labels in the labels.
        """
        gt_occ = results['gt_occ']
        if not isinstance(gt_occ, torch.Tensor):
            gt_occ = torch.from_numpy(gt_occ)
        gt_occ = gt_occ.clone()
        gt_semantics = results.get('gt_semantics', None)
        if gt_semantics is not None:
            gt_semantics = gt_semantics.clone()

        for index in self.indices_to_ignore:
            gt_occ[gt_occ == index] = self.ignore_index
            if gt_semantics is not None:
                gt_semantics[gt_semantics == index] = self.ignore_index

        results['gt_occ'] = gt_occ
        if gt_semantics is not None:
            results['gt_semantics'] = gt_semantics

        return results
