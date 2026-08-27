# CGFormer stage 1 (seg + depth pretraining) on ALLO.
# Trains CGFormerSegDepth: 2D backbone + neck + GeometryDepth_Net + plugin_segmentation_head.
# Produces last.ckpt, which stage 2 (CGFormer-Efficient-Swin-ALLO.py) loads via --load.
#
# Paths are read from the environment so the job script can point at the staged dataset.
# __import__ is used inline so no module object ends up in the config dump.
data_root = __import__('os').environ.get('ALLO_3D_ROOT', '/ws/dataset/allo_3d')
depth_root = __import__('os').environ.get('ALLO_MONODEPTH_ROOT', '/ws/dataset/allo_3d/mono_depth')

dataset_type = 'ALLODataset'
point_cloud_range = [0, -12.8, -12.8, 25.6, 12.8, 12.8]
occ_size = [128, 128, 128]
lss_downsample = [2, 2, 2]

voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
voxel_size = [voxel_x, voxel_y, voxel_z]

# Must match stage 2 exactly: 'dbound' fixes the depth-bin count D of the depth net,
# so a mismatch would make the pretrained depth weights unloadable.
grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x * lss_downsample[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y * lss_downsample[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z * lss_downsample[2]],
    'dbound': [1.0, 29.0, 0.25],   # (29-1)/0.25 = 112 bins
}

empty_idx = 0

# Raw ALLO labels hold 9 classes; AnomalyMapLabels drops the anomaly and IgnoreLabels
# drops celestial_bodies, leaving these 7 trainable classes.
class_names = [
    "background", "robotic_arms", "solar_arrays", "pressurized_modules",
    "airlock_docking_ports", "truss_other", "modules_other"
]
num_class = len(class_names)

indices_to_ignore = [7, 8]  #* post-remap 7: "celestial_bodies", 8: "anomaly"

# dataset config #
bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    flip_dz_ratio=0
)

data_config={
    'input_size': (672, 1344),  #* to be divisible by 32
    'resize': (0., 0.),
    'rot': (0.0, 0.0 ),
    'flip': True,
    'flip_v': False,   # vertical flip is geometrically invalid here (cost ~10 mIoU on VoxDet)
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

pixel_mean = [0.18615150076452516, 0.18466143346552077, 0.1804993998048694]
pixel_std = [0.24937694186317588, 0.24854584057424572, 0.24403493825212044]

# anomaly_raw_index=7: OUR ALLO writes the injected anomaly at raw index 7 (voxel AND seg),
# not the canonical raw-1. Do not remove -- see CLAUDE.md.
train_pipeline = [
    dict(type='LoadMultiViewImageFromFilesWithSegDepth', data_config=data_config, is_train=True,
         color_jitter=(0.4, 0.4, 0.4), mean=pixel_mean, std=pixel_std),
    # Kept even though this stage has no occupancy loss: it appends the bda matrix to
    # img_inputs, which depth_net.get_mlp_input() reads as its 6th argument. Dropping it
    # would silently pass cam2lidar as bda and desync stage 1 from stage 2.
    dict(type='LoadAnnotationOcc', bda_aug_conf=bda_aug_conf, apply_bda=False,
            is_train=True, point_cloud_range=point_cloud_range),
    dict(type='AnomalyMapLabels', num_classes=num_class, with_seg=True, anomaly_index=1,
         with_anomalies=False, anomaly_raw_index=7),
    dict(type='IgnoreLabels', indices_to_ignore=indices_to_ignore, ignore_index=255),
    dict(type='FilterDepth', min_depth=0, max_depth=100, background_index=0),
    dict(type='CollectData', keys=['img_inputs', 'gt_semantics'],
            meta_keys=['pc_range', 'occ_size', 'raw_img', 'stereo_depth', 'img_shape', 'gt_depths']),
]

trainset_config=dict(
    type=dataset_type,
    data_root=data_root,
    depth_root=depth_root,
    pipeline=train_pipeline,
    split='train',
    occ_size=occ_size,
    pc_range=point_cloud_range,
    test_mode=False,
)

test_pipeline = [
    dict(type='LoadMultiViewImageFromFilesWithSegDepth', data_config=data_config, is_train=False,
         color_jitter=None, mean=pixel_mean, std=pixel_std),
    dict(type='LoadAnnotationOcc', bda_aug_conf=bda_aug_conf, apply_bda=False,
            is_train=False, point_cloud_range=point_cloud_range),
    dict(type='AnomalyMapLabels', num_classes=num_class, with_seg=True, anomaly_index=1,
         with_anomalies=False, anomaly_raw_index=7),
    dict(type='IgnoreLabels', indices_to_ignore=indices_to_ignore, ignore_index=255),
    dict(type='FilterDepth', min_depth=0, max_depth=100, background_index=0),
    dict(type='CollectData', keys=['img_inputs', 'gt_semantics'],
            meta_keys=['pc_range', 'occ_size', 'raw_img', 'stereo_depth', 'img_shape', 'gt_depths']),
]

testset_config=dict(
    type=dataset_type,
    data_root=data_root,
    depth_root=depth_root,
    pipeline=test_pipeline,
    split='test',
    occ_size=occ_size,
    pc_range=point_cloud_range,
    test_mode=True,
)

data = dict(
    train=trainset_config,
    val=testset_config,
    test=testset_config
)

train_dataloader_config = dict(
    batch_size=2,
    num_workers=8)

test_dataloader_config = dict(
    batch_size=2,
    num_workers=8)

# model params #
numC_Trans = 128
voxel_channels = [128, 256, 512]
voxel_out_indices = (0, 1, 2)
voxel_out_channels = [128, 128, 128]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    type='CGFormerSegDepth',
    img_backbone=dict(
        type='CustomEfficientNet',
        arch='b7',
        drop_path_rate=0.2,
        frozen_stages=0,
        norm_eval=False,
        out_indices=(2, 3, 4, 5, 6),
        with_cp=True,
        init_cfg=dict(type='Pretrained', prefix='backbone',
        checkpoint='./ckpt/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth'),
    ),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[48, 80, 224, 640, 2560],
        upsample_strides=[0.5, 1, 2, 4, 4],
        out_channels=[128, 128, 128, 128, 128]),
    depth_net=dict(
        type='GeometryDepth_Net',
        downsample=8,
        numC_input=640,
        numC_Trans=numC_Trans,
        cam_channels=33,
        grid_config=grid_config,
        loss_depth_type='kld'
    ),
    plugin_head=dict(
        type='plugin_segmentation_head',
        in_channels=numC_Trans,
        out_channel_list=[128, 64, 32],
        num_class=num_class,
    )
)

"""Training params."""
learning_rate=3e-4
training_steps=50000

optimizer = dict(
    type="AdamW",
    lr=learning_rate,
    weight_decay=0.01
)

lr_scheduler = dict(
    type="OneCycleLR",
    max_lr=learning_rate,
    total_steps=training_steps + 10,
    pct_start=0.05,
    cycle_momentum=False,
    anneal_strategy="cos",
    interval="step",
    frequency=1
)
