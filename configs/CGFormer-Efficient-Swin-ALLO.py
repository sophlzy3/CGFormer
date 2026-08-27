# CGFormer stage 2 (semantic scene completion) on ALLO.
# Cold-start it from the stage-1 checkpoint produced by CGFormer-Efficient-Swin-ALLO-Pretrain.py:
#   python main.py --config_path configs/CGFormer-Efficient-Swin-ALLO.py --load <stage1 last.ckpt>
# (--load sets load_from; use --ckpt_path only to resume this stage's own trainer state.)
data_root = __import__('os').environ.get('ALLO_3D_ROOT', '/ws/dataset/allo_3d')
depth_root = __import__('os').environ.get('ALLO_MONODEPTH_ROOT', '/ws/dataset/allo_3d/mono_depth')

dataset_type = 'ALLODataset'
point_cloud_range = [0, -12.8, -12.8, 25.6, 12.8, 12.8]
occ_size = [128, 128, 128]

# Voxel counts per final class on OUR ALLO train split at 128^3 (same list VoxDet uses).
allo_class_frequencies = [
    57254772110,
    25880433,
    142881366,
    1858037410,
    222495459,
    1313280749,
    281078841,
]

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
    dict(type='LoadAnnotationOcc', bda_aug_conf=bda_aug_conf, apply_bda=False,
            is_train=True, point_cloud_range=point_cloud_range),
    dict(type='AnomalyMapLabels', num_classes=num_class, with_seg=True, anomaly_index=1,
         with_anomalies=False, anomaly_raw_index=7),
    dict(type='IgnoreLabels', indices_to_ignore=indices_to_ignore, ignore_index=255),
    dict(type='FilterDepth', min_depth=0, max_depth=100, background_index=0),
    dict(type='DownsampleVoxels', occ_size=occ_size),
    dict(type='CollectData', keys=['img_inputs', 'gt_occ'],
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
    dict(type='DownsampleVoxels', occ_size=occ_size),
    dict(type='CollectData', keys=['img_inputs', 'gt_occ'],
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
    batch_size=1,
    num_workers=4)

test_dataloader_config = dict(
    batch_size=1,
    num_workers=4)

# model
numC_Trans = 128
lss_downsample = [2, 2, 2]
voxel_out_channels = [128]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]

# 'dbound' must match the stage-1 config: it fixes the depth-bin count D of the depth net.
grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x * lss_downsample[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y * lss_downsample[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z * lss_downsample[2]],
    'dbound': [1.0, 29.0, 0.25],   # (29-1)/0.25 = 112 bins
}

# occ_size / lss_downsample -> the query volume the VoxFormer head and TPV branch operate on
volume_h = occ_size[0] // lss_downsample[0]
volume_w = occ_size[1] // lss_downsample[1]
volume_z = occ_size[2] // lss_downsample[2]

_num_layers_cross_ = 3
_num_points_cross_ = 8
_num_levels_ = 1
_num_cams_ = 1
_dim_ = 128
_pos_dim_ = _dim_//2

_num_layers_self_ = 2
_num_points_self_ = 8

model = dict(
    type='CGFormer',
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
        loss_depth_type='kld',
        loss_depth_weight=0.0001,
    ),
    img_view_transformer=dict(
        type='LSSViewTransformer',
        downsample=8,
        grid_config=grid_config,
        data_config=data_config,
    ),
    proposal_layer=dict(
        type='VoxelProposalLayer',
        point_cloud_range=point_cloud_range,
        input_dimensions=[volume_h, volume_w, volume_z],
        data_config=data_config,
        init_cfg=None
    ),
    VoxFormer_head=dict(
        type='VoxFormerHead',
        volume_h=volume_h,
        volume_w=volume_w,
        volume_z=volume_z,
        data_config=data_config,
        point_cloud_range=point_cloud_range,
        embed_dims=_dim_,
        cross_transformer=dict(
           type='PerceptionTransformer_DFA3D',
           rotate_prev_bev=True,
           use_shift=True,
           embed_dims=_dim_,
           num_cams = _num_cams_,
           encoder=dict(
               type='VoxFormerEncoder_DFA3D',
               num_layers=_num_layers_cross_,
               pc_range=point_cloud_range,
               data_config=data_config,
               num_points_in_pillar=8,
               return_intermediate=False,
               transformerlayers=dict(
                   type='VoxFormerLayer',
                   attn_cfgs=[
                       dict(
                           type='DeformCrossAttention_DFA3D',
                           pc_range=point_cloud_range,
                           num_cams=_num_cams_,
                           deformable_attention=dict(
                               type='MSDeformableAttention3D_DFA3D',
                               embed_dims=_dim_,
                               num_points=_num_points_cross_,
                               num_levels=_num_levels_),
                           embed_dims=_dim_,
                       )
                   ],
                   ffn_cfgs=dict(
                       type='FFN',
                       embed_dims=_dim_,
                       feedforward_channels=1024,
                       num_fcs=2,
                       ffn_drop=0.,
                       act_cfg=dict(type='ReLU', inplace=True),
                   ),
                   feedforward_channels=_dim_ * 2,
                   ffn_dropout=0.1,
                   operation_order=('cross_attn', 'norm', 'ffn', 'norm')))),
        self_transformer=dict(
           type='PerceptionTransformer_DFA3D',
           rotate_prev_bev=True,
           use_shift=True,
           embed_dims=_dim_,
           num_cams = _num_cams_,
           use_level_embeds = False,
           use_cams_embeds = False,
           encoder=dict(
               type='VoxFormerEncoder',
               num_layers=_num_layers_self_,
               pc_range=point_cloud_range,
               data_config=data_config,
               num_points_in_pillar=8,
               return_intermediate=False,
               transformerlayers=dict(
                   type='VoxFormerLayer',
                   attn_cfgs=[
                       dict(
                           type='DeformSelfAttention',
                           embed_dims=_dim_,
                           num_levels=1,
                           num_points=_num_points_self_)
                   ],
                   ffn_cfgs=dict(
                       type='FFN',
                       embed_dims=_dim_,
                       feedforward_channels=1024,
                       num_fcs=2,
                       ffn_drop=0.,
                       act_cfg=dict(type='ReLU', inplace=True),
                   ),
                   feedforward_channels=_dim_ * 2,
                   ffn_dropout=0.1,
                   operation_order=('self_attn', 'norm', 'ffn', 'norm')))),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=512,
            col_num_embed=512,
           ),
        mlp_prior=True
    ),

    occ_encoder_backbone=dict(
        type='Fuser',
        embed_dims=128,
        global_aggregator=dict(
            type='TPVGlobalAggregator',
            embed_dims=_dim_,
            split=[8,8,8],
            grid_size=[volume_h, volume_w, volume_z],
            global_encoder_backbone=dict(
                type='Swin',
                embed_dims=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                in_channels=128,
                patch_size=4,
                strides=[1,2,2,2],
                frozen_stages=-1,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.2,
                patch_norm=True,
                out_indices=[1,2,3],
                with_cp=False,
                convert_weights=True,
                init_cfg=dict(
                    type='Pretrained',
                    checkpoint='./ckpt/swin_tiny_patch4_window7_224.pth'),
                    ),
            global_encoder_neck=dict(
                type='GeneralizedLSSFPN',
                in_channels=[192, 384, 768],
                out_channels=_dim_,
                start_level=0,
                num_outs=3,
                norm_cfg=dict(
                type='BN2d',
                requires_grad=True,
                track_running_stats=False),
                act_cfg=dict(
                type='ReLU',
                inplace=True),
                upsample_cfg=dict(
                mode='bilinear',
                align_corners=False),
            ),
        ),
        local_aggregator=dict(
            type='LocalAggregator',
            local_encoder_backbone=dict(
                type='CustomResNet3D',
                numC_input=128,
                num_layer=[2, 2, 2],
                num_channels=[128, 128, 128],
                stride=[1, 2, 2]
            ),
            local_encoder_neck=dict(
                type='GeneralizedLSSFPN',
                in_channels=[128, 128, 128],
                out_channels=_dim_,
                start_level=0,
                num_outs=3,
                norm_cfg=norm_cfg,
                conv_cfg=dict(type='Conv3d'),
                act_cfg=dict(
                    type='ReLU',
                    inplace=True),
                upsample_cfg=dict(
                    mode='trilinear',
                    align_corners=False
                )
            )
        )
    ),
    pts_bbox_head=dict(
        type='OccHead',
        in_channels=[sum(voxel_out_channels)],
        out_channel=num_class,
        empty_idx=0,
        num_level=1,
        with_cp=True,
        occ_size=occ_size,
        loss_weight_cfg = {
                "loss_voxel_ce_weight": 1.0,
                "loss_voxel_sem_scal_weight": 1.0,
                "loss_voxel_geo_scal_weight": 1.0
        },
        conv_cfg=dict(type='Conv3d', bias=False),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        class_frequencies=allo_class_frequencies
    )
)

"""Training params."""
learning_rate=3e-4
training_steps=100000   # VoxDet on ALLO peaked around step ~100k; shorter runs undertrain

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

load_from = None   # set by cg2-allo.sh via `--load <stage-1 last.ckpt>`
