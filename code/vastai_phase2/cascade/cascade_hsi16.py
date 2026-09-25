auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'cascade_hsi',
    ])
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=2,
        max_keep_ckpts=2,
        rule='greater',
        save_best='coco/bbox_mAP',
        type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = '/workspace/cascade/cascade_r50_coco_16band.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        in_channels=16,
        init_cfg=None,
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=None,
        pad_size_divisor=32,
        std=None,
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        type='FPN'),
    roi_head=dict(
        bbox_head=[
            dict(
                bbox_coder=dict(
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.1,
                        0.1,
                        0.2,
                        0.2,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(beta=1.0, loss_weight=1.0, type='SmoothL1Loss'),
                loss_cls=dict(
                    loss_weight=1.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=False),
                num_classes=18,
                reg_class_agnostic=True,
                roi_feat_size=7,
                type='Shared2FCBBoxHead'),
            dict(
                bbox_coder=dict(
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.05,
                        0.05,
                        0.1,
                        0.1,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(beta=1.0, loss_weight=1.0, type='SmoothL1Loss'),
                loss_cls=dict(
                    loss_weight=1.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=False),
                num_classes=18,
                reg_class_agnostic=True,
                roi_feat_size=7,
                type='Shared2FCBBoxHead'),
            dict(
                bbox_coder=dict(
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.033,
                        0.033,
                        0.067,
                        0.067,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(beta=1.0, loss_weight=1.0, type='SmoothL1Loss'),
                loss_cls=dict(
                    loss_weight=1.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=False),
                num_classes=18,
                reg_class_agnostic=True,
                roi_feat_size=7,
                type='Shared2FCBBoxHead'),
        ],
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        num_stages=3,
        stage_loss_weights=[
            1,
            0.5,
            0.25,
        ],
        type='CascadeRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(
            beta=0.1111111111111111, loss_weight=1.0, type='SmoothL1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=300,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.001),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=[
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=False,
                    min_pos_iou=0.5,
                    neg_iou_thr=0.5,
                    pos_iou_thr=0.5,
                    type='MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=False,
                    min_pos_iou=0.6,
                    neg_iou_thr=0.6,
                    pos_iou_thr=0.6,
                    type='MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=False,
                    min_pos_iou=0.7,
                    neg_iou_thr=0.7,
                    pos_iou_thr=0.7,
                    type='MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
        ],
        rpn=dict(
            allowed_border=0,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=2000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    type='CascadeRCNN')
optim_wrapper = dict(
    loss_scale='dynamic',
    optimizer=dict(lr=0.005, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=24,
        gamma=0.1,
        milestones=[
            16,
            22,
        ],
        type='MultiStepLR'),
]
randomness = dict(seed=42)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='test.json',
        data_prefix=dict(img='test/'),
        data_root='/workspace/cascade/data',
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        metainfo=dict(
            classes=(
                'apple',
                'apple_plastic',
                'badminton',
                'banana',
                'banana_plastic',
                'car',
                'car_toy',
                'charger_head',
                'e-bike',
                'egg',
                'egg_plastic',
                'egg_wood',
                'orange',
                'orange_plastic',
                'people',
                'rubik',
                'stone_block',
                'table_tennis',
            )),
        pipeline=[
            dict(
                mean=[
                    33.470811263394104,
                    28.074022453048325,
                    28.916012518094753,
                    28.81981274219943,
                    51.22605775193261,
                    33.58337190878782,
                    36.94786652967696,
                    47.25410415987701,
                    54.83571272883085,
                    60.98661500261104,
                    52.199984635868844,
                    56.5599994424368,
                    60.96106379609638,
                    74.46624834162849,
                    77.58655114499173,
                    83.88652410153362,
                ],
                std=[
                    38.182542157556526,
                    35.67720514998361,
                    35.88285260331031,
                    35.79748920602295,
                    46.26948969622273,
                    38.379684414902826,
                    40.33703600311197,
                    47.02433866881178,
                    50.4395241728424,
                    53.35242165949097,
                    48.885807517846736,
                    51.06496415197088,
                    52.602898613370485,
                    57.200104260740694,
                    58.589467291699314,
                    60.96704711348963,
                ],
                type='LoadNpyImage'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='/workspace/cascade/data/test.json',
    classwise=True,
    metric='bbox',
    outfile_prefix='/workspace/cascade/holdout_preds',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=24, type='EpochBasedTrainLoop', val_interval=2)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=4,
    dataset=dict(
        ann_file='train.json',
        data_prefix=dict(img='train/'),
        data_root='/workspace/cascade/data',
        filter_cfg=dict(filter_empty_gt=True, min_size=0),
        metainfo=dict(
            classes=(
                'apple',
                'apple_plastic',
                'badminton',
                'banana',
                'banana_plastic',
                'car',
                'car_toy',
                'charger_head',
                'e-bike',
                'egg',
                'egg_plastic',
                'egg_wood',
                'orange',
                'orange_plastic',
                'people',
                'rubik',
                'stone_block',
                'table_tennis',
            )),
        pipeline=[
            dict(
                mean=[
                    33.470811263394104,
                    28.074022453048325,
                    28.916012518094753,
                    28.81981274219943,
                    51.22605775193261,
                    33.58337190878782,
                    36.94786652967696,
                    47.25410415987701,
                    54.83571272883085,
                    60.98661500261104,
                    52.199984635868844,
                    56.5599994424368,
                    60.96106379609638,
                    74.46624834162849,
                    77.58655114499173,
                    83.88652410153362,
                ],
                std=[
                    38.182542157556526,
                    35.67720514998361,
                    35.88285260331031,
                    35.79748920602295,
                    46.26948969622273,
                    38.379684414902826,
                    40.33703600311197,
                    47.02433866881178,
                    50.4395241728424,
                    53.35242165949097,
                    48.885807517846736,
                    51.06496415197088,
                    52.602898613370485,
                    57.200104260740694,
                    58.589467291699314,
                    60.96704711348963,
                ],
                type='LoadNpyImage'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        test_mode=False,
        type='CocoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='val.json',
        data_prefix=dict(img='val/'),
        data_root='/workspace/cascade/data',
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        metainfo=dict(
            classes=(
                'apple',
                'apple_plastic',
                'badminton',
                'banana',
                'banana_plastic',
                'car',
                'car_toy',
                'charger_head',
                'e-bike',
                'egg',
                'egg_plastic',
                'egg_wood',
                'orange',
                'orange_plastic',
                'people',
                'rubik',
                'stone_block',
                'table_tennis',
            )),
        pipeline=[
            dict(
                mean=[
                    33.470811263394104,
                    28.074022453048325,
                    28.916012518094753,
                    28.81981274219943,
                    51.22605775193261,
                    33.58337190878782,
                    36.94786652967696,
                    47.25410415987701,
                    54.83571272883085,
                    60.98661500261104,
                    52.199984635868844,
                    56.5599994424368,
                    60.96106379609638,
                    74.46624834162849,
                    77.58655114499173,
                    83.88652410153362,
                ],
                std=[
                    38.182542157556526,
                    35.67720514998361,
                    35.88285260331031,
                    35.79748920602295,
                    46.26948969622273,
                    38.379684414902826,
                    40.33703600311197,
                    47.02433866881178,
                    50.4395241728424,
                    53.35242165949097,
                    48.885807517846736,
                    51.06496415197088,
                    52.602898613370485,
                    57.200104260740694,
                    58.589467291699314,
                    60.96704711348963,
                ],
                type='LoadNpyImage'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='/workspace/cascade/data/val.json',
    classwise=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/workspace/cascade/work'
