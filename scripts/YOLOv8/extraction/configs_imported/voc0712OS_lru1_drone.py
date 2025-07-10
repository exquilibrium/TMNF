# dataset settings

dataset_type = 'XMLDataset' ### <<<<<<<<<<---------- Important ---------->>>>>>>>>>
#data_root = '/volume/hot_storage/slurm_data/chen_le/ARCHES/lru1_all'
data_root = '/media/chen/76AECF8EAECF4579/data/lru1_all'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

CS_suffix = "_CS_drone"
#voc_os_classes=["drone", "lander", "lru2"] # !!! Using custom
voc_os_classes=["lander", "lru2", "drone"] # !!! Excluded OS classes need to be on the right
voc_cs_classes=["lander", "lru2"]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            classes=voc_cs_classes, # CS
            ann_file=[
                data_root + f'/ImageSets/Main{CS_suffix}/train.txt',
            ],
            img_prefix=[data_root],
            pipeline=train_pipeline)),
    trainCS=dict(
        type=dataset_type,
        classes=voc_cs_classes, # CS
        ann_file=data_root + f'/ImageSets/Main{CS_suffix}/train.txt',
        img_prefix=data_root,
        pipeline=test_pipeline), ### <<<<<<<<<<---------- test pipeline ---------->>>>>>>>>> 
    val=dict(
        type=dataset_type,
        classes=voc_cs_classes, # CS
        ann_file=data_root + f'/ImageSets/Main{CS_suffix}/val.txt',
        img_prefix=data_root,
        pipeline=test_pipeline),
    testCS=dict(
        type=dataset_type,
        classes=voc_cs_classes, # CS
        ann_file=data_root + f'/ImageSets/Main{CS_suffix}/test.txt',
        img_prefix=data_root,
        pipeline=test_pipeline),
    testOS=dict(
        type=dataset_type,
        classes=voc_os_classes, # OS
        ann_file=data_root + '/ImageSets/Main/test.txt',
        img_prefix=data_root,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')
