"""
Configuration complète pour TrackNetV3
Adapté de tracknetv2_b2e30_adamw_lr1e-4.py
"""

from pathlib import Path

# ------------------- 1. MODÈLE (Model) -------------------
model = dict(
    type='TrackNetV3',  # Doit correspondre au nom enregistré dans MODELS
    backbone=dict(
        type='TrackNetV3Backbone',
        in_dim=9,
        channels=[64, 128, 256, 512],
        use_depthwise=False
    ),
    neck=dict(
        type='TrackNetV3Neck',
        channels=[64, 128, 256, 512],
        use_depthwise=False
    ),
    head=dict(
        type='TrackNetV3Head',
        in_channels=64,
        out_channels=3
    )
)

# ------------------- 2. DONNÉES (Data) -------------------
input_size = (288, 512)  #
original_size = (720, 1280) 
data_root = './data/benchmark'

pipeline = [
    dict(type='LoadMultiImagesFromPaths', to_rgb=True),
    dict(type='Resize', keys=['path_prev', 'path', 'path_next'], size=input_size),
    dict(type='ConcatChannels',
         keys=['path_prev', 'path', 'path_next'],
         output_key='image'),
    dict(type='LoadAndFormatMultiTargets',
         keys=['gt_path_prev', 'gt_path', 'gt_path_next'],
         output_key='target'),
    dict(type='Finalize',
         image_key='image',
         final_keys=['image', 'target', 'coords', 'visibility', 'original_info'])
]

data = dict(
    samples_per_gpu=2, # Tu peux augmenter à 4 si ta GPU le permet (V3 est optimisée)
    workers_per_gpu=4,
    train=dict(
        type='TennisDataset',
        data_dir=data_root,
        csv_path=f'{data_root}/labels_context_train.csv',
        input_height=input_size[0],
        input_width=input_size[1],
        pipeline=pipeline
    ),
    val=dict(
        type='TennisDataset',
        data_dir=data_root,
        csv_path=f'{data_root}/labels_context_val.csv',
        input_height=input_size[0],
        input_width=input_size[1],
        pipeline=pipeline
    )
)

# ------------------- 3. PERTE (Loss) -------------------
loss = dict(
    type='TrackNetV2Loss' # La V3 utilise généralement la même WBCE loss que la V2
)

# ------------------- 4. OPTIMISATION -------------------
optimizer = dict(type='AdamW', lr=1e-4)

optimizer_config = dict(
    grad_clip=dict(max_norm=1.0)
)

lr_config = dict(
    policy='Step',
    step=[20, 25],
    gamma=0.1
)

# ------------------- 5. ÉVALUATION -------------------
evaluation = dict(
    interval=1,
    metric=dict(
        type='TrackNetV2Metric',
        min_dist=4,
        original_size=original_size
    )
)

# ------------------- 6. RUNTIME -------------------
total_epochs = 30
work_dir = f'./workdirs/{Path(__file__).stem}' 

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)

custom_hooks = [
    dict(type='ValidationVisualizerV2Hook', num_samples_to_save=100, original_size=original_size)
]

seed = 42
resume_from = None