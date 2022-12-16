models = list()  # type: ignore[var-annotated]
hooks = dict()  # type: ignore[var-annotated]
adapts = {
    'attn_spatial': dict(
        inputs=('neck', ),
        parallel=True,
        action=dict(
            type='AbsMeanSpatialAttention',
            temperature=0.5,
        ),
    ),
    'teacher_attn_spatial': dict(
        inputs=('teacher_neck', ),
        parallel=True,
        action=dict(
            type='AbsMeanSpatialAttention',
            temperature=0.5,
        ),
    ),
    'attn_channel': dict(
        inputs=('neck', ),
        parallel=True,
        action=dict(
            type='AbsMeanChannelAttention',
            temperature=0.5,
        ),
    ),
    'teacher_attn_channel': dict(
        inputs=('teacher_neck', ),
        parallel=True,
        action=dict(
            type='AbsMeanChannelAttention',
            temperature=0.5,
        ),
    ),
    'masks': dict(
        inputs=('img_shape', 'gt_bboxes'),
        action=dict(
            type='FGDMask',
            neg_gain=0.5,
            strides=[8, 16, 32, 64, 128],
            ceil_mode=True,
        ),
    ),
    'global_': dict(
        inputs=('neck', ),
        parallel=5,
        action=dict(
            type='mmcv_ContextBlock',
            in_channels=4,
            ratio=0.5,
        ),
    ),
    'teacher_global': dict(
        inputs=('teacher_neck', ),
        parallel=5,
        action=dict(
            type='mmcv_ContextBlock',
            in_channels=4,
            ratio=0.5,
        ),
    ),
}
losses = {
    'loss_feat': dict(
        inputs=(
            'neck',
            'teacher_neck',
            'teacher_attn_spatial',
            'teacher_attn_channel',
            'masks',
        ),
        parallel=True,
        action=dict(
            type='FGDLoss',
            weight=5e-4,
            reduction='sum',
        ),
    ),
    'loss_attn_spatial': dict(
        inputs=('attn_spatial', 'teacher_attn_spatial'),
        parallel=True,
        action=dict(
            type='L1Loss',
            weight=2.5e-4,
            reduction='sum',
        ),
    ),
    'loss_attn_channel': dict(
        inputs=('attn_channel', 'teacher_attn_channel'),
        parallel=True,
        action=dict(
            type='L1Loss',
            weight=2.5e-4,
            reduction='sum',
        ),
    ),
    'loss_global': dict(
        inputs=('global_', 'teacher_global'),
        parallel=True,
        action=dict(
            type='MSELoss',
            weight=2.5e-6,
            reduction='sum',
        ),
    ),
}
