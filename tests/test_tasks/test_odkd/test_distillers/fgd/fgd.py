models = list()  # type: ignore[var-annotated]
hook_pipelines = list()  # type: ignore[var-annotated]
adapt_pipelines = {
    'attn_spatial': dict(
        type='SharedParallelPipeline',
        args=('neck', ),
        callable_=dict(
            type='SpatialAttention',
            temperature=0.5,
        ),
    ),
    'teacher_attn_spatial': dict(
        type='SharedParallelPipeline',
        args=('teacher_neck', ),
        callable_=dict(
            type='SpatialAttention',
            temperature=0.5,
        ),
    ),
    'attn_channel': dict(
        type='SharedParallelPipeline',
        args=('neck', ),
        callable_=dict(
            type='ChannelAttention',
            temperature=0.5,
        ),
    ),
    'teacher_attn_channel': dict(
        type='SharedParallelPipeline',
        args=('teacher_neck', ),
        callable_=dict(
            type='ChannelAttention',
            temperature=0.5,
        ),
    ),
    'masks': dict(
        type='VanillaPipeline',
        args=('img_shape', 'gt_bboxes'),
        callable_=dict(
            type='FGDMask',
            neg_gain=0.5,
            strides=[8, 16, 32, 64, 128],
            ceil_mode=True,
        ),
    ),
    'global_': dict(
        type='ParallelPipeline',
        args=('neck', ),
        callables=[dict(
            type='mmcv_ContextBlock',
            in_channels=4,
            ratio=0.5,
        )] * 5,
    ),
    'teacher_global': dict(
        type='ParallelPipeline',
        args=('teacher_neck', ),
        callables=[dict(
            type='mmcv_ContextBlock',
            in_channels=4,
            ratio=0.5,
        )] * 5,
    ),
}
loss_pipelines = {
    'loss_feat': dict(
        type='SharedParallelPipeline',
        args=(
            'neck',
            'teacher_neck',
        ),
        kwargs=dict(
            attn_spatial='teacher_attn_spatial',
            attn_channel='teacher_attn_channel',
            mask='masks',
        ),
        callable_=dict(
            type='FGDLoss',
            weight=5e-4,
            reduction='sum',
        ),
    ),
    'loss_attn_spatial': dict(
        type='SharedParallelPipeline',
        args=('attn_spatial', 'teacher_attn_spatial'),
        callable_=dict(
            type='L1Loss',
            weight=2.5e-4,
            reduction='sum',
        ),
    ),
    'loss_attn_channel': dict(
        type='SharedParallelPipeline',
        args=('attn_channel', 'teacher_attn_channel'),
        callable_=dict(
            type='L1Loss',
            weight=2.5e-4,
            reduction='sum',
        ),
    ),
    'loss_global': dict(
        type='SharedParallelPipeline',
        args=('global_', 'teacher_global'),
        callable_=dict(
            type='MSELoss',
            weight=2.5e-6,
            reduction='sum',
        ),
    ),
}
