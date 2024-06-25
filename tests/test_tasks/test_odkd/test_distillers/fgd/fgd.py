# pylint: disable=invalid-name

loss_scope = 'ModelRegistry.LossRegistry.'
kd_scope = 'TaskRegistry.KDRegistry.'
kd_adapt_scope = kd_scope + 'KDDistillerRegistry.KDAdaptRegistry.'
kd_loss_scope = kd_scope + 'KDModelRegistry.KDLossRegistry.'
odkd_scope = 'TaskRegistry.ODKDRegistry.'
odkd_adapt_scope = odkd_scope + 'ODKDDistillerRegistry.ODKDAdaptRegistry.'
odkd_loss_scope = odkd_scope + 'ODKDModelRegistry.ODKDLossRegistry.'
distiller = dict(
    models=[],
    hook_pipelines=dict(),
    adapt_pipelines=[
        dict(
            type='SingleParallelOperator',
            args=('neck', ),
            outputs='attn_spatial',
            atom=dict(
                type=kd_adapt_scope + 'SpatialAttention',
                temperature=0.5,
            ),
        ),
        dict(
            type='SingleParallelOperator',
            args=('teacher_neck', ),
            outputs='teacher_attn_spatial',
            atom=dict(
                type=kd_adapt_scope + 'SpatialAttention',
                temperature=0.5,
            ),
        ),
        dict(
            type='SingleParallelOperator',
            args=('neck', ),
            outputs='attn_channel',
            atom=dict(
                type=kd_adapt_scope + 'ChannelAttention',
                temperature=0.5,
            ),
        ),
        dict(
            type='SingleParallelOperator',
            args=('teacher_neck', ),
            outputs='teacher_attn_channel',
            atom=dict(
                type=kd_adapt_scope + 'ChannelAttention',
                temperature=0.5,
            ),
        ),
        dict(
            type='SingleOperator',
            args=('img_shape', 'gt_bboxes'),
            outputs='masks',
            atom=dict(
                type=odkd_adapt_scope + 'FGDMask',
                neg_gain=0.5,
                strides=[8, 16, 32, 64, 128],
                ceil_mode=True,
            ),
        ),
        dict(
            type='MultipleParallelOperator',
            args=('neck', ),
            outputs='global_',
            atoms=[
                dict(
                    type=kd_adapt_scope + 'mmcv_ContextBlock',
                    in_channels=4,
                    ratio=0.5,
                )
            ] * 5,
        ),
        dict(
            type='MultipleParallelOperator',
            args=('teacher_neck', ),
            outputs='teacher_global',
            atoms=[
                dict(
                    type=kd_adapt_scope + 'mmcv_ContextBlock',
                    in_channels=4,
                    ratio=0.5,
                )
            ] * 5,
        ),
    ],
    loss_pipelines=[
        dict(
            type='SingleParallelOperator',
            args=('neck', 'teacher_neck'),
            outputs='loss_feat',
            kwargs=dict(
                attn_spatial='teacher_attn_spatial',
                attn_channel='teacher_attn_channel',
                mask='masks',
            ),
            atom=dict(
                type=odkd_loss_scope + 'FGDLoss',
                weight=5e-4,
                reduction='sum',
            ),
        ),
        dict(
            type='SingleParallelOperator',
            args=('attn_spatial', 'teacher_attn_spatial'),
            outputs='loss_attn_spatial',
            atom=dict(
                type=loss_scope + 'L1Loss',
                weight=2.5e-4,
                reduction='sum',
            ),
        ),
        dict(
            type='SingleParallelOperator',
            args=('attn_channel', 'teacher_attn_channel'),
            outputs='loss_attn_channel',
            atom=dict(
                type=loss_scope + 'L1Loss',
                weight=2.5e-4,
                reduction='sum',
            ),
        ),
        dict(
            type='SingleParallelOperator',
            args=('global_', 'teacher_global'),
            outputs='loss_global',
            atom=dict(
                type=loss_scope + 'MSELoss',
                weight=2.5e-6,
                reduction='sum',
            ),
        ),
    ],
)
