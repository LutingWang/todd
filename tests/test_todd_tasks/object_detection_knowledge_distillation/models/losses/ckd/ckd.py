# pylint: disable=invalid-name,duplicate-code

loss_scope = 'ModelRegistry.LossRegistry.'
kd_scope = 'TaskRegistry.KDRegistry.'
kd_adapt_scope = kd_scope + 'KDDistillerRegistry.KDAdaptRegistry.'
kd_loss_scope = kd_scope + 'KDModelRegistry.KDLossRegistry.'
odkd_scope = 'TaskRegistry.ODKDRegistry.'
odkd_adapt_scope = odkd_scope + 'ODKDDistillerRegistry.ODKDAdaptRegistry.'
odkd_loss_scope = odkd_scope + 'ODKDModelRegistry.ODKDLossRegistry.'
distiller = dict(
    models=[],
    hook_pipelines=[],
    adapt_pipeline={
        'pred_reshaped': dict(
            type='SingleParallelOperator',
            args=('preds', ),
            atom=dict(
                type=kd_adapt_scope + 'Model',
                model=dict(
                    type='einops_layers_torch_Rearrange',
                    pattern='bs dim h w -> bs h w dim',
                ),
            ),
        ),
        'targets, bboxes, bbox_poses, anchor_ids': dict(
            type='SingleOperator',
            args=('targets', 'bboxes', 'bbox_ids'),
            atom=dict(
                type=kd_adapt_scope + 'CustomAdapt',
                stride=1,
            ),
        ),
        'pred_indexed': dict(
            type='SingleOperator',
            args=('pred_reshaped', 'bbox_poses'),
            atom=dict(type=kd_adapt_scope + 'Index'),
        ),
        'preds': dict(
            type='SingleOperator',
            args=('pred_indexed', 'anchor_ids'),
            atom=dict(
                type=kd_adapt_scope + 'Decouple',
                num=9,
                in_features=4,
                out_features=16,
                bias=False,
            ),
        ),
    },
    loss_pipeline={
        'loss_ckd': dict(
            type='SingleOperator',
            args=('preds', 'targets', 'bboxes'),
            atom=dict(
                type=odkd_loss_scope + 'CKDLoss',
                weight=0.5,
            ),
        ),
    },
)
