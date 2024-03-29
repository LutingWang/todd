models = list()  # type: ignore[var-annotated]
hook_pipelines = list()  # type: ignore[var-annotated]
adapt_pipelines = {
    'pred_reshaped': dict(
        type='SharedParallelPipeline',
        args=('preds', ),
        callable_=dict(
            type='Rearrange',
            pattern='bs dim h w -> bs h w dim',
        ),
    ),
    'targets, bboxes, bbox_poses, anchor_ids': dict(
        type='VanillaPipeline',
        args=('targets', 'bboxes', 'bbox_ids'),
        callable_=dict(
            type='CustomAdapt',
            stride=1,
        ),
    ),
    'pred_indexed': dict(
        type='VanillaPipeline',
        args=('pred_reshaped', 'bbox_poses'),
        callable_=dict(type='Index'),
    ),
    'preds': dict(
        type='VanillaPipeline',
        args=('pred_indexed', 'anchor_ids'),
        callable_=dict(
            type='Decouple',
            num=9,
            in_features=4,
            out_features=16,
            bias=False,
        ),
    ),
}
loss_pipelines = dict(
    loss_ckd=dict(
        type='VanillaPipeline',
        args=('preds', 'targets', 'bboxes'),
        callable_=dict(
            type='CKDLoss',
            weight=0.5,
        ),
    ),
)
