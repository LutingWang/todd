models = list()  # type: ignore[var-annotated]
hooks = dict()  # type: ignore[var-annotated]
adapts = {
    'pred_reshaped': dict(
        inputs=('preds', ),
        parallel=True,
        action=dict(
            type='Rearrange',
            pattern='bs dim h w -> bs h w dim',
        ),
    ),
    'targets, bboxes, bbox_poses, anchor_ids': dict(
        inputs=('targets', 'bboxes', 'bbox_ids'),
        action=dict(
            type='CustomAdapt',
            stride=1,
        ),
    ),
    'pred_indexed': dict(
        inputs=('pred_reshaped', 'bbox_poses'),
        action=dict(type='Index'),
    ),
    'preds': dict(
        inputs=('pred_indexed', 'anchor_ids'),
        action=dict(
            type='Decouple',
            num=9,
            in_features=4,
            out_features=16,
            bias=False,
        ),
    ),
}
losses = dict(
    loss_ckd=dict(
        inputs=('preds', 'targets', 'bboxes'),
        action=dict(
            type='CKDLoss',
            weight=0.5,
        ),
    ),
)
