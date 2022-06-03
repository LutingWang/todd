# import logging
# import math
# from typing import List
# import torch
# import torch.nn.functional as F
# from torch import nn

# from checkpointer import TeacherDetectionCheckpointer
# from GID.get_instance import find_top_rpn_proposals
# from .base import BaseLoss
# from .builder import LOSSES

# @LOSSES.register_module()
# class GIDLoss(BaseLoss):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # fmt: off
#         self.num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
#         self.in_features = cfg.MODEL.RETINANET.IN_FEATURES
#         # Loss parameters:
#         self.focal_loss_alpha = cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA
#         self.focal_loss_gamma = cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA
#         self.smooth_l1_loss_beta = cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA
#         self.reg_weight = cfg.MODEL.RETINANET.REG_WEIGHT
#         # Inference parameters:
#         self.score_threshold = cfg.MODEL.RETINANET.SCORE_THRESH_TEST
#         self.topk_candidates = cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST
#         self.nms_threshold = cfg.MODEL.RETINANET.NMS_THRESH_TEST
#         self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
#         # fmt: on

#         # Student Network
#         self.backbone = cfg.build_backbone(
#             cfg, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))

#         backbone_shape = self.backbone.output_shape()
#         # ROI Pooling
#         pooler_resolution = (10, 10)
#         pooler_scales = tuple(1.0 / backbone_shape[k].stride for k in self.in_features)
#         sampling_ratio = 2
#         pooler_type = "ROIAlignV2"

#         self.box_pooler = ROIPooler(
#             output_size=pooler_resolution,
#             scales=pooler_scales,
#             sampling_ratio=sampling_ratio,
#             pooler_type=pooler_type,
#         )

#         self.matcher_for_semi_neg = Matcher(
#             [0.3, 0.7],
#             [0, -1, 1],
#             allow_low_quality_matches=False,
#         )

#         feature_shapes = [backbone_shape[f] for f in self.in_features]
#         self.head = RetinaNetHead(cfg, feature_shapes)
#         self.anchor_generator = cfg.build_anchor_generator(cfg, feature_shapes)

#         # Teacher Network
#         self.teacher_backbone = cfg.build_backbone(
#             cfg.TEACHER, input_shape=ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)))
#         self.teacher_in_features = cfg.TEACHER.MODEL.RETINANET.IN_FEATURES
#         teacher_backbone_shape = self.teacher_backbone.output_shape()
#         teacher_feature_shapes = [teacher_backbone_shape[f] for f in self.teacher_in_features]
#         self.teacher_head = RetinaNetHead(cfg.TEACHER, teacher_feature_shapes)
#         self.anchor_generator_teacher = cfg.build_anchor_generator(cfg.TEACHER, teacher_feature_shapes)
#         self.distill_weight = cfg.TEACHER.MODEL.DISTILL_WEIGHT
#         self.distill_adapt_layer = None
#         if cfg.TEACHER.MODEL.DISTILL_ADAPT_LAYER:
#             self.distill_adapt_layer = nn.Sequential(
#                 nn.Conv2d(
#                     self.backbone.output_shape()["p3"].channels,
#                     self.teacher_backbone.output_shape()["p3"].channels,
#                     kernel_size=3,
#                     stride=1,
#                     padding=1
#                 ),
#                 # nn.ReLU(),
#             )

#         # Matching and loss
#         self.box2box_transform = Box2BoxTransform(
#             weights=cfg.MODEL.RETINANET.BBOX_REG_WEIGHTS)
#         self.matcher = Matcher(
#             cfg.MODEL.RETINANET.IOU_THRESHOLDS,
#             cfg.MODEL.RETINANET.IOU_LABELS,
#             allow_low_quality_matches=True,
#         )

#         pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(
#             3, 1, 1)
#         pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(
#             3, 1, 1)
#         self.normalizer = lambda x: (x - pixel_mean) / pixel_std
#         self.to(self.device)

#     def forward(self, batched_inputs):
#         """
#         Args:
#             batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
#                 Each item in the list contains the inputs for one image.
#                 For now, each item in the list is a dict that contains:

#                 * image: Tensor, image in (C, H, W) format.
#                 * instances: Instances

#                 Other information that's included in the original dicts, such as:

#                 * "height", "width" (int): the output resolution of the model, used in inference.
#                     See :meth:`postprocess` for details.
#         Returns:
#             dict[str: Tensor]:
#                 mapping from a named loss to a tensor storing the loss. Used during training only.
#         """
#         images = self.preprocess_image(batched_inputs)
#         if "instances" in batched_inputs[0]:
#             gt_instances = [
#                 x["instances"].to(self.device) for x in batched_inputs
#             ]
#         elif "targets" in batched_inputs[0]:
#             log_first_n(
#                 logging.WARN,
#                 "'targets' in the model inputs is now renamed to 'instances'!",
#                 n=10)
#             gt_instances = [
#                 x["targets"].to(self.device) for x in batched_inputs
#             ]
#         else:
#             gt_instances = None

#         features = self.backbone(images.tensor)
#         features = [features[f] for f in self.in_features]
#         box_cls, box_delta = self.head(features)
#         anchors = self.anchor_generator(features)

#         if self.training:
#             self.teacher_backbone.eval()
#             self.teacher_head.eval()
#             with torch.no_grad():
#                 teacher_features = self.teacher_backbone(images.tensor)
#                 teacher_features = [teacher_features[f] for f in self.teacher_in_features]
#                 teacher_box_cls, teacher_box_delta = self.teacher_head(teacher_features)
#                 te_anchors = self.anchor_generator_teacher(teacher_features)
#             gt_classes, gt_anchors_reg_deltas = self.get_ground_truth(
#                 anchors, gt_instances)
#             return self.losses(gt_classes, gt_anchors_reg_deltas, box_cls,
#                                box_delta, features, teacher_features, anchors, images, teacher_box_cls,
#                                teacher_box_delta, gt_instances, te_anchors)
#         else:
#             results = self.inference(box_cls, box_delta, anchors, images)
#             processed_results = []
#             for results_per_image, input_per_image, image_size in zip(
#                     results, batched_inputs, images.image_sizes):
#                 height = input_per_image.get("height", image_size[0])
#                 width = input_per_image.get("width", image_size[1])
#                 r = detector_postprocess(results_per_image, height, width)
#                 processed_results.append({"instances": r})
#             return processed_results

#     def predict_proposals(self, anchor, pred_anchor_deltas):
#         """
#         Transform anchors into proposals by applying the predicted anchor deltas.

#         Returns:
#             proposals (list[Tensor]): A list of L tensors. Tensor i has shape
#                 (N, Hi*Wi*A, B), where B is box dimension (4 or 5).
#         """
#         proposals = []
#         # Transpose anchors from images-by-feature-maps (N, L) to feature-maps-by-images (L, N)
#         anchors = list(zip(*anchor))
#         # For each feature map
#         for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
#             B = anchors_i[0].tensor.size(1)
#             N, _, Hi, Wi = pred_anchor_deltas_i.shape
#             # Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N*Hi*Wi*A, B)
#             pred_anchor_deltas_i = (
#                 pred_anchor_deltas_i.view(N, -1, B, Hi, Wi).permute(0, 3, 4, 1, 2).reshape(-1, B)
#             )
#             # Concatenate all anchors to shape (N*Hi*Wi*A, B)
#             # type(anchors_i[0]) is Boxes (B = 4) or RotatedBoxes (B = 5)
#             anchors_i = type(anchors_i[0]).cat(anchors_i)
#             proposals_i = self.box2box_transform.apply_deltas(
#                 pred_anchor_deltas_i, anchors_i.tensor
#             )
#             # Append feature map proposals with shape (N, Hi*Wi*A, B)
#             proposals.append(proposals_i.view(N, -1, B))
#         return proposals

#     def predict_objectness_logits(self, pred_instance_logits):
#         """
#         Return objectness logits in the same format as the proposals returned by
#         :meth:`predict_proposals`.

#         Returns:
#             pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape
#                 (N, Hi*Wi*A).
#         """
#         N, C, Hi, Wi = pred_instance_logits[0].shape
#         pred_objectness_logits = [
#             # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
#             score.permute(0, 2, 3, 1).reshape(N, -1)
#             for score in pred_instance_logits
#         ]
#         return pred_objectness_logits

#     def losses(self, gt_classes, gt_anchors_deltas, pred_class_logits,
#                pred_anchor_deltas, pred_features, teacher_features, stu_anchors, images, teacher_class_logits,
#                teacher_box_delta, gt_instances, te_anchors):
#         """
#         Args:
#             feat: n x c x h x w
#             cls: n x (a x k) x h x w
#             reg: n x (a x k x 4) x h x w
#             teacher_feat: n x c x h x w
#             teacher_cls: n x (a x k) x h x w
#             teacher_reg: n x (a x k x 4) x h x w
#             gt_cls: n x (a x k) x h x w
#             gt_reg: n x (a x k x 4) x h x w
#         """

#         # get instance proposals
#         pred_instance_logits = []
#         for pred_class_logits_i in pred_class_logits:
#             N, C, Hi, Wi = pred_class_logits_i.shape
#             # Reshape: (N, A*K, Hi, Wi)-> (N, A, k, Hi, Wi) -> max(dim=2) -> (N, A, Hi, Wi)
#             pred_instance_logits_i, _ = pred_class_logits_i.view(N, -1, self.num_classes, Hi, Wi).max(dim=2)
#             pred_instance_logits.append(pred_instance_logits_i)

#         # get teacher logits
#         teacher_instance_logits = []
#         for pred_class_logits_i in teacher_class_logits:
#             N, C, Hi, Wi = pred_class_logits_i.shape
#             # Reshape: (N, A*K, Hi, Wi)-> (N, A, k, Hi, Wi) -> max(dim=2) -> (N, A, Hi, Wi)
#             pred_instance_logits_i, _ = pred_class_logits_i.view(N, -1, self.num_classes, Hi, Wi).max(dim=2)
#             teacher_instance_logits.append(pred_instance_logits_i)

#         st_proposal_list = self.predict_proposals(stu_anchors, pred_anchor_deltas)
#         te_proposal_list = self.predict_proposals(te_anchors, teacher_box_delta)
#         st_objectness_list = self.predict_objectness_logits(pred_instance_logits)
#         te_objectness_list = self.predict_objectness_logits(teacher_instance_logits)
#         cat_st_objectness = cat(st_objectness_list, dim=1)
#         cat_te_objectness = cat(te_objectness_list, dim=1)
#         cat_st_proposal = cat(st_proposal_list, dim=1)
#         cat_te_proposal = cat(te_proposal_list, dim=1)
#         proposal_score = torch.sigmoid(cat_te_objectness) - torch.sigmoid(cat_st_objectness)

#         cat_st_proposal[proposal_score > 0, :] = cat_te_proposal[proposal_score > 0, :]
#         proposal_score = torch.abs(proposal_score)

#         proposals = find_top_rpn_proposals(
#             [cat_st_proposal],
#             [proposal_score],
#             [gt_classes],
#             images,
#             nms_thresh=0.3,
#             pre_nms_topk=3000,
#             post_nms_topk=10,
#             min_box_side_len=10,
#             training=True,
#         )

#         # backbone_shape = self.backbone.output_shape()
#         # feature_shapes = [backbone_shape[f] for f in self.in_features]

#         st_box_features, te_box_features, distill_loss = self.mimic_loss(pred_features, teacher_features, proposals)

#         def smooth_l1_loss_IRKD(pred, gt, sigma, loss_mode='D'):
#             sigma2 = sigma ** 2
#             cond_point = 1 / sigma2
#             x = pred - gt
#             abs_x = x.abs()
#             in_mask = abs_x < cond_point
#             out_mask = ~in_mask
#             in_value = 0.5 * (sigma * x) ** 2
#             out_value = abs_x - 0.5 / sigma2
#             value = in_value * in_mask + out_value * out_mask

#             # 为 Rkd 服务，只统计非对角线个数
#             if loss_mode == 'D':
#                 value = value.sum() / (value.shape[0] * value.shape[0] - value.shape[0])

#             if loss_mode == 'A':
#                 value = value.sum() / (torch.pow(value.shape[0], 3))
#             return value

#         def pdist(e):
#             # 初始维度：(N, C)
#             Batch = e.shape[0]
#             # 1. 平方和
#             e_square = e.pow(2.0).sum(dim=1)
#             # 2. 交叉部分
#             prod = torch.matmul(e, e.permute(1, 0))
#             # 3. 求结果，保证无0
#             res = torch.sqrt(torch.max(e_square.reshape(-1, 1) + e_square.reshape(1, -1) - 2 * prod,
#                                        other=torch.tensor([1e-12]).to(e_square.device)))
#             # 对角线置0,保证下一步操作
#             mask_res = torch.ones_like(res).reshape(-1)
#             mask_res[::(Batch + 1)] = 0
#             mask_res = mask_res.reshape(Batch, Batch)
#             res = res * mask_res
#             # res = res.reshape(-1)
#             # res[::(Batch + 1)] = torch.zeros(Batch)
#             # res = res.reshape(Batch, Batch)
#             # 只计算非对角线均值
#             mask = res > 0
#             res = res * mask
#             norm_c = res.sum() / torch.max(mask.sum(), other=torch.tensor([1]).to(e_square.device))

#             res_norm = res / torch.max(norm_c, other=torch.tensor([1e-12]).to(e_square.device))
#             return res_norm

#         # RKdDistance
#         def RkdDistance(feature_te, feature_st, RKD_weight):
#             # N x C x H x W
#             # assert feature_st.partial_shape[0] == feature_te.partial_shape[0], 'invalid batch in RKdDistance'

#             # 1. 维度变为 (N, CHW)
#             feature_te = feature_te.reshape(feature_te.shape[0], -1)
#             feature_st = feature_st.reshape(feature_st.shape[0], -1)

#             # 2. 求各自特征，teacher梯度置0
#             # t_d = O.ZeroGrad(pdist(feature_te)) * RKD_weight
#             t_d = pdist(feature_te) * RKD_weight
#             s_d = pdist(feature_st) * RKD_weight

#             # 3. 求两者关系
#             loss = smooth_l1_loss_IRKD(t_d, s_d, sigma=1, loss_mode='D')  # Huber Loss
#             # loss = ((t_d - s_d) ** 2).sum()
#             return loss

#         ## RkdDistance for stem_fpn
#         def cal_stem_fpn_RkdD_loss(t_fea, s_fea, RKD_weight):
#             # loss = []
#             # for t, s in zip(t_fea, s_fea):
#             #     loss.append(RkdDistance(t, s, RKD_weight))
#             return RkdDistance(t_fea, s_fea, RKD_weight)

#         if st_box_features.shape[0] >= 3:
#             distill_loss_IRKD = cal_stem_fpn_RkdD_loss(st_box_features, te_box_features, 1.0) * 40
#         else:
#             distill_loss_IRKD = torch.tensor([0.]).to(self.device)

#         pred_class_logits, pred_anchor_deltas = \
#             permute_all_cls_and_box_to_N_HWA_K_and_concat(
#                 pred_class_logits, pred_anchor_deltas, self.num_classes
#             )  # Shapes: (N x R, K) and (N x R, 4), respectively.

#         te_pred_class_logits, te_pred_anchor_deltas = \
#             permute_all_cls_and_box_to_N_HWA_K_and_concat(
#                 teacher_class_logits, teacher_box_delta, self.num_classes
#             )  # Shapes: (N x R, K) and (N x R, 4), respectively.

#         gt_classes = gt_classes.flatten()
#         gt_anchors_deltas = gt_anchors_deltas.view(-1, 4)

#         foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)

#         gt_classes_target = torch.zeros_like(pred_class_logits)
#         gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

#         # IRKD loss

#         GI_distill_mask = self.get_GI_instance_mask(stu_anchors, proposals)
#         GI_distill_mask = GI_distill_mask.flatten()
#         # regression distillation for negative instance loss
#         reg_neg_distill_mask = (GI_distill_mask > 0)
#         reg_neg_distill_num = reg_neg_distill_mask.sum()
#         loss_box_reg_neg_KD = smooth_l1_loss(
#             pred_anchor_deltas[reg_neg_distill_mask],
#             te_pred_anchor_deltas[reg_neg_distill_mask],
#             beta=self.smooth_l1_loss_beta,
#             reduction="sum",
#         ) / max(1, reg_neg_distill_num) * 1.0

#         loss_box_reg_KD = loss_box_reg_neg_KD

#         teacher_class_logits = torch.cat([
#             permute_to_N_HWA_K(x, self.num_classes)
#             for x in teacher_class_logits
#         ], dim=1).view(-1, self.num_classes)

#         # cls distillation for neg instance
#         cls_neg_distill_mask = (GI_distill_mask > 0)
#         cls_neg_distill_num = cls_neg_distill_mask.sum()
#         loss_neg_distill = F.binary_cross_entropy_with_logits(
#             pred_class_logits[cls_neg_distill_mask],
#             (teacher_class_logits[cls_neg_distill_mask]).sigmoid(),
#             reduction="sum",
#         ) / max(1, cls_neg_distill_num) * 0.1

#         loss_distill = loss_neg_distill

#         return {"loss_distill_Instane_hint": distill_loss,
#                 "loss_distill_KD": loss_distill, "loss_box_reg_KD": loss_box_reg_KD,
#                 "distill_loss_IRKD": distill_loss_IRKD}

#     def mimic_loss(self, pred_features, teacher_features, proposals):
#         pred_features = [self.distill_adapt_layer(x) for x in pred_features]
#         st_box_features = self.box_pooler(pred_features, [x.proposal_boxes for x in proposals])
#         te_box_features = self.box_pooler(teacher_features, [x.proposal_boxes for x in proposals])
#         distill_loss = F.mse_loss(
#             st_box_features,
#             te_box_features,
#             reduction="sum",
#         ) / max(1, st_box_features.shape[0] * st_box_features.shape[2] * st_box_features.shape[3]) * self.distill_weight

#         return st_box_features,te_box_features,distill_loss

#     def get_GI_instance_mask(self, anchors, proposals):
#         neg_distill_mask = []
#         anchors = [Boxes.cat(anchors_i) for anchors_i in anchors]
#         for anchors_per_image, proposals_per_image in zip(anchors, proposals):
#             neg_instance_boxes = proposals_per_image.proposal_boxes
#             match_quality_matrix = pairwise_iou(neg_instance_boxes, anchors_per_image)
#             gt_matched_idxs, anchor_labels = self.matcher(match_quality_matrix)
#             anchor_labels[anchor_labels == -1] = 0
#             neg_distill_mask.append(anchor_labels)
#         return torch.stack(neg_distill_mask)

#     @torch.no_grad()
#     def get_ground_truth(self, anchors, targets):
#         """
#         Args:
#             anchors (list[list[Boxes]]): a list of N=#image elements. Each is a
#                 list of #feature level Boxes. The Boxes contains anchors of
#                 this image on the specific feature level.
#             targets (list[Instances]): a list of N `Instances`s. The i-th
#                 `Instances` contains the ground-truth per-instance annotations
#                 for the i-th input image.  Specify `targets` during training only.

#         Returns:
#             gt_classes (Tensor):
#                 An integer tensor of shape (N, R) storing ground-truth
#                 labels for each anchor.
#                 R is the total number of anchors, i.e. the sum of Hi x Wi x A for all levels.
#                 Anchors with an IoU with some target higher than the foreground threshold
#                 are assigned their corresponding label in the [0, K-1] range.
#                 Anchors whose IoU are below the background threshold are assigned
#                 the label "K". Anchors whose IoU are between the foreground and background
#                 thresholds are assigned a label "-1", i.e. ignore.
#             gt_anchors_deltas (Tensor):
#                 Shape (N, R, 4).
#                 The last dimension represents ground-truth box2box transform
#                 targets (dx, dy, dw, dh) that map each anchor to its matched ground-truth box.
#                 The values in the tensor are meaningful only when the corresponding
#                 anchor is labeled as foreground.
#         """
#         gt_classes = []
#         gt_anchors_deltas = []
#         anchors = [Boxes.cat(anchors_i) for anchors_i in anchors]
#         # list[Tensor(R, 4)], one for each image

#         for anchors_per_image, targets_per_image in zip(anchors, targets):
#             match_quality_matrix = pairwise_iou(targets_per_image.gt_boxes,
#                                                 anchors_per_image)
#             gt_matched_idxs, anchor_labels = self.matcher(match_quality_matrix)

#             has_gt = len(targets_per_image) > 0
#             if has_gt:
#                 # ground truth box regression
#                 matched_gt_boxes = targets_per_image.gt_boxes[gt_matched_idxs]
#                 gt_anchors_reg_deltas_i = self.box2box_transform.get_deltas(
#                     anchors_per_image.tensor, matched_gt_boxes.tensor
#                 )

#                 gt_classes_i = targets_per_image.gt_classes[gt_matched_idxs]
#                 # Anchors with label 0 are treated as background.
#                 gt_classes_i[anchor_labels == 0] = self.num_classes
#                 # Anchors with label -1 are ignored.
#                 gt_classes_i[anchor_labels == -1] = -1
#             else:
#                 gt_classes_i = torch.zeros_like(
#                     gt_matched_idxs) + self.num_classes
#                 gt_anchors_reg_deltas_i = torch.zeros_like(anchors_per_image.tensor)

#             gt_classes.append(gt_classes_i)
#             gt_anchors_deltas.append(gt_anchors_reg_deltas_i)

#         return torch.stack(gt_classes), torch.stack(gt_anchors_deltas)

#     def inference(self, box_cls, box_delta, anchors, images):
#         """
#         Arguments:
#             box_cls, box_delta: Same as the output of :meth:`RetinaNetHead.forward`
#             anchors (list[list[Boxes]]): a list of #images elements. Each is a
#                 list of #feature level Boxes. The Boxes contain anchors of this
#                 image on the specific feature level.
#             images (ImageList): the input images

#         Returns:
#             results (List[Instances]): a list of #images elements.
#         """
#         assert len(anchors) == len(images)
#         results = []

#         box_cls = [permute_to_N_HWA_K(x, self.num_classes) for x in box_cls]
#         box_delta = [permute_to_N_HWA_K(x, 4) for x in box_delta]
#         # list[Tensor], one per level, each has shape (N, Hi x Wi x A, K or 4)

#         for img_idx, anchors_per_image in enumerate(anchors):
#             image_size = images.image_sizes[img_idx]
#             box_cls_per_image = [
#                 box_cls_per_level[img_idx] for box_cls_per_level in box_cls
#             ]
#             box_reg_per_image = [
#                 box_reg_per_level[img_idx] for box_reg_per_level in box_delta
#             ]
#             results_per_image = self.inference_single_image(
#                 box_cls_per_image, box_reg_per_image, anchors_per_image,
#                 tuple(image_size))
#             results.append(results_per_image)
#         return results

#     def inference_single_image(self, box_cls, box_delta, anchors, image_size):
#         """
#         Single-image inference. Return bounding-box detection results by thresholding
#         on scores and applying non-maximum suppression (NMS).

#         Arguments:
#             box_cls (list[Tensor]): list of #feature levels. Each entry contains
#                 tensor of size (H x W x A, K)
#             box_delta (list[Tensor]): Same shape as 'box_cls' except that K becomes 4.
#             anchors (list[Boxes]): list of #feature levels. Each entry contains
#                 a Boxes object, which contains all the anchors for that
#                 image in that feature level.
#             image_size (tuple(H, W)): a tuple of the image height and width.

#         Returns:
#             Same as `inference`, but for only one image.
#         """
#         boxes_all = []
#         scores_all = []
#         class_idxs_all = []

#         # Iterate over every feature level
#         for box_cls_i, box_reg_i, anchors_i in zip(box_cls, box_delta, anchors):
#             # (HxWxAxK,)
#             box_cls_i = box_cls_i.flatten().sigmoid_()

#             # Keep top k top scoring indices only.
#             num_topk = min(self.topk_candidates, box_reg_i.size(0))
#             # torch.sort is actually faster than .topk (at least on GPUs)
#             predicted_prob, topk_idxs = box_cls_i.sort(descending=True)
#             predicted_prob = predicted_prob[:num_topk]
#             topk_idxs = topk_idxs[:num_topk]

#             # filter out the proposals with low confidence score
#             keep_idxs = predicted_prob > self.score_threshold
#             predicted_prob = predicted_prob[keep_idxs]
#             topk_idxs = topk_idxs[keep_idxs]

#             anchor_idxs = topk_idxs // self.num_classes
#             classes_idxs = topk_idxs % self.num_classes

#             box_reg_i = box_reg_i[anchor_idxs]
#             anchors_i = anchors_i[anchor_idxs]
#             # predict boxes
#             predicted_boxes = self.box2box_transform.apply_deltas(
#                 box_reg_i, anchors_i.tensor)

#             boxes_all.append(predicted_boxes)
#             scores_all.append(predicted_prob)
#             class_idxs_all.append(classes_idxs)

#         boxes_all, scores_all, class_idxs_all = [
#             cat(x) for x in [boxes_all, scores_all, class_idxs_all]
#         ]
#         keep = batched_nms(boxes_all, scores_all, class_idxs_all,
#                            self.nms_threshold)
#         keep = keep[:self.max_detections_per_image]

#         result = Instances(image_size)
#         result.pred_boxes = Boxes(boxes_all[keep])
#         result.scores = scores_all[keep]
#         result.pred_classes = class_idxs_all[keep]
#         return result

#     def preprocess_image(self, batched_inputs):
#         """
#         Normalize, pad and batch the input images.
#         """
#         images = [x["image"].to(self.device) for x in batched_inputs]
#         images = [self.normalizer(x) for x in images]
#         images = ImageList.from_tensors(images,
#                                         self.backbone.size_divisibility)
#         return images

# class RetinaNetHead(nn.Module):
#     """
#     The head used in RetinaNet for object classification and box regression.
#     It has two subnets for the two tasks, with a common structure but separate parameters.
#     """

#     def __init__(self, cfg, input_shape: List[ShapeSpec]):
#         super().__init__()
#         # fmt: off
#         in_channels = input_shape[0].channels
#         num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
#         num_convs = cfg.MODEL.RETINANET.NUM_CONVS
#         prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
#         num_anchors = cfg.build_anchor_generator(cfg, input_shape).num_cell_anchors
#         # fmt: on
#         assert (
#                 len(set(num_anchors)) == 1
#         ), "Using different number of anchors between levels is not currently supported!"
#         num_anchors = num_anchors[0]

#         cls_subnet = []
#         bbox_subnet = []
#         for _ in range(num_convs):
#             cls_subnet.append(
#                 nn.Conv2d(in_channels,
#                           in_channels,
#                           kernel_size=3,
#                           stride=1,
#                           padding=1))
#             cls_subnet.append(nn.ReLU())
#             bbox_subnet.append(
#                 nn.Conv2d(in_channels,
#                           in_channels,
#                           kernel_size=3,
#                           stride=1,
#                           padding=1))
#             bbox_subnet.append(nn.ReLU())

#         self.cls_subnet = nn.Sequential(*cls_subnet)
#         self.bbox_subnet = nn.Sequential(*bbox_subnet)
#         self.cls_score = nn.Conv2d(in_channels,
#                                    num_anchors * num_classes,
#                                    kernel_size=3,
#                                    stride=1,
#                                    padding=1)
#         self.bbox_pred = nn.Conv2d(in_channels,
#                                    num_anchors * 4,
#                                    kernel_size=3,
#                                    stride=1,
#                                    padding=1)

#         # Initialization
#         for modules in [
#             self.cls_subnet, self.bbox_subnet, self.cls_score,
#             self.bbox_pred
#         ]:
#             for layer in modules.modules():
#                 if isinstance(layer, nn.Conv2d):
#                     torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
#                     torch.nn.init.constant_(layer.bias, 0)

#         # Use prior in model initialization to improve stability
#         bias_value = -math.log((1 - prior_prob) / prior_prob)
#         torch.nn.init.constant_(self.cls_score.bias, bias_value)

#     def forward(self, features):
#         """
#         Arguments:
#             features (list[Tensor]): FPN feature map tensors in high to low resolution.
#                 Each tensor in the list correspond to different feature levels.

#         Returns:
#             logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
#                 The tensor predicts the classification probability
#                 at each spatial position for each of the A anchors and K object
#                 classes.
#             bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
#                 The tensor predicts 4-vector (dx,dy,dw,dh) box
#                 regression values for every anchor. These values are the
#                 relative offset between the anchor and the ground truth box.
#         """
#         logits = []
#         bbox_reg = []
#         for feature in features:
#             logits.append(self.cls_score(self.cls_subnet(feature)))
#             bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
#         return logits, bbox_reg
