import time
from collections import UserList, defaultdict
from typing import Iterable, NamedTuple

import einops
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO

import todd

GTBBox = tuple[float, float, float, float, bool]
PredBBox = tuple[float, float, float, float, float]
Product = tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]


class Performance(NamedTuple):
    precisions: torch.Tensor
    recalls: torch.Tensor


class Performances(UserList[Performance]):

    @property
    def precisions(self) -> torch.Tensor:
        return torch.stack([p.precisions for p in self], -1)

    @property
    def recalls(self) -> torch.Tensor:
        return torch.stack([p.recalls for p in self], -1)


class GTMixin(todd.BBoxes):

    def __init__(self, bboxes: list[GTBBox], *args, **kwargs) -> torch.Tensor:
        self._ignored = torch.tensor(
            [bbox[-1] for bbox in bboxes],
            dtype=torch.bool,
        )
        bboxes = torch.tensor([bbox[:-1] for bbox in bboxes])
        super().__init__(bboxes, *args, **kwargs)

    @property
    def ignored(self) -> torch.Tensor:
        return self._ignored

    @property
    def num_unignored(self) -> int:
        return (~self._ignored).sum()

    @property
    def has_ignored(self) -> int:
        return self._ignored.any()

    @property
    def has_unignored(self) -> int:
        return not self._ignored.all()


class CrowdMixin(todd.BBoxes):

    def unions(self, other: todd.BBoxes, *args, **kwargs) -> torch.Tensor:
        return einops.repeat(other.area, 'n -> m n', m=len(self)).contiguous()


class PredMixin(todd.BBoxes):

    def __init__(self, bboxes: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        super().__init__(bboxes[:, :-1], *args, **kwargs)
        self._scores = bboxes[:, -1]

    @property
    def scores(self) -> torch.Tensor:
        return self._scores


class GTBBoxesXYWH(GTMixin, todd.BBoxesXYWH):
    pass


class CrowdBBoxesXYWH(CrowdMixin, todd.BBoxesXYWH):
    pass


class PredBBoxesXYWH(PredMixin, todd.BBoxesXYWH):
    pass


class Producer:

    def __init__(self, iou_thresholds: torch.Tensor) -> None:
        self._iou_thresholds = iou_thresholds

    def produce(
        self,
        pred_list: list[PredBBox],
        gt_list: list[GTBBox],
    ) -> Product | None:
        if len(pred_list) == 0 and len(gt_list) == 0:
            return None

        if len(gt_list) == 0:
            tps = torch.zeros(
                self._iou_thresholds.numel(),
                len(pred_list),
                dtype=torch.bool,
            )
            fps = torch.ones(
                self._iou_thresholds.numel(),
                len(pred_list),
                dtype=torch.bool,
            )
            scores = torch.tensor([pred[-1] for pred in pred_list])
            return tps, fps, scores, 0
        gts = GTBBoxesXYWH(gt_list)

        if len(pred_list) == 0:
            tps = fps = torch.empty(
                self._iou_thresholds.numel(),
                0,
                dtype=torch.bool,
            )
            scores = torch.empty(0)
            return tps, fps, scores, gts.num_unignored

        gt_indices = gts.indices()
        ignored_gt_indices, = torch.where(gt_indices & gts.ignored)
        unignored_gt_indices, = torch.where(gt_indices & ~gts.ignored)

        preds = PredBBoxesXYWH(torch.tensor(pred_list))
        unignored_ious = todd.BBoxesXYWH(gts.to_tensor()).ious(preds)
        ignored_ious = CrowdBBoxesXYWH(gts.to_tensor()).ious(preds)

        tp_list = [[False] * len(preds)
                   for _ in range(self._iou_thresholds.numel())]
        if gts.has_unignored:
            ious = unignored_ious[unignored_gt_indices].tolist()
            pred_indices = preds.scores.argsort(descending=True).tolist()
            for threshold_index, threshold in enumerate(
                self._iou_thresholds.tolist()
            ):
                unmatched_gt_indices = unignored_gt_indices.tolist()
                for pred_index in pred_indices:
                    if len(unmatched_gt_indices) == 0:
                        break
                    unmatched_ious = [
                        ious[gt_index][pred_index]
                        for gt_index in unmatched_gt_indices
                    ]
                    if (iou := max(unmatched_ious)) >= threshold:
                        tp_list[threshold_index][pred_index] = True
                        unmatched_gt_indices.pop(unmatched_ious.index(iou))
        tps = torch.tensor(tp_list, dtype=torch.bool)

        fps = ~tps
        if gts.has_ignored:
            ious = ignored_ious[ignored_gt_indices].max(0).values
            fps &= ious < einops.rearrange(self._iou_thresholds, 't -> t 1')

        return tps, fps, preds.scores, unignored_gt_indices.numel()


class Consumer:

    def __init__(
        self, recall_thresholds: torch.Tensor, eps: float = 1e-6
    ) -> None:
        self._recall_thresholds = recall_thresholds
        self._eps = eps

    def consume(self, product: Product) -> Performance:
        """Compute the precision and recall scores.

        Args:
            product: the product.

        Returns:
            The performance.
        """
        tps, fps, scores, gts = product

        indices = torch.argsort(scores, descending=True)
        tps = torch.cumsum(tps[:, indices], -1)
        fps = torch.cumsum(fps[:, indices], -1)

        precisions = tps / (tps + fps + self._eps)
        precisions = precisions.flip(-1)
        precisions = torch.cummax(precisions, -1).values
        precisions = precisions.flip(-1)
        precisions = F.pad(precisions, (0, 1), value=0)

        recalls = tps / gts
        indices = torch.searchsorted(recalls, self._recall_thresholds)

        return Performance(precisions.gather(1, indices), recalls[:, -1])


class Store:

    def __init__(self) -> None:
        self._tps = []
        self._fps = []
        self._scores = []
        self._gts = []

    def store(self, product: Product) -> None:
        tps, fps, scores, gts = product
        self._tps.append(tps)
        self._fps.append(fps)
        self._scores.append(scores)
        self._gts.append(gts)

    def pack(self) -> Product | None:
        if (
            len(self._tps) == 0 and len(self._fps) == 0
            and len(self._scores) == 0 and len(self._gts) == 0
        ):
            return None
        tps = torch.cat(self._tps, -1)
        fps = torch.cat(self._fps, -1)
        scores = torch.cat(self._scores, -1)
        gts = sum(self._gts)
        return tps, fps, scores, gts


class Evaluator:

    def __init__(
        self,
        iou_thresholds: torch.Tensor,
        recall_thresholds: torch.Tensor,
        category_ids: Iterable[int],
        image_ids: Iterable[int],
    ) -> None:
        recall_thresholds = einops.repeat(
            recall_thresholds,
            'r -> i r',
            i=iou_thresholds.numel(),
        ).contiguous()

        self._iou_thresholds = iou_thresholds
        self._recall_thresholds = recall_thresholds
        self._category_ids = list(category_ids)
        self._image_ids = list(image_ids)

        self._producer = Producer(iou_thresholds)
        self._consumer = Consumer(recall_thresholds)

    def __call__(
        self,
        dts: dict[tuple[int, int], list[PredBBox]],
        gts: dict[tuple[int, int], list[todd.BBox]],
    ) -> Performances:
        category_ids = []
        performances = Performances()
        for category_id in self._category_ids:
            store = Store()
            for image_id in self._image_ids:
                product = self._producer.produce(
                    dts[image_id, category_id],
                    gts[image_id, category_id],
                )
                if product is None:
                    continue
                store.store(product)
            product = store.pack()
            if product is None:
                continue
            performance = self._consumer.consume(product)
            category_ids.append(category_id)
            performances.append(performance)

        return performances

    def stat(self, performances: Performances) -> tuple[float, ...]:
        precisions = performances.precisions
        recalls = performances.recalls
        iou_thresholds = self._iou_thresholds.tolist()
        p = precisions.mean()
        p_50 = (
            precisions[iou_thresholds.index(0.5)].mean()
            if 0.5 in iou_thresholds else -1
        )
        p_75 = (
            precisions[iou_thresholds.index(0.75)].mean()
            if 0.75 in iou_thresholds else -1
        )
        r = recalls.mean()
        return (p, p_50, p_75, r)

    def summarize(self, *args, **kwargs) -> None:
        p, p_50, p_75, r = self.stat(*args, **kwargs)
        summary = f'\nAP@0.50:0.95 = {p:.3f}'
        summary += f'\nAP@0.50      = {p_50:.3f}'
        summary += f'\nAP@0.75      = {p_75:.3f}'
        summary += f'\nAR@0.50:0.95 = {r:.3f}'
        todd.logger.info(summary)


class COCOeval:

    def __init__(self, cocoGt, cocoDt):
        self._timeout = 10
        self.cocoGt = cocoGt  # ground truth COCO API
        self.cocoDt = cocoDt  # detections COCO API

        self._iou_thresholds = np.array([0.5, 0.95])

        iou_thresholds = torch.arange(0.5, 1, 0.05)
        # iou_thresholds = torch.tensor([0.5])

        recall_thresholds = torch.arange(0, 101, dtype=torch.float) / 100

        category_ids = sorted(cocoGt.getCatIds())
        image_ids = sorted(cocoGt.getImgIds())

        self._evaluator = Evaluator(
            iou_thresholds, recall_thresholds, category_ids, image_ids
        )

        gts = cocoGt.loadAnns(
            cocoGt.getAnnIds(imgIds=image_ids, catIds=category_ids)
        )
        dts = cocoDt.loadAnns(
            cocoDt.getAnnIds(imgIds=image_ids, catIds=category_ids)
        )

        # set ignore flag
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        # self._dt_scores = defaultdict(list)  # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'],
                      gt['category_id']].append(gt['bbox'] + [gt['iscrowd']])
        for dt in dts:
            self._dts[dt['image_id'],
                      dt['category_id']].append(dt['bbox'] + [dt['score']])

    def run(self):
        print('Accumulating evaluation results...')
        tic = time.time()
        performances = self._evaluator(self._dts, self._gts)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))
        self._evaluator.summarize(performances)


if __name__ == '__main__':
    import ipdb

    gt = COCO(
        '/Users/lutingwang/Developer/datasets/coco/annotations/instances_val2017.json.COCO_48_17.filtered'
    )
    pred = gt.loadRes('/Users/lutingwang/Downloads/debug_all_ext2.bbox.json')
    coco_eval = COCOeval(gt, pred)
    coco_eval.run()
