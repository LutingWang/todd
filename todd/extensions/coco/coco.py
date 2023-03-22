import copy
import time
from collections import defaultdict
from typing import Iterable, NamedTuple

import einops
import numpy as np
import pycocotools.mask as maskUtils
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO


class Record(NamedTuple):
    matched: torch.Tensor
    ignored: torch.Tensor
    scores: torch.Tensor
    num_gts: int


class Records(NamedTuple):
    matched: torch.Tensor
    ignored: torch.Tensor
    ranks: torch.Tensor
    num_gts: int


def merge_records(records: Iterable[Record]) -> Records | None:
    records = [record for record in records if record is not None]
    if len(records) == 0:
        return None

    num_gts = sum([record.num_gts for record in records])
    if num_gts == 0:
        return None

    matched = torch.cat([record.matched for record in records], axis=1)
    ignored = torch.cat([record.ignored for record in records], axis=1)
    scores = torch.cat([record.scores for record in records])
    ranks = torch.cat([torch.arange(len(e.scores)) for e in records])

    indices = torch.argsort(-scores)
    return Records(
        matched[:, indices],
        ignored[:, indices],
        ranks[indices],
        num_gts,
    )


def evaluate_records(
    records: Records | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pass


class COCOeval:

    def __init__(self, cocoGt=None, cocoDt=None):
        self.cocoGt = cocoGt  # ground truth COCO API
        self.cocoDt = cocoDt  # detections COCO API
        self.evalImgs = defaultdict(
            list
        )  # per-image per-category evaluation results [KxAxI] elements
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = Params()  # parameters
        self._paramsEval = {}  # parameters for evaluation
        self.stats = []  # result summarization
        self.ious = {}  # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

        self.recThrs = einops.repeat(
            torch.arange(101),
            'r -> i r',
            i=len(self.params.iouThrs),
        ).contiguous()
        self.recFactor = 100

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''

        p = self.params
        gts = self.cocoGt.loadAnns(
            self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
        )
        dts = self.cocoDt.loadAnns(
            self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds)
        )

        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(
            list
        )  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        print('Evaluate annotation type *bbox*')
        p.imgIds = list(np.unique(p.imgIds))
        p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds

        computeIoU = self.computeIoU
        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [[[
            evaluateImg(imgId, catId, areaRng, maxDet) for imgId in p.imgIds
        ] for areaRng in p.areaRng] for catId in catIds]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def computeIoU(self, imgId, catId):
        p = self.params
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        g = [g['bbox'] for g in gt]
        d = [d['bbox'] for d in dt]

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d, g, iscrowd)
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        gt = self._gts[imgId, catId]
        dt = self._dts[imgId, catId]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId,
                         catId][:, gtind] if len(self.ious[imgId, catId]
                                                 ) > 0 else self.ious[imgId,
                                                                      catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = torch.zeros((T, D), dtype=torch.bool)
        gtIg = torch.tensor([g['_ignore'] for g in gt], dtype=torch.bool)
        dtIg = torch.zeros((T, D), dtype=torch.bool)
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and not gtIg[m] and gtIg[gind]:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtIg[tind, dind] = gtIg[m]
                    # dtm[tind, dind] = gt[m]['id']
                    dtm[tind, dind] = True
                    gtm[tind, m] = d['id']
        # set unmatched detections outside of area range to ignore
        a = torch.tensor([
            d['area'] < aRng[0] or d['area'] > aRng[1] for d in dt
        ],
                         dtype=torch.bool).reshape((1, len(dt))).repeat(T, 1)
        # dtIg = np.logical_or(
        #     dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0))
        # )
        dtIg |= ~dtm & a
        # store results for given image and category
        return Record(
            # 'image_id': imgId,
            # 'category_id': catId,
            # 'aRng': aRng,
            # 'maxDet': maxDet,
            # 'dtIds': [d['id'] for d in dt],
            # 'gtIds': [g['id'] for g in gt],
            dtm,
            # 'gtMatches': gtm,
            dtIg,
            torch.tensor([d['score'] for d in dt]),
            gtIg.numel() - gtIg.sum(),
        )

    def accumulate(self, evalImgs):
        """Accumulate per image evaluation results and store the result in
        self.eval.

        :param p: input params for evaluation
        :return: None
        """
        print('Accumulating evaluation results...')
        tic = time.time()
        # allows input customized parameters
        p = self.params
        T = len(p.iouThrs)
        # R = len(p.recThrs)
        R = self.recThrs.shape[-1]
        K = len(p.catIds)
        A = len(p.areaRng)
        M = len(p.maxDets)
        precision = -torch.ones((T, R, K, A, M)
                                )  # -1 for the precision of absent categories
        recall = -torch.ones((T, K, A, M))

        # retrieve E at each category, area range, and max number of detections
        for k in range(len(self.params.catIds)):
            for a in range(len(self.params.areaRng)):
                E = merge_records(evalImgs[k][a])
                if E is None:
                    continue
                matched, ignored, ranks, num_gts = E
                for m, maxDet in enumerate(self.params.maxDets):
                    p, r = self._evaluate(
                        matched[:, ranks < maxDet],
                        ignored[:, ranks < maxDet],
                        num_gts,
                    )

                    precision[:, :, k, a, m] = p
                    recall[:, k, a, m] = r

        self.eval = {
            'precision': precision.numpy(),
            'recall': recall.numpy(),
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    # def _gather(
    #     self,
    #     E: list[Record],
    # ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    #     E = [e for e in E if e is not None]
    #     if len(E) == 0:
    #         return None
    #     num_gts = sum([e.num_gts for e in E])
    #     # npig = np.count_nonzero(gtIg == 0)
    #     if num_gts == 0:
    #         return None
    #     scores = torch.cat([e.scores for e in E])
    #     indices = torch.argsort(-scores)
    #     ranks = torch.cat([torch.arange(len(e.scores)) for e in E])[indices]
    #     matched = torch.cat([e.matched for e in E], axis=1)[:, indices]
    #     ignored = torch.cat([e.ignored for e in E], axis=1)[:, indices]
    #     return ranks, matched, ignored, num_gts

    def _evaluate(
        self,
        matched: torch.Tensor,
        ignored: torch.Tensor,
        num_gts: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tps = torch.cumsum(matched & ~ignored, -1)
        fps = torch.cumsum(~matched & ~ignored, -1)
        precisions = tps / (tps + fps)
        precisions.nan_to_num_(0)
        precisions = precisions.flip(-1)
        precisions = torch.cummax(precisions, -1).values
        precisions = precisions.flip(-1)
        precisions = F.pad(precisions, (0, 1), value=0)

        # use integer to avoid precision problems
        indices = torch.searchsorted(
            tps * self.recFactor,
            self.recThrs * num_gts,
        )
        precisions = precisions.gather(1, indices)

        recalls = tps[:, -1] / num_gts

        return precisions, recalls

    def summarize(self):
        """Compute and display summary metrics for evaluation results.

        Note this function can *only* be applied on the default
        parameter setting
        """

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [
                i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng
            ]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(
                iStr.format(
                    titleStr, typeStr, iouStr, areaRng, maxDets, mean_s
                )
            )
            return mean_s

        if not self.eval:
            raise Exception('Please run accumulate() first')
        stats = np.zeros((12, ))
        stats[0] = _summarize(1)
        stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
        # import ipdb
        # ipdb.set_trace()
        stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
        stats[3] = _summarize(
            1, areaRng='small', maxDets=self.params.maxDets[2]
        )
        stats[4] = _summarize(
            1, areaRng='medium', maxDets=self.params.maxDets[2]
        )
        stats[5] = _summarize(
            1, areaRng='large', maxDets=self.params.maxDets[2]
        )
        stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
        stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
        stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
        stats[9] = _summarize(
            0, areaRng='small', maxDets=self.params.maxDets[2]
        )
        stats[10] = _summarize(
            0, areaRng='medium', maxDets=self.params.maxDets[2]
        )
        stats[11] = _summarize(
            0, areaRng='large', maxDets=self.params.maxDets[2]
        )
        self.stats = stats

    def __str__(self):
        self.summarize()


class Params:

    def __init__(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(
            .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True
        )
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0**2, 1e5**2], [0**2, 32**2], [32**2, 96**2],
                        [96**2, 1e5**2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']


if __name__ == '__main__':
    gt = COCO(
        '/Users/lutingwang/Developer/datasets/coco/annotations/instances_val2017.json.COCO_48_17.filtered'
    )
    pred = gt.loadRes('/Users/lutingwang/Downloads/debug_all_ext2.bbox.json')
    coco_eval = COCOeval(gt, pred)
    coco_eval.evaluate()
    evalImgs = coco_eval.evalImgs
    # torch.save(evalImgs, 'coco.pth')
    # evalImgs = torch.load('coco.pth')
    coco_eval.accumulate(evalImgs)
    coco_eval.summarize()
    assert np.allclose(
        coco_eval.stats,
        np.array([
            0.31159549, 0.49961171, 0.3303898, 0.20755473, 0.34233785,
            0.41034368, 0.27638768, 0.4994253, 0.55246888, 0.40431019,
            0.59136161, 0.67892689
        ])
    ), coco_eval.stats
