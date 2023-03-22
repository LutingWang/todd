from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if __name__ == '__main__':
    # gt = COCO(
    #     '/Users/lutingwang/Developer/datasets/coco/annotations/instances_val2017.json.COCO_48_17.filtered'
    # )
    # pred = gt.loadRes('/Users/lutingwang/Downloads/debug_all_ext2.bbox.json')
    # coco_eval = COCOeval(gt, pred, iouType='bbox')
    # coco_eval.evaluate()
    import pickle as pkl
    with open('gt.pkl', 'rb') as f:
        coco_eval = pkl.load(f)
    coco_eval.accumulate()
    coco_eval.summarize()
