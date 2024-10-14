import json
from typing import Any, cast

from todd.datasets.coco import URL
from todd.datasets.lvis import LVISDataset

annotations_file = LVISDataset.ANNOTATIONS_ROOT / 'lvis_v1_val.json'
with annotations_file.open() as f:
    lvis: dict[str, Any] = json.load(f)

images = [
    image for image in lvis['images']
    if cast(str, image['coco_url']).startswith(f'{URL}val2017/')
]
image_ids = {image['id'] for image in images}
annotations = [
    annotation for annotation in lvis['annotations']
    if annotation['image_id'] in image_ids
]
lvis.update(images=images, annotations=annotations)

annotations_file = LVISDataset.ANNOTATIONS_ROOT / 'lvis_v1_minival.json'
with annotations_file.open('w') as f:
    json.dump(lvis, f, separators=(',', ':'))
