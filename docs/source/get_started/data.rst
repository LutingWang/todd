Data
====

ImageNet
--------

MS-COCO
-------

.. code-block:: bash

    root=data/coco
    mkdir -p ${root} && cd ${root}
    wget http://images.cocodataset.org/zips/train2017.zip
    wget http://images.cocodataset.org/zips/val2017.zip
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip -q *.zip

.. code::

    data/coco/
    ├── annotations
    │   ├── instances_train2017.json
    |   ├── instances_val2017.json
    ├── train2017
    │   └── *.jpg
    └── val2017
        └── *.jpg

LVIS v1.0
---------

.. code-block:: bash

    root=data/lvis
    mkdir -p ${root} && cd ${root}
    ln -s ../coco/train2017 .
    ln -s ../coco/val2017 .
    mkdir annotations && cd annotations
    wget https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip
    wget https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip
    unzip -q *.zip

.. code::

    data/lvis/
    ├── annotations
    │   ├── lvis_v1_train.json
    |   ├── lvis_v1_val.json
    ├── train2017 -> ../coco/train2017
    └── val2017 -> ../coco/val2017
