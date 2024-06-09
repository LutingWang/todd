LVIS
====

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
    cd ../..

.. code::

    data/lvis/
    ├── annotations
    │   ├── lvis_v1_train.json
    |   └── lvis_v1_val.json
    ├── train2017 -> ../coco/train2017
    └── val2017 -> ../coco/val2017
