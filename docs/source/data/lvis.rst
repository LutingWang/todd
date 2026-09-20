LVIS
====

https://www.lvisdataset.org/

https://github.com/lvis-dataset/lvis-api

.. code-block:: bash

    root=data/lvis
    mkdir -p ${root} && cd ${root}
    ln -s ../coco/train2017 .
    ln -s ../coco/val2017 .
    mkdir annotations
    cd ../..

.. code::

    data/lvis/
    ├── train2017 -> ../coco/train2017
    └── val2017 -> ../coco/val2017

LVIS v0.5
---------

.. code-block:: bash

    cd ${root}/annotations
    wget https://dl.fbaipublicfiles.com/LVIS/lvis_v0.5_{train,val}.json.zip
    unzip -q "*.zip"
    cd ../../..

.. code::

    data/lvis/
    └── annotations
        ├── lvis_v0.5_train.json
        └── lvis_v0.5_val.json

LVIS v1.0
---------

.. code-block:: bash

    cd ${root}/annotations
    wget https://dl.fbaipublicfiles.com/LVIS/lvis_v1_{train,val}.json.zip
    unzip -q "*.zip"
    cd ../../..

    python lvis_v1_minival.py

:download:`lvis_v1_minival.py <lvis_v1_minival.py>` generates
``lvis_v1_minival.json`` by selecting COCO ``val2017`` images and their
annotations from ``lvis_v1_val.json``.

.. code::

    data/lvis/
    └── annotations
        ├── lvis_v1_minival.json
        ├── lvis_v1_train.json
        └── lvis_v1_val.json
