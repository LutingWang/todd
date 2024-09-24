MS-COCO
=======

https://cocodataset.org/

MS-COCO 2017
------------

.. code-block:: bash

    root=data/coco
    mkdir -p ${root} && cd ${root}
    wget http://images.cocodataset.org/zips/train2017.zip
    wget http://images.cocodataset.org/zips/val2017.zip
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    for f in *.zip; do unzip -q ${f}; done
    cd ../..

.. code::

    data/coco/
    ├── annotations
    │   ├── instances_train2017.json
    |   └── instances_val2017.json
    ├── train2017
    │   ├── 000000000009.jpg
    │   └── ...
    └── val2017
        ├── 000000000139.jpg
        └── ...
