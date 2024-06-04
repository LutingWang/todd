Data
====

ImageNet-1k
-----------

.. code-block:: bash

    root=data/imagenet
    mkdir -p ${root} && cd ${root}
    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz -P ${DATA_ROOT}
    tar -zxf ILSVRC2012_devkit_t12.tar.gz

    mkdir annotations train val
    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar -P ${DATA_ROOT}
    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar -P ${DATA_ROOT}
    cd ../..

    python imagenet.py

.. literalinclude:: imagenet.py
    :language: python
    :linenos:

.. code::

    data/imagenet/
    ├── annotations
    │   ├── train.json
    |   └── val.json
    ├── train
    │   ├── n1440764
    │   │   ├── 18.JPEG
    │   │   └── ...
    |   └── ...
    └── val
        ├── n1440764
        │   ├── 293.JPEG
        │   └── ...
        └── ...


MS-COCO 2017
------------

.. code-block:: bash

    root=data/coco
    mkdir -p ${root} && cd ${root}
    wget http://images.cocodataset.org/zips/train2017.zip
    wget http://images.cocodataset.org/zips/val2017.zip
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip -q *.zip
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

MPI-Sintel
----------

.. code-block:: bash

    root=data/sintel
    mkdir -p ${root} && cd ${root}
    wget http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip
    unzip -q *.zip

.. literalinclude:: sintel.py
    :language: python
    :linenos:

.. code::

    data/sintel
    ├── training
    │   ├── albedo/clean/final/flow_viz/invalid/occlusions
    │   │   ├── alley_1
    │   │   │   ├── frame_0001.png
    │   │   │   └── ...
    │   │   └── ...
    │   └── flow
    │       ├── alley_1
    │       │   ├── frame_0001.flo
    │       │   └── ...
    │       └── ...
    └── test
        └── clean/final
            ├── ambush_1
            │   ├── frame_0001.png
            │   └── ...
            └── ...
