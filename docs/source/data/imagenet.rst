ImageNet
========

https://image-net.org/

ImageNet-1k
-----------

Run the following shell commands to prepare the dataset.

.. code-block:: bash

    root=data/imagenet
    mkdir -p ${root} && cd ${root}

    f=ILSVRC2012_devkit_t12.tar.gz
    wget https://image-net.org/data/ILSVRC/2012/${f}
    tar -zxf ${f}

    mkdir annotations train val
    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
    wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
    cd ../..

    python imagenet_1k.py

:download:`imagenet_1k.py <imagenet_1k.py>` is used to rename image files and
generate the annotations.
After processing, the directory structure should look like:

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
    ├── val
    │   ├── n1440764
    │   │   ├── 293.JPEG
    │   │   └── ...
    |   └── ...
    └── synsets.json

Both ``train.json`` and ``val.json`` exhibit the following structure:

.. code-block:: python

    [{"image":"12925.JPEG","synset_id":449},...]

``synsets.json`` contains the mapping from synset ID to synset information:

.. code-block:: python

    {"1":{"WNID":"n02119789","words":"kit fox, Vulpes macrotis",...},...}

ImageNet-21k
------------

.. code::

    data/imagenet-21k/
    ├── n00004475
    └── ...
