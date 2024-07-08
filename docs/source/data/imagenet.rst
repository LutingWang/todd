ImageNet
========

https://image-net.org/

ImageNet-1k
-----------

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

.. literalinclude:: imagenet_1k.py
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
