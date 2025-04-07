Open Images
===========

https://storage.googleapis.com/openimages/web/index.html

https://github.com/cvdfoundation/open-images-dataset

.. code-block:: bash

    root=data/open_images
    mkdir -p ${root}/annotations

    for i in {0..9}; do
        echo ${i}
        aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_${i}.tar.gz ${root}
    done
    for i in {a..f}; do
        echo ${i}
        aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_${i}.tar.gz ${root}
    done
    aws s3 --no-sign-request cp s3://open-images-dataset/tar/validation.tar.gz ${root}
    aws s3 --no-sign-request cp s3://open-images-dataset/tar/test.tar.gz ${root}

    for f in ${root}/*.tar.gz; do
        echo ${f}
        tar -zxf ${f} -C ${root}
    done

.. code::

    data/open_images/
    ├── train_0
    |   ├── 000002b66c9c498e.jpg
    |   └── ...
    ├── ...
    ├── validation
    |   ├── 0001eeaf4aed83f9.jpg
    |   └── ...
    └── test
        ├── 000026e7ee790996.jpg
        └── ...

The training split contains 1,743,042 images, the validation split contains
41,620 images, and the test split contains 125,436 images.

Open Images v6
--------------

.. code-block:: bash

    wget https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv -P ${root}/annotations
