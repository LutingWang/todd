TAL
===

https://ai.100tal.com/dataset

https://huggingface.co/datasets/LutingWang/tal

.. code-block:: bash

    root=data/tal
    mkdir -p ${root}
    hf download LutingWang/tal --repo-type dataset --local-dir ${root}

HME100K
-------

.. code-block:: bash

    cd ${root}
    unzip -q HME100K.zip
    cd HME100K
    unzip -q "*.zip"
    cd ../../..

.. code::

    data/tal/HME100K/
    ├── subset
    │   ├── easy.json
    │   ├── medium.json
    │   └── hard.json
    ├── train_images
    │   ├── train_0.jpg
    │   └── ...
    ├── train_labels.txt
    ├── test_images
    │   ├── test_2.jpg
    │   └── ...
    └── test_labels.txt


K-12 印刷体
-----------

.. code-block:: bash

    cd ${root}
    unzip -q "K-12 印刷体.zip"
    cd ../..

.. code::

    data/tal/印刷体/
    ├── images
    │   ├── 00IODs4kpkp6GSX7mcH4_z2ET-YvtuVHnk65O_WfZV4%3D_0.jpg
    │   └── ...
    └── labels
        └── label.txt
