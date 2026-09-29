RVL-CDIP
========

https://www.cs.cmu.edu/~aharley/rvl-cdip/

https://huggingface.co/datasets/aharley/rvl_cdip

.. code-block:: bash

    root=data/rvl-cdip
    base=https://huggingface.co/datasets/rvl_cdip/resolve/main/data
    mkdir -p ${root}/labels
    wget ${base}/rvl-cdip.tar.gz -O ${root}/rvl-cdip.tar.gz
    for f in train val test; do
        wget ${base}/${f}.txt -O ${root}/labels/${f}.txt
    done
    tar -zxf ${root}/rvl-cdip.tar.gz -C ${root}

.. code::

    data/rvl-cdip
    ├── images
    │   ├── imagesb/b/c/c/bcc60f00/0012181633.png
    │   └── ...
    ├── labels
    │   ├── train.txt                              320000 lines
    │   ├── val.txt                                 40000 lines
    │   └── test.txt                                40000 lines
    ├── readme.txt
    ├── dataset_infos.json
    └── rvl_cdip.py
