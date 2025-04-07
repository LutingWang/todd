Objects365
==========

https://www.objects365.org/

Objects365 v1
-------------

Objects365 v2
-------------

.. code-block:: bash

    root=data/objects365v2
    mkdir -p ${root} && cd ${root}

    base_url=https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86

    mkdir annotations && cd annotations
    wget ${base_url}/train/zhiyuan_objv2_train.tar.gz
    wget ${base_url}/val/zhiyuan_objv2_val.json
    tar -zxf *.tar.gz
    cd ..

    mkdir train && cd train
    for i in {0..50}; do wget ${base_url}/train/patch${i}.tar.gz; done
    for f in *.tar.gz; do echo ${f}; tar -zxf ${f}; done
    cd ..

    mkdir val && cd val
    for i in {0..15}; do wget ${base_url}/val/images/v1/patch${i}.tar.gz; done
    for i in {16..43}; do wget ${base_url}/val/images/v2/patch${i}.tar.gz; done
    for f in *.tar.gz; do echo ${f}; tar -zxf ${f}; done
    cd ..

    cd ../..

.. code::

    data/objects365v2/
    ├── annotations
    |   ├── zhiyuan_objv2_train.json
    |   └── zhiyuan_objv2_val.json
    ├── train
    |   ├── patch0
    |   |   ├── objects365_v1_00000000.jpg
    |   |   └── ...
    |   └── ...
    └── val
        ├── patch0
        |   ├── objects365_v1_00000016.jpg
        |   └── ...
        └── ...

There are 51 patches in the train split and 44 patches in the val split.
