Object365
=========

https://www.objects365.org/

Object365 v2
------------

.. code-block:: bash

    root=data/object365
    mkdir -p ${root} && cd ${root}

    mkdir annotations train val

    base_url="https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86"

    cd annotations
        annotations_train=zhiyuan_objv2_train.tar.gz
        wget ${base_url}/train/${annotations_train}
        tar -zxf ${annotations_train}

        wget ${base_url}/val/zhiyuan_objv2_val.json
    cd ..

    cd train
        for i in {0..50}; do
            patch=patch${i}.tar.gz
            wget ${base_url}/train/${patch}
            tar -zxf ${patch}
        done
    cd ..

    cd val
        for i in {0..15}; do
            patch=patch${i}.tar.gz
            wget ${base_url}/val/images/v1/${patch}
            tar -zxf ${patch}
        done
        for i in {16..43}; do
            patch=patch${i}.tar.gz
            wget ${base_url}/val/images/v2/${patch}
            tar -zxf ${patch}
        done
    cd ..

    cd ../..

.. code::

    data/v3det/
