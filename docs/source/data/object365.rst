Object365
=========

https://www.objects365.org/

.. code-block:: bash

    root=data/object365
    mkdir -p ${root} && cd ${root}
        mkdir annotations train val
    cd ../..

    base_url=https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86

    download() {
        local url=$1
        local i=$2

        patch=patch${i}.tar.gz
        wget ${url}/${patch}
        tar -zxf ${patch}
    }

.. code::

    data/object365/

Object365 v1
------------

.. code-block:: bash

    cd ${root}
        train_v1=train/v1
        mkdir ${train_v1} && cd ${train_v1}
            for i in {0..15}; do download ${base_url}/train ${i}; done
        cd ../..

        val_v1=val/v1
        mkdir ${val_v1} && cd ${val_v1}
            for i in {0..15}; do download ${base_url}/val/images/v1 ${i}; done
        cd ../..
    cd ../..

Object365 v2
------------

.. code-block:: bash

    cd ${root}
        cd annotations
            annotations_train_v2=zhiyuan_objv2_train.tar.gz
            wget ${base_url}/train/${annotations_train_v2}
            tar -zxf ${annotations_train_v2}

            annotations_val_v2=zhiyuan_objv2_val.json
            wget ${base_url}/val/${annotations_val_v2}
        cd ..

        train_v2=train/v2
        mkdir ${train_v2} && cd ${train_v2}
            for i in {16..50}; do download ${base_url}/train ${i}; done
        cd ../..

        val_v2=val/v2
        mkdir ${val_v2} && cd ${val_v2}
            for i in {16..43}; do download ${base_url}/val/images/v2 ${i}; done
        cd ../..
    cd ../..
