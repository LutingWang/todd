Pseudo Labels
=============

MDETR
-----

.. code-block:: bash

    root=data/mdetr
    mkdir -p ${root} && cd ${root}
    f=mdetr_annotations.tar.gz
    wget "https://zenodo.org/record/4729015/files/${f}?download=1"
    mv "${f}?download=1" ${f}
    tar -zxf ${f}
    mv OpenSource annotations
    cd ../..
