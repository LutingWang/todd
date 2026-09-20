UniMER
======

https://github.com/opendatalab/UniMERNet

.. code-block:: bash

    root=data/unimer-1m
    hf download wanderkid/UniMER_Dataset --repo-type dataset --local-dir ${root}
    unzip -q "${root}/*.zip" -d ${root}

.. code::

    data/unimer-1m/
    ├── UniMER-1M
    │   ├── images
    │   │   └── {0000000..1061790}.png
    │   └── train.txt
    └── UniMER-Test
        ├── {cpe,hwe,sce,spe}
        │   ├── 0000000.png
        │   └── ...
        └── {cpe,hwe,sce,spe}.txt
