SA-Med2D
========

https://openxlab.org.cn/datasets/GMAI/SA-Med2D-20M

.. code-block:: bash

    cd data
    git clone git@hf.co:datasets/OpenGVLab/SA-Med2D-20M

    prefix=SA-Med2D-20M/raw/SA-Med2D-16M
    zip ${prefix}.zip ${prefix}.z{01..10} -s 0 --out SA-Med2D-16M.zip
    unzip -q SA-Med2D-16M.zip

    mv SAMed2Dv1 sa_med2d
    cd sa_med2d
    mkdir annotations
    mv *.json annotations

    cd ../..

.. code::

    data/sa_med2d/
    ├── annotations
    │   ├── SAMed2D_v1_class_mapping_id.json
    |   └── SAMed2D_v1.json
    ├── images
    │   ├── ct_00--AbdomenCT1K--Case_00011--x_0040.png
    |   └── ... (total 3772180)
    └── masks
        ├── ct_00--AbdomenCT1K--Case_00011--x_0041--0000_001.png
        └── ...
