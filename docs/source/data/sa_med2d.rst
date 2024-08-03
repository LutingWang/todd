SA-Med2D
========

https://openxlab.org.cn/datasets/GMAI/SA-Med2D-20M

.. code-block:: bash

    cd data

    repo=SA-Med2D-20M
    git clone https://huggingface.co/datasets/OpenGVLab/${repo}
    # rm -rf ${repo}/.git

    prefix=${repo}/raw/SA-Med2D-16M
    f=SA-Med2D-16M.zip
    zip ${prefix}.zip ${prefix}.z0* ${prefix}.z10 -s=0 --out ${f}
    # rm -rf ${repo}

    unzip -q ${f}
    # rm ${f}

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
