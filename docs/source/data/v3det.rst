V3Det
=====

https://v3det.openxlab.org.cn/

.. code-block:: bash

    cd data

    repo=V3Det_Backup
    git lfs install
    git clone https://huggingface.co/datasets/yhcao/${repo}.git
    cd ${repo}
        python v3det_exemplar_image_download.py
        python v3det_image_download.py
        python v3det_text_image_download.py
    cd ..

    mkdir v3det & cd v3det
        ln -s ${PWD}/../${repo}/V3Det/images
        mkdir annotations && cd annotations
            ln -s ${PWD}/../../${repo}/category_name_13204_v3det_2023_v1.txt
            ln -s ${PWD}/../../${repo}/*.json .
        cd ..
    cd ..

    cd ..

.. code::

    data/v3det/
    └── images
        ├── a00000066
        │   ├── 0_2530_11591900204_c1c10c1531_c.jpg
        │   └── ...
        └── ...

The training split contains 183,354 images, the validation split contains
29,821 images, and the test split contains 29,863 images.
There is a total of 13,433 categories in the images folder.
