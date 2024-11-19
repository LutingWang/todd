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
