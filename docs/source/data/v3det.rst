V3Det
=====

https://v3det.openxlab.org.cn/

.. code-block:: bash

    cd data

    repo=V3Det_Backup
    git lfs install
    git clone https://huggingface.co/datasets/yhcao/${repo}.git
    cd ${repo}
        git apply v3det.txt
        python v3det_exemplar_image_download.py
        python v3det_image_download.py
        python v3det_text_image_download.py
    cd ..

    mkdir v3det & cd v3det
        ln -s ../${repo}/V3Det/images train
        ln -s ../${repo}/V3Det/test
        mkdir annotations & cd annotations
            ln -s ../../${repo}/*.json .
        cd ..
    cd ..

    cd ..

.. literalinclude:: v3det.txt

.. code::

    data/v3det/
