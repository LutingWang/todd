EVA
===

https://github.com/baaivision/EVA

.. code-block:: bash

    root=pretrained/eva
    mkdir -p ${root}

EVA-CLIP
--------

https://huggingface.co/QuanSun/EVA-CLIP

.. code-block:: bash

    cd ${root}
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/QuanSun/EVA-CLIP eva_clip

    cd eva_clip
    git lfs checkout EVA02_CLIP_E_psz14_plus_s9B.pt
    cd ..

    cd ../..
