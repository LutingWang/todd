SciTSR-Logical
==============

https://huggingface.co/datasets/saeed11b95/SciTsr-Logical

.. code-block:: bash

    root=data/scitsr-logical
    hf download saeed11b95/SciTsr-Logical --repo-type dataset --local-dir ${root}
    tar -zxf ${root}/SciTsr_Logical.tar.gz -C ${root}

.. code::

    data/scitsr-logical
    ├── README.md
    ├── SciTsr_Logical.tar.gz
    └── SciTsr_Logical
        ├── train
        │   ├── gt
        │   │   ├── 0001020v1.11.json
        │   │   └── ...
        │   └── images
        │       ├── 0001020v1.11.png
        │       └── ...
        └── test
            ├── gt
            └── images
