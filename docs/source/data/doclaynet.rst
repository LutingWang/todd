DocLayNet
=========

https://github.com/DS4SD/DocLayNet

.. code-block:: bash

    root=data/doclaynet
    hf download docling-project/DocLayNet-v1.2 --repo-type dataset --local-dir ${root}

.. code::

    data/doclaynet/data
    ├── train-{00000..00071}-of-00072.parquet
    ├── validation-{00000..00006}-of-00007.parquet
    └── test-{00000..00005}-of-00006.parquet
