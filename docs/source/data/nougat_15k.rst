nougat-15k
==========

https://www.kaggle.com/datasets/therealoranges/nougat-15k

.. code-block:: bash

    root=data/nougat-15k
    hf download deepcopy/nougat-15k --repo-type dataset --local-dir ${root}

.. code::

    data/nougat-15k/data
    ├── train-{00000..00012}-of-00013-*.parquet
    ├── validation-{00000..00003}-of-00004-*.parquet
    └── test-{00000..00001}-of-00002-*.parquet
