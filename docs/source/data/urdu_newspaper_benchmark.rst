Urdu Newspaper Benchmark
========================

https://huggingface.co/datasets/ULRs/Urdu-Newspaper-Benchmark

.. code-block:: bash

    root=data/urdu_newspaper_benchmark
    hf download ULRs/Urdu-Newspaper-Benchmark --repo-type dataset --local-dir ${root}

.. code::

    data/urdu_newspaper_benchmark
    ├── README.md
    ├── data
    │   └── train-00000-of-00001.parquet
    ├── hr-images
    │   └── {1..829}.png
    └── lr-images
        └── {1..162}.png
