TextbookQA Images
=================

https://huggingface.co/datasets/Wenjian1/TextbookQA

.. code-block:: bash

    root=data/textbookqa-images
    hf download Wenjian1/TextbookQA --repo-type dataset --local-dir ${root}
    mv ${root}/data/train-*.parquet ${root}/

.. code::

    data/textbookqa-images
    ├── README.md
    ├── sha256sums.txt
    └── train-0000{0..3}-of-00004.parquet
