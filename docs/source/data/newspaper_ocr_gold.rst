Newspaper OCR Gold
==================

https://huggingface.co/datasets/NealCaren/newspaper-ocr-gold

.. code-block:: bash

    root=data/newspaper_ocr_gold
    hf download NealCaren/newspaper-ocr-gold --repo-type dataset --local-dir ${root}
    for s in train val test; do
        tar xzf ${root}/${s}_images.tar.gz -C ${root}
    done

.. code::

    data/newspaper_ocr_gold
    ├── README.md
    ├── sample_metadata.json
    ├── verified_lines.jsonl
    ├── {train,val,test}_images.tar.gz
    ├── {train,val,test}
    │   └── <page_id>/lines/line_NNNN.png
    └── data
        ├── train-{00000..00001}-of-00002.parquet
        ├── val-00000-of-00001.parquet
        └── test-00000-of-00001.parquet
