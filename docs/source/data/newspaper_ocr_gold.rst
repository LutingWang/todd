Newspaper OCR Gold
==================

.. code-block:: bash

    root=data/newspaper_ocr_gold
    hf download NealCaren/newspaper-ocr-gold --repo-type dataset --local-dir ${root}
    for s in train val test; do
        tar xzf ${root}/${s}_images.tar.gz -C ${root}
    done
