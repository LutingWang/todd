PubTables-1M
============

https://github.com/microsoft/table-transformer

.. code-block:: bash

    root=data/pubtables-1m
    hf download bsmock/pubtables-1m --repo-type dataset --local-dir ${root}
    cd ${root}
    bash extract_structure_dataset.sh
    cd ../..

.. code::

    data/pubtables-1m/
    ├── PubTables-1M-Detection
    │   ├── images
    │   │   ├── PMC4967509_3.jpg
    │   │   └── ...
    │   ├── {train,val,test}
    │   │   ├── PMC1064082_1.xml
    │   │   └── ...
    │   └── words
    │       ├── PMC5836426_5_words.json
    │       └── ...
    ├── PubTables-1M-Structure
    │   ├── images
    │   │   ├── PMC4840909_table_0.jpg
    │   │   └── ...
    │   ├── {train,val,test}
    │   │   ├── PMC2268688_table_0.xml
    │   │   └── ...
    │   └── words
    │       ├── PMC4688294_table_0_words.json
    │       └── ...
    └── PubTables-1M-PDF-Annotations
        ├── PMC6170187_tables.json
        └── ...
