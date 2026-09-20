FinTabNet
=========

https://developer.ibm.com/exchanges/data/all/fintabnet/

FinTabNet.c
-----------

https://github.com/microsoft/table-transformer

.. code-block:: bash

    root=data/fintabnet-c
    hf download bsmock/FinTabNet.c --repo-type dataset --local-dir ${root}
    cd ${root}
    for f in *.tar.gz; do
        tar -zxf ${f}
    done
    cd ../..

.. code::

    data/fintabnet-c/
    ├── FinTabNet.c-Structure
    │   ├── images
    │   │   ├── FAST_2015_page_67_table_2.jpg
    │   │   └── ...
    │   ├── {train,val,test}
    │   │   ├── AAL_2002_page_41_table_1.xml
    │   │   └── ...
    │   └── words
    │       ├── ETFC_2016_page_154_table_0_words.json
    │       └── ...
    └── FinTabNet.c-PDF_Annotations
        ├── FLS_2012_page_67_tables.json
        └── ...
