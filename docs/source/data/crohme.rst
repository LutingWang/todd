CROHME
======

`Competition on Recognition of Handwritten Mathematical Expressions <https://zenodo.org/records/8428035>`_

https://huggingface.co/datasets/LutingWang/crohme

.. code-block:: bash

    root=data/crohme
    mkdir -p ${root} && cd ${root}
    f=CROHME23.zip
    wget "https://zenodo.org/records/8428035/files/${f}?download=1" -O ${f}
    unzip -q ${f}
    cd ../..

.. code::

    data/crohme/TC11_CROHME23/
    ├── IMG/{train,val,test}/<source>/*.png
    ├── INKML/{train,val,test}/<source>/*.inkml
    └── SymLG/{train,val,test}/<source>/*.lg

.. table::

    =======================================================  ========  ========  ========
    SOURCE                                                   IMG       INKML     SymLG
    =======================================================  ========  ========  ========
    train/CROHME2023_train                                   misspell  |yes|     |yes|
    train/CROHME2019                                         |yes|     |yes|     misspell
    train/OffHME                                             |yes|     |no|      |yes|
    val/CROHME2016_test                                      |yes|     |yes|     |yes|
    val/CROHME2023_val                                       |yes|     |yes|     |yes|
    test/CROHME2019_test                                     |yes|     |yes|     |yes|
    test/CROHME2023_test                                     |yes|     |yes|     |yes|
    train/Artificial_data/gen_LaTeX_data_CROHME_2019         |no|      |yes|     |yes|
    train/Artificial_data/gen_LaTeX_data_CROHME_2023_corpus  |no|      |yes|     |yes|
    train/Artificial_data/gen_syntactic_data                 |no|      misspell  |yes|
    =======================================================  ========  ========  ========

.. |yes| unicode:: U+2713 .. CHECK MARK
.. |no| unicode:: U+2717 .. BALLOT X
