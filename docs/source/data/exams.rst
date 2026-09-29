EXAMS
=====

https://huggingface.co/datasets/mhardalov/exams

https://github.com/mhardalov/exams-qa

.. code-block:: bash

    root=data/exams
    hf download mhardalov/exams --repo-type dataset --local-dir ${root}

.. code::

    data/exams
    ├── alignments
    │   └── full-00000-of-00001.parquet
    ├── crosslingual_{bg,hr,hu,it,mk,pl,pt,sq,sr,tr,vi}
    │   └── {train,validation}-00000-of-00001.parquet
    ├── crosslingual_test
    │   └── test-00000-of-00001.parquet
    ├── crosslingual_with_para_{bg,hr,hu,it,mk,pl,pt,sq,sr,tr,vi}
    │   └── {train,validation}-00000-of-00001.parquet
    ├── crosslingual_with_para_test
    │   └── test-00000-of-00001.parquet
    ├── multilingual
    │   └── {train,validation,test}-00000-of-00001.parquet
    ├── multilingual_with_para
    │   └── {train,validation,test}-00000-of-00001.parquet
    └── README.md  .gitattributes
