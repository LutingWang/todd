MMTab
=====

https://github.com/SpursGoZmy/Table-LLaVA

.. code-block:: bash

    root=data/mmtab
    hf download SpursgoZmy/MMTab --repo-type dataset --local-dir ${root}
    for f in ${root}/*.zip; do
        unzip -q ${f} -d ${root}
    done

.. code::

    data/mmtab
    ├── README.md
    ├── all_test_image
    │   ├── AIT-QA_tab-0.jpg
    │   └── ...
    ├── IID_train_image
    │   ├── FeTaQA_dev_example-0.jpg
    │   └── ...
    └── table_pretrain_part_2
        ├── ToTTo_train_table_0.jpg
        └── ...
