DocBank
=======

https://github.com/doc-analysis/DocBank

https://huggingface.co/datasets/liminghao1630/DocBank

.. code-block:: bash

    root=data/docbank
    hf download liminghao1630/DocBank --repo-type dataset --local-dir ${root}
    7z x ${root}/DocBank_500K_ori_img.zip.001 -o${root}
    unzip -q "${root}/*.zip" -d ${root}

.. code::

    data/docbank/
    ├── DocBank_500K_ori_img
    │   ├── 1.tar_1401.0001.gz_infoingames_without_metric_arxiv_0_ori.jpg
    │   └── ...
    ├── DocBank_500K_txt
    │   ├── 1.tar_1401.0001.gz_infoingames_without_metric_arxiv_0.txt
    │   └── ...
    └── MSCOCO_Format_Annotation
        └── 500K_{all,train,valid,test}.json
