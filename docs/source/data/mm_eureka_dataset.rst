MM-Eureka Dataset
=================

.. code-block:: bash

    root=data/mm-eureka-dataset
    hf download FanqingM/MM-Eureka-Dataset --repo-type dataset --local-dir ${root}

    tmp=$(mktemp -d)
    p=inspire/hdd/global_user/shaowenqi-shaowenqi/mengfanqing/OpenRLHF-InternVL/dataset/report_data
    for f in K12 MMPR; do
        unzip -q ${root}/${f}.zip -d ${tmp} && mv ${tmp}/${p}/${f} ${root}/
    done
    rm -rf ${tmp}

.. code::

    data/mm-eureka-dataset
    ├── dataset.jsonl
    ├── K12
    │   ├── 56041f8b15c83565bb9fddd66ecb09ad52ac5df2cf51b6757bb66642b4a870f5.png
    │   └── ...
    └── MMPR
        └── images
            ├── ai2d
            └── ...
