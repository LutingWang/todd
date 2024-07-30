Kinetics
========

https://github.com/cvdfoundation/kinetics-dataset

Kinetics-700-2020
-----------------

.. code-block:: bash

    cd data
    repo=kinetics-dataset
    git clone https://github.com/cvdfoundation/${repo}.git
    cd ${repo}
    bash ./k700_2020_downloader.sh
    bash ./k700_2020_extractor.sh
    cd ..
    mkdir kinetics
    ln -s ${PWD}/${repo}/k700-2020 kinetics
    cd ..

.. code::

    data/kinetics/
    └── k700-2020
        ├── annotations
        │   ├── test.csv
        │   ├── train.csv
        │   └── val.csv
        ├── train
        │   ├── abseiling
        │   │   ├── __NrybzYzUg_000415_000425.mp4
        │   │   └── ...
        │   └── ...
        └── val
            ├── abseiling
            │   ├── 3E7Jib8Yq5M_000118_000128.mp4
            │   └── ...
            └── ...
