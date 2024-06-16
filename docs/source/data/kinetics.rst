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
