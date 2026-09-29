NewsEye ICPR2020
================

https://zenodo.org/records/4943582

.. code-block:: bash

    root=data/newseye_icpr2020
    mkdir -p ${root} && cd ${root}
    base=https://zenodo.org/records/4943582/files

    for f in {simple,complex}_pages_{train,test}.zip; do
        wget "${base}/${f}?download=1" -O ${f}
        unzip -q ${f}
    done
    # the test ground truth is AES-256 encrypted, and Info-ZIP's unzip cannot
    # read AES archives at all -- it fails even given the correct password.
    # Use 7z with the passwords published in the Zenodo record description.
    for t in simple complex; do
        wget "${base}/${t}_pages_test_gt.zip?download=1" -O ${t}_pages_test_gt.zip
        7z x -y -p"icpr2020!tb_${t}" ${t}_pages_test_gt.zip
    done
    cd ../..

.. code::

    data/newseye_icpr2020
    ├── {simple,complex}_pages_train.zip
    ├── {simple,complex}_pages_test.zip
    ├── {simple,complex}_pages_test_gt.zip
    ├── {simple,complex}_pages_train
    │   ├── images/<page_id>.{jpg,png,tif}
    │   └── xmls/<page_id>.xml
    ├── {simple,complex}_pages_test
    │   ├── images/<page_id>.{jpg,png,tif}
    │   └── xmls/<page_id>.xml
    └── {simple,complex}_pages_test_gt
        ├── images
        └── xmls
