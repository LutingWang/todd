SynthTabNet
===========

https://github.com/IBM/SynthTabNet

.. code-block:: bash

    root=data/synthtabnet
    mkdir -p ${root} && cd ${root}
    base=https://ds4sd-public-artifacts.s3.eu-de.cloud-object-storage.appdomain.cloud/datasets/synthtabnet_public/v2.0.0
    wget ${base}/{fintabnet,marketing,pubtabnet,sparse}.zip
    unzip -q "*.zip"
    cd ../..

.. code::

    data/synthtabnet/
    └── {fintabnet,marketing,pubtabnet,sparse}
        ├── images/{train,val,test}
        │   ├── image_000000_1634629328.513163.png
        │   └── ...
        └── synthetic_data.jsonl
