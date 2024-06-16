DAVIS
=====

https://davischallenge.org/

DAVIS 2017
----------

.. code-block:: bash

    cd data
    f=DAVIS-2017-trainval-480p.zip
    wget https://data.vision.ee.ethz.ch/csergi/share/davis/${f}
    unzip -q ${f}
    mv DAVIS davis

.. code::

    data/davis/
    ├── Annotations/480p/
    │   ├── bear
    │   │   ├── 00000.png
    │   │   └── ...
    |   └── ...
    ├── ImageSets/2017/
    │   ├── train.txt
    |   └── val.txt
    └── JPEGImages/480p/
        ├── bear
        │   ├── 00000.jpg
        │   └── ...
        └── ...
