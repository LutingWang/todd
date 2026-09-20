DIBCO
=====

`Document Image Binarization COmpetition <https://vc.ee.duth.gr/>`_

https://huggingface.co/datasets/LutingWang/dibco

.. code-block:: bash

    root=data/dibco
    hf download LutingWang/dibco --repo-type dataset --local-dir ${root}

DIBCO2009
---------

.. code-block:: bash

    mkdir -p ${root}/DIBCO-2009 && cd ${root}/DIBCO-2009
    unrar x "*.rar"
    cd ../../..

.. code::

    data/dibco/DIBCO-2009/
    ├── DIBC02009_Test_images-handwritten
    │   └── H{01..05}.bmp
    ├── DIBCO2009_Test_images-printed
    │   └── P{01..05}.bmp
    ├── DIBCO2009-GT-Test-images_handwritten
    │   └── H{01..05}.tiff
    └── DIBCO2009-GT-Test-images_printed
        └── P{01..05}.tiff

H-DIBCO2010
-----------

.. code-block:: bash

    mkdir -p ${root}/H-DIBCO-2010 && cd ${root}/H-DIBCO-2010
    unrar x "*.rar"
    cd ../../..

.. code::

    data/dibco/H-DIBCO-2010/
    ├── H{01..10}.*
    ├── H{01..10}_estGT.tiff
    └── H{01..10}_skelGT.tiff

DIBCO2011
---------

.. code-block:: bash

    mkdir -p ${root}/DIBCO-2011 && cd ${root}/DIBCO-2011
    unrar x "*.rar"
    cd ../../..

.. code::

    data/dibco/DIBCO-2011/
    ├── HW{1..8}.png
    ├── HW{1..8}_GT.tiff
    ├── PR{1..8}.png
    └── PR{1..8}_GT.tiff

H-DIBCO2012
-----------

.. code-block:: bash

    mkdir -p ${root}/H-DIBCO-2012 && cd ${root}/H-DIBCO-2012
    f=H-DIBCO2012-dataset.rar
    unrar x ${f}
    cd ../../..

.. code::

    data/dibco/H-DIBCO-2012/H-DIBCO2012-dataset/H-DIBCO2012-dataset/
    ├── H{01..14}.png
    └── H{01..14}_GT.tif

DIBCO2013
---------

.. code-block:: bash

    mkdir -p ${root}/DIBCO-2013 && cd ${root}/DIBCO-2013
    f=DIBCO2013-dataset.rar
    unrar x ${f}
    cd ../../..

.. code::

    data/dibco/DIBCO-2013/
    ├── OriginalImages
    │   ├── HW{01..08}.*
    │   └── PR{01..08}.bmp
    └── GTimages
        ├── HW{01..08}_estGT.tiff
        └── PR{01..08}_estGT.tiff

H-DIBCO2014
-----------

.. code-block:: bash

    mkdir -p ${root}/H-DIBCO-2014 && cd ${root}/H-DIBCO-2014
    unrar x "*.rar"
    cd ../../..

.. code::

    data/dibco/H-DIBCO-2014/
    ├── H{01..10}.png
    └── H{01..10}_estGT.tiff

H-DIBCO2016
-----------

.. code-block:: bash

    mkdir -p ${root}/H-DIBCO-2016 && cd ${root}/H-DIBCO-2016
    unzip -q "*.zip"
    cd ../../..

.. code::

    data/dibco/H-DIBCO-2016/
    ├── DIPCO2016_dataset
    │   └── {1..10}.bmp
    └── DIPCO2016_Dataset_GT
        └── {1..10}_gt.bmp

DIBCO2017
---------

.. code-block:: bash

    mkdir -p ${root}/DIBCO-2017 && cd ${root}/DIBCO-2017
    7z x -y "*.7z"
    cd ../../..

.. code::

    data/dibco/DIBCO-2017/
    ├── Dataset
    │   └── {1..20}.bmp
    └── GT
        └── {1..20}_gt.bmp

H-DIBCO2018
-----------

.. code-block:: bash

    mkdir -p ${root}/H-DIBCO-2018 && cd ${root}/H-DIBCO-2018
    unzip -q "*.zip"
    cd ../../..

.. code::

    data/dibco/H-DIBCO-2018/
    ├── dataset
    │   └── {1..10}.bmp
    ├── gt
    │   └── {1..10}_gt.bmp
    └── weights
        └── {1..10}_gt_{P,R}Weights.dat

DIBCO2019
---------

.. code-block:: bash

    mkdir -p ${root}/DIBCO-2019 && cd ${root}/DIBCO-2019
    unzip -q "*.zip"
    cd ../../..

.. code::

    data/dibco/DIBCO-2019/
    ├── Dataset
    │   └── {1..20}.bmp
    ├── GT
    │   └── {1..20}.bmp
    └── Weights
        └── {1..20}_{P,R}Weights.dat
