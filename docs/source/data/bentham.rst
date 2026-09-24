Bentham
=======

https://zenodo.org/records/44519

.. code-block:: bash

    root=data/bentham
    mkdir -p ${root} && cd ${root}
    base=https://zenodo.org/records/44519/files
    wget ${base}/BenthamDatasetR0-{Images,GT}.tbz
    for f in *.tbz; do tar -jxf ${f}; done
    cd ../..

.. code::

    data/bentham/
    ├── BenthamDatasetR0-Images/Images/Pages
    │   ├── 071_184_003.jpg
    │   └── ...
    └── BenthamDatasetR0-GT
        ├── Images/Lines
        │   ├── 096_051_002_03_03.png
        │   └── ...
        ├── PAGE
        │   ├── 071_184_003.xml
        │   └── ...
        └── Transcriptions
            ├── 115_073_002_02_06.txt
            └── ...
