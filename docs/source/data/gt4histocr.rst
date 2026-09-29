GT4HistOCR
==========

https://zenodo.org/records/1344132

https://github.com/qurator-spk/train-calamari-gt4histocr

.. code-block:: bash

    root=data/gt4histocr
    mkdir -p ${root} && cd ${root}
    base=https://zenodo.org/records/1344132/files
    wget "${base}/GT4HistOCR.tar?download=1" -O GT4HistOCR.tar
    tar xf GT4HistOCR.tar
    cd ../..

.. code::

    data/gt4histocr
    ├── corpus
    │   ├── dta19.tar.bz2
    │   ├── EarlyModernLatin.tar.bz2
    │   ├── Kallimachos.tar.bz2
    │   ├── RefCorpus-ENHG-Incunabula.tar.bz2
    │   └── RIDGES-Fraktur.tar.bz2
    ├── models
    │   ├── incunabula-00184000.pyrnn.gz
    │   ├── latin1-00081000.pyrnn.gz
    │   ├── latin2-00069000.pyrnn.gz
    │   ├── ridges1-00085000.pyrnn.gz
    │   └── ridges2-00062000.pyrnn.gz
    └── tools
        └── regularize.pl
