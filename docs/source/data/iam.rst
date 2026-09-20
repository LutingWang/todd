IAM
===

https://fki.tic.heia-fr.ch/

https://huggingface.co/datasets/LutingWang/iam

.. code-block:: bash

    root=data/iam
    hf download LutingWang/iam --repo-type dataset --local-dir ${root}

IAM Handwriting Database
------------------------

.. code-block:: bash

    d="${root}/IAM Handwriting Database"
    for f in ascii lines sentences words xml; do
        mkdir -p "${d}/${f}"
        tar -xzf "${d}/${f}.tgz" -C "${d}/${f}"
    done
    mkdir -p "${d}/forms"
    for f in formsA-D formsE-H formsI-Z; do
        tar -xzf "${d}/${f}.tgz" -C "${d}/forms"
    done

.. code::

    data/iam/IAM Handwriting Database/
    ├── ascii
    │   └── {forms,lines,sentences,words}.txt
    ├── forms
    │   ├── a01-000u.png
    │   └── ...
    ├── lines
    │   ├── a01/a01-000u/a01-000u-00.png
    │   └── ...
    ├── sentences
    │   ├── a01/a01-000u/a01-000u-s00-00.png
    │   └── ...
    ├── words
    │   ├── a01/a01-000u/a01-000u-00-00.png
    │   └── ...
    └── xml
        ├── a01-000u.xml
        └── ...

IAM On-Line Handwriting Database
--------------------------------

.. code-block:: bash

    d="${root}/IAM On-Line Handwriting Database"
    for f in ascii-all lineImages-all lineStrokes-all \
        original-xml-all original-xml-part; do
        mkdir -p "${d}/${f}"
        tar -xzf "${d}/${f}.tar.gz" -C "${d}/${f}"
    done

.. code::

    data/iam/IAM On-Line Handwriting Database/
    ├── ascii-all
    │   └── ascii
    │       ├── a01/a01-000/a01-000u.txt
    │       └── ...
    ├── lineImages-all
    │   └── lineImages
    │       ├── a01/a01-000/a01-000u-01.tif
    │       └── ...
    ├── lineStrokes-all
    │   └── lineStrokes
    │       ├── a01/a01-000/a01-000u-01.xml
    │       └── ...
    ├── original-xml-all
    │   └── original
    │       ├── a01/a01-000/strokesu.xml
    │       └── ...
    ├── original-xml-part
    │   └── original
    │       ├── a01/a01-001/strokesz.xml
    │       └── ...
    └── writers.xml

IAM-HistDB
----------

.. code-block:: bash

    cd ${root}/IAM-HistDB
    unzip -q "*.zip"
    cd ../../..

IAMonDo-database
----------------

.. code-block:: bash

    d=${root}/IAMonDo-database
    tar -xzf ${d}/IAMonDo-db-1.0.tar.gz -C ${d}
