CASIA-HWDB and CASIA-OLHWDB
===========================

`CASIA Online and Offline Chinese Handwriting Databases <https://nlpr.ia.ac.cn/databases/handwriting/Home.html>`_

https://huggingface.co/datasets/LutingWang/casia-hwdb

=========  =======  ===================
Dataset
=========  =======  ===================
HWDB1.0    Offline  Isolated characters
HWDB1.1    Offline  Isolated characters
HWDB1.2    Offline  Isolated characters
OLHWDB1.0  Online   Isolated characters
OLHWDB1.1  Online   Isolated characters
OLHWDB1.2  Online   Isolated characters
HWDB2.0    Offline  Handwritten texts
HWDB2.1    Offline  Handwritten texts
HWDB2.2    Offline  Handwritten texts
OLHWDB2.0  Online   Handwritten texts
OLHWDB2.1  Online   Handwritten texts
OLHWDB2.2  Online   Handwritten texts
=========  =======  ===================

.. code-block:: bash

    root=data/casia-hwdb
    mkdir -p ${root}
    base=https://nlpr.ia.ac.cn/Download

CASIA-HWDB1
-----------

.. code-block:: bash

    wget \
        ${base}/Offline/CharData/Gnt1.0{TrainPart{1..3},Test}.zip \
        ${base}/Offline/CharData/Gnt1.{1..2}{TrainPart{1..2},Test}.zip \
        -P ${root}

    for i in {0..2}; do
        unzip -q "${root}/Gnt1.${i}TrainPart*.zip" -d ${root}/Gnt1.${i}Train
        unzip -q ${root}/Gnt1.${i}Test.zip -d ${root}/Gnt1.${i}Test
    done

.. code::

    data/casia-hwdb/
    └── Gnt1.{0..2}{Train,Test}
        ├── 001-f.gnt
        └── ...

CASIA-OLHWDB1
-------------

.. code-block:: bash

    for f in Pot1.{0..2}{Train,Test}.zip; do
        wget ${base}/Online/CharData/${f} -P ${root}
        unzip -q ${root}/${f} -d ${root}/${f%.zip}
    done

.. code::

    data/casia-hwdb/
    └── Pot1.{0..2}{Train,Test}
        ├── 001.pot
        └── ...

CASIA-HWDB2
-----------

.. code-block:: bash

    for f in HWDB2.{0..2}{Train,Test}.zip; do
        wget ${base}/Offline/TextlineData/${f} -P ${root}
        unzip -q ${root}/${f} -d ${root}/${f%.zip}
    done

.. code::

    data/casia-hwdb/
    └── HWDB2.{0..2}{Train,Test}
        ├── 001-P16.dgrl
        └── ...

CASIA-OLHWDB2
-------------

.. code-block:: bash

    for f in WPTT2.{0..2}-{Train,Test}.zip; do
        wget ${base}/Online/WPTTData/${f} -P ${root}
        unzip -q ${root}/${f} -d ${root}/${f%.zip}
    done

.. code::

    data/casia-hwdb/
    └── WPTT2.{0..2}-{Train,Test}
        ├── 501-P14.wptt
        └── ...

Competition
-----------

.. code-block:: bash

    mkdir ${root}/competition
    wget \
        ${base}/competition/competition{-gnt,_POT}.zip \
        ${base}/Offline/TextlineData/Competition13Line.zip \
        ${base}/Online/WPTTData/Competition13wptt.zip \
        -P ${root}/competition
    for f in ${root}/competition/*.zip; do
        unzip -q ${f} -d ${f%.zip}
    done

.. code::

    data/casia-hwdb/competition/
    ├── competition-gnt
    │   ├── C001-f-f.gnt
    │   └── ...
    ├── competition_POT
    │   ├── C001-f.pot
    │   └── ...
    ├── Competition13Line
    │   ├── C001-P16.dgrl
    │   └── ...
    └── Competition13wptt
        ├── C001-P16.wptt
        └── ...
