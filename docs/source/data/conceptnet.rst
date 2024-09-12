ConceptNet
==========

https://github.com/commonsense/conceptnet5

.. code-block:: bash

    root=data/conceptnet
    mkdir -p ${root}

ConceptNet 5.7.0
----------------

.. code-block:: bash

    wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz -P ${root}

Numberbatch
-----------

https://github.com/commonsense/conceptnet-numberbatch

.. code-block:: bash

    wget http://conceptnet.s3.amazonaws.com/precomputed-data/2016/numberbatch/19.08/mini.h5 -P ${root}
