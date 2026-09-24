MathWriting
===========

https://github.com/google-research/google-research/tree/master/mathwriting

.. code-block:: bash

    root=data/mathwriting
    mkdir -p ${root} && cd ${root}
    base=https://storage.googleapis.com/mathwriting_data
    wget ${base}/mathwriting-2024{,-excerpt}.tgz
    for f in *.tgz; do tar -zxf ${f}; done
    cd ../..

.. code::

    data/mathwriting/
    └── mathwriting-2024
        ├── {train,valid,test,synthetic,symbols}
        │   ├── 708b55f278d89aad.inkml
        │   └── ...
        ├── symbols.jsonl
        └── synthetic-bboxes.jsonl
