LAION
=====

https://laion.ai/

LAION-Aesthetics
----------------

https://laion.ai/blog/laion-aesthetics/

.. code-block:: bash

    root=data/laion/aesthetics
    mkdir -p ${root} && cd ${root}
    mkdir annotations
    cd ../../..

.. code::

    data/laion/aesthetics/
    ├── annotations
    |   └── v2_6.5plus.tsv
    └── v2_6.5plus
        └── ...

LAION-Aesthetics V2 6.5+
~~~~~~~~~~~~~~~~~~~~~~~~

https://dagshub.com/DagsHub-Datasets/LAION-Aesthetics-V2-6.5plus

.. code-block:: bash

    split=v2_6.5plus
    mkdir ${split} && cd ${split}
    pip install dagshub
    dagshub download DagsHub-Datasets/LAION-Aesthetics-V2-6.5plus data .
    cd ..

    mv ${split}/labels.tsv annotations/${split}.tsv
