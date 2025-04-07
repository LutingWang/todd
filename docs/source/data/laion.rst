LAION
=====

https://laion.ai/

.. code-block:: bash

    root=data/laion
    mkdir -p ${root}

LAION-Aesthetics
----------------

https://laion.ai/blog/laion-aesthetics/

https://opendatalab.com/LutingWang/LAION-Aesthetics/

.. code-block:: bash

    cd ${root}
    mkdir -p aesthetics/annotations
    cd ../..

LAION-Aesthetics V2 6.5+
~~~~~~~~~~~~~~~~~~~~~~~~

https://dagshub.com/DagsHub-Datasets/LAION-Aesthetics-V2-6.5plus

https://opendatalab.com/LutingWang/LAION-Aesthetics/

.. code-block:: bash

    cd ${root}/aesthetics

    split=v2_6.5plus
    mkdir ${split} && cd ${split}
    pip install dagshub
    dagshub download DagsHub-Datasets/LAION-Aesthetics-V2-6.5plus data .
    cd ..
    mv ${split}/labels.tsv annotations/${split}.tsv

    cd ../../..

.. code::

    data/laion/aesthetics/
    ├── annotations
    |   └── v2_6.5plus.tsv
    └── v2_6.5plus
        ├── 1000014033385563832.jpg
        └── ...

There are 542,247 images in total.
