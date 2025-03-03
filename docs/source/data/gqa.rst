GQA
===

https://cs.stanford.edu/people/dorarad/gqa/index.html

.. code-block:: bash

    root=data/gqa
    mkdir -p ${root} && cd ${root}
    f=images.zip
    wget https://downloads.cs.stanford.edu/nlp/data/gqa/${f}
    unzip -q ${f}
    cd ../..

.. code::

    data/gqa/
    └── images
        ├── 1.jpg
        └── ...

There is a total of 148,854 images.
