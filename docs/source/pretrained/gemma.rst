Gemma
=====

.. code-block:: bash

    root=pretrained/gemma
    mkdir -p ${root} && cd ${root}
    git clone git@hf.co:google/gemma-1.1-2b-it
    git clone git@hf.co:google/gemma-1.1-7b-it
    cd ../..

.. code::

    pretrained/gemma/
    ├── gemma-1.1-2b-it
    └── gemma-1.1-7b-it

.. literalinclude:: gemma.py
    :language: python
    :linenos:
