Pretrained
==========

Gemma
-----

.. code-block:: bash

    root=pretrained/gemma
    mkdir -p ${root} && cd ${root}
    git clone git@hf.co:google/gemma-1.1-2b-it
    git clone git@hf.co:google/gemma-1.1-7b-it

.. code::

    pretrained/gemma/
    ├── gemma-1.1-2b-it
    └── gemma-1.1-7b-it

.. literalinclude:: gemma.py
    :language: python
    :linenos:

Stable Diffusion
----------------

.. code-block:: bash

    root=pretrained/stable_diffusion
    mkdir -p ${root} && cd ${root}
    git clone git@hf.co:stabilityai/stable-diffusion-xl-base-1.0
    git clone git@hf.co:stabilityai/stable-diffusion-xl-refiner-1.0

.. code::

    pretrained/stable_diffusion/
    ├── stable-diffusion-xl-base-1.0
    └── stable-diffusion-xl-refiner-1.0
