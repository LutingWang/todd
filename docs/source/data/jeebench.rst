JEEBench
========

https://github.com/dair-iitd/jeebench

.. code-block:: bash

    root=data/jeebench
    hf download daman1209arora/jeebench --repo-type dataset --local-dir ${root}
    cd ${root} && unzip -q data.zip && cd ../..

.. code::

    data/jeebench
    ├── data
    │   ├── dataset.json
    │   └── few_shot_examples.json
    └── test.json
