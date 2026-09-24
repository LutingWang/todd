GTDB
====

.. code-block:: bash

    root=data/gtdb
    git clone https://github.com/uchidalab/GTDB-Dataset.git ${root}
    for f in ${root}/*.zip; do
        unzip -q ${f} -d ${root}
    done

.. code::

    data/gtdb/
    ├── GTDB-1
    │   ├── AIF_1970_493_498.csv
    │   └── ...
    └── GTDB-2
        ├── Alford94.csv
        └── ...
