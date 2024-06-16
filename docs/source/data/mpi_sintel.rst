MPI-Sintel
==========

http://sintel.is.tue.mpg.de/

.. code-block:: bash

    root=data/sintel
    mkdir -p ${root} && cd ${root}
    wget http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip
    unzip -q *.zip

.. literalinclude:: optical_flow.py
    :language: python
    :linenos:

.. literalinclude:: sintel.py
    :language: python
    :linenos:

.. code::

    data/sintel/
    ├── training
    │   ├── albedo|clean|final|flow_viz|invalid|occlusions
    │   │   ├── alley_1
    │   │   │   ├── frame_0001.png
    │   │   │   └── ...
    │   │   └── ...
    │   └── flow
    │       ├── alley_1
    │       │   ├── frame_0001.flo
    │       │   └── ...
    │       └── ...
    └── test
        └── clean|final
            ├── ambush_1
            │   ├── frame_0001.png
            │   └── ...
            └── ...
