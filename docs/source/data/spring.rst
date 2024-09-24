Spring
======

https://spring-benchmark.org/

.. code-block:: bash

    # download train_frame_left.zip and train_flow_FW_left.zip from
    # https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-3376
    f=spring_sample.zip
    wget https://cloud.visus.uni-stuttgart.de/index.php/s/5MtTY23RWcWfgPE/download -O ${f
    unzip -q ${f}

.. literalinclude:: optical_flow.py
    :language: python
    :linenos:

.. literalinclude:: spring.py
    :language: python
    :linenos:

.. code::

    data/spring/train/
    ├── 0001
    │   ├── flow_FW_left
    │   │   ├── flow_FW_left_0001.flo5
    │   │   └── ...
    │   └── frame_left
    │       ├── frame_left_0001.png
    │       └── ...
    └── ...

    data/spring_sample/train/
    ├── 0001
    │   ├── cam_data
    │   │   ├── extrinsics.txt
    │   │   ├── focaldistance.txt
    │   │   └── intrinsics.txt
    │   ├── disp1_left|disp2_FW_left
    │   │   ├── (disp1_left|disp2_FW_left)_0001.dsp5
    │   │   └── ...
    │   ├── flow_FW_left
    │   │   ├── flow_FW_left_0001.flo5
    │   │   └── ...
    │   ├── frame_left
    │   │   ├── frame_left_0001.png
    │   │   └── ...
    │   └── maps
    │       └── detailmap_disp1_left|detailmap_disp2_FW_left|detailmap_flow_FW_left|matchmap_disp1_left|matchmap_disp2_FW_left|matchmap_flow_FW_left|rigidmap_FW_left|skymap_left
    │          ├── (detailmap_disp1_left|detailmap_disp2_FW_left|detailmap_flow_FW_left|matchmap_disp1_left|matchmap_disp2_FW_left|matchmap_flow_FW_left|rigidmap_FW_left|skymap_left)_0001.png
    │          └── ...
    └── ...
