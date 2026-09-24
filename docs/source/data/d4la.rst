D4LA
====

https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/DocumentUnderstanding/VGT

.. code-block:: bash

    root=data/d4la
    modelscope download --dataset iic/D4LA --local_dir ${root}
    unzip -q ${root}/D4LA.zip -d ${root}

.. code::

    data/d4la/D4LA
    ├── train_images
    │   ├── budget_0000000867.png
    │   └── ...
    ├── test_images
    │   ├── budget_0000009238.png
    │   └── ...
    ├── json
    │   └── {train,test,map_info}.json
    └── VGT_D4LA_grid_pkl
        ├── budget_0000000867.pkl
        └── ...
