AI2D
====

https://prior.allenai.org/projects/diagram-understanding

https://registry.opendata.aws/allenai-diagrams/

.. code-block:: bash

    root=data/ai2d
    hf download LutingWang/ai2d --repo-type dataset --local-dir ${root}
    ln -s ai2d/ai2d-all.zip data/
    unzip -q data/ai2d-all.zip -d data

.. code::

    data/ai2d/
        ├── annotations/
        │   └── {0..4907}.png.json
        ├── images/
        │   └── {0..4907}.png
        ├── questions/
        │   └── {0..4907}.png.json
        └── categories.json
