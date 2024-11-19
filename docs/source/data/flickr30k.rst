Flickr
======

Flickr 30k
----------

https://shannon.cs.illinois.edu/DenotationGraph/

https://shannon.cs.illinois.edu/DenotationGraph/data/index.html

.. code-block:: bash

    root=data/flickr-30k
    mkdir -p ${root} && cd ${root}
    f=flickr30k-images.tar.gz
    # download ${f} from https://uofi.box.com/s/1cpolrtkckn4hxr1zhmfg0ln9veo6jpl
    tar -zxf ${f}
    mv flickr30k-images images
    cd ../..

Flickr 30k Entities
-------------------

https://bryanplummer.com/Flickr30kEntities/

.. code-block:: bash

    root=data/flickr-30k-entities
    mkdir -p ${root} && cd ${root}
    f=annotations.zip
    wget https://github.com/BryanPlummer/flickr30k_entities/raw/refs/heads/master/${f}
    unzip -q ${f}
    cd ../..
