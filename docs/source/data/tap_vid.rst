TAP-Vid
=======

https://github.com/google-deepmind/tapnet/tree/main/data

TAP-Vid-DAVIS
-------------

.. code-block:: bash

    root=data/tap_vid
    mkdir -p ${root} && cd ${root}
    f=tapvid_davis.zip
    wget https://storage.googleapis.com/dm-tapnet/${f}
    unzip -p ${f} tapvid_davis/tapvid_davis.pkl > davis.pkl
    cd ../..

.. literalinclude:: tap_vid_davis.py
    :language: python
    :linenos:

.. code::

    data/tap_vid/
    └── davis.pkl
