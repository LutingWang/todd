RVL-CDIP
========

https://www.cs.cmu.edu/~aharley/rvl-cdip/

.. code-block:: bash

    root=data/rvl-cdip
    hf download aharley/rvl_cdip --repo-type dataset --local-dir ${root}
    tar -zxf ${root}/data/rvl-cdip.tar.gz -C ${root}/data

.. code::

    data/rvl-cdip/data
    ├── images
    │   ├── imagesb/b/c/c/bcc60f00/0012181633.tif
    │   └── ...
    └── {train,val,test}.txt
