Irish Junior Cert Extracts
==========================

https://huggingface.co/datasets/ash12321/exampapers

.. code-block:: bash

    root=data/irish_junior_cert_extracts
    hf download ash12321/exampapers --repo-type dataset --local-dir ${root}

.. code::

    data/irish_junior_cert_extracts
    ├── JC001ALP000IV.pdf … JC223CLP000EV-4.pdf     76 PDFs, 245,641,946 B
    ├── .gitattributes
    └── extracted-json
        ├── README.md
        ├── jc    29 subject .json files
        └── lc     9 subject .json files
