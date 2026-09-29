M3Exam
======

https://github.com/DAMO-NLP-SG/M3Exam

.. code-block:: bash

    root=data/m3exam
    mkdir -p ${root}
    wget https://cutt.ly/m3exam-data
    unzip -q -P 12317 m3exam-data -d ${root}

.. code::

    data/m3exam/data
    ├── multimodal-questions
    │   ├── english-questions-image.json
    │   └── images-english
    │       └── ...
    └── text-questions
        ├── english-questions-dev.json
        └── english-questions-test.json
