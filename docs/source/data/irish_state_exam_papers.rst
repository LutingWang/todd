Irish State Exam Papers
=======================

https://huggingface.co/datasets/ash12321/exam-papers

.. code-block:: bash

    root=data/irish_state_exam_papers
    mkdir -p ${root} && cd ${root}
    base=https://huggingface.co/datasets/ash12321/exam-papers/resolve/main

    # metadata.json maps year + clean_filename -> folder; use it to select a
    # slice, because the /tree API truncates each folder at 1,000 entries
    curl -L -o metadata.json ${base}/metadata.json

.. code::

    data/irish_state_exam_papers
    ├── metadata.json
    ├── junior_cert
    │   ├── exam_papers       84 PDFs, 2024 & 2025
    │   └── marking_schemes   88 PDFs
    └── leaving_cert
        ├── exam_papers      377 PDFs
        └── marking_schemes  366 PDFs
