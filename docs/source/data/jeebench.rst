JEEBench
========

https://github.com/dair-iitd/jeebench

https://huggingface.co/datasets/daman1209arora/jeebench

.. code-block:: bash

    root=data/jeebench
    hf download daman1209arora/jeebench --repo-type dataset --local-dir ${root}
    cd ${root} && unzip -q data.zip && cd ../..

.. code::

    data/jeebench
    ├── data
    │   ├── dataset.json
    │   ├── few_shot_examples.json
    │   └── responses
    │       ├── GPT4_CoT_responses/responses.json
    │       ├── GPT4_CoT+SC_responses
    │       │   ├── responses.json
    │       │   └── marks_dump
    │       └── ... 7 model runs in total
    ├── data.zip
    ├── test.json
    ├── inference.py
    ├── compute_metrics.py
    └── README.md  .gitattributes
