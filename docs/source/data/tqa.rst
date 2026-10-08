TQA
===

https://allenai.org/data/tqa

https://registry.opendata.aws/allenai-tqa/

.. code-block:: bash

    root=data/tqa
    mkdir -p ${root}
    aws s3 cp --no-sign-request s3://ai2-public-datasets/tqa/tqa_train_val_test.zip ${root}
    unzip -q ${root}/tqa_train_val_test.zip -d ${root}

.. code::

    data/tqa/tqa_train_val_test
    └── {train,val,test}
        ├── {question,abc_question,teaching,textbook}_images
        │   └── {...}.png
        └── tqa_{v1_train,v1_val,v2_test}.json
