Conceptual Captions
===================

Conceptual Captions (CC3M)
--------------------------

https://ai.google.com/research/ConceptualCaptions/

https://github.com/google-research-datasets/conceptual-captions

https://huggingface.co/datasets/google-research-datasets/conceptual_captions

.. code-block:: bash

    git clone git@hf.co:datasets/google-research-datasets/conceptual_captions data/conceptual_captions
    img2dataset \
        --url_list data/conceptual_captions/labeled \
        --input_format parquet \
        --url_col image_url \
        --caption_col caption \
        --output_folder data/cc3m/labeled \
        --processes_count 4 \
        --resize_mode no \
        --skip_reencode True \
        --retries 3 \
        --disallowed_header_directives '[]'
    img2dataset \
        --url_list data/conceptual_captions/unlabeled \
        --input_format parquet \
        --url_col image_url \
        --caption_col caption \
        --output_folder data/cc3m/unlabeled \
        --processes_count 4 \
        --resize_mode no \
        --skip_reencode True \
        --retries 3 \
        --disallowed_header_directives '[]'

Conceptual 12M (CC12M)
----------------------

https://github.com/google-research-datasets/conceptual-12m

https://openxlab.org.cn/datasets/OpenDataLab/CC12M

https://huggingface.co/datasets/laion/conceptual-captions-12m-webdataset/

.. code-block:: bash

    root=data/cc12m
    GIT_LFS_SKIP_SMUDGE=1 git clone git@hf.co:datasets/laion/conceptual-captions-12m-webdataset ${root}
    cd ${root}
    for i in {00000..01100}; do echo ${i}; git lfs pull -I data/${i}.tar; done
    cd ../..
