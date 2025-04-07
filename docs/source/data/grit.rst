GRIT
====

https://github.com/microsoft/unilm/tree/master/kosmos-2#grit-large-scale-training-corpus-of-grounded-image-text-pairs

https://huggingface.co/datasets/zzliang/GRIT

.. code-block:: bash

    git clone git@hf.co:datasets/zzliang/GRIT data/GRIT
    img2dataset \
        --url_list data/GRIT/grit-20m \
        --input_format parquet \
        --caption_col caption \
        --output_folder data/grit \
        --save_additional_columns '["id","noun_chunks","ref_exps","clip_similarity_vitb32","clip_similarity_vitl14"]' \
        --processes_count 1 \
        --resize_mode keep_ratio \
        --resize_only_if_bigger True \
        --skip_reencode True \
        --retries 3 \
        --disallowed_header_directives '[]'
