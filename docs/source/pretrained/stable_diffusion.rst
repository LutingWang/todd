Stable Diffusion
================

.. code-block:: bash

    root=pretrained/stable_diffusion
    mkdir -p ${root}

    base_url=git@hf.co:stabilityai
    # huggingface-cli login

Stable Diffusion 3
------------------

.. code-block:: bash

    cd ${root}
    repo=stable-diffusion-3-medium
    GIT_LFS_SKIP_SMUDGE=1 git clone ${base_url}/${repo}
    git -C ${repo} lfs pull -I sd3_medium_incl_clips_t5xxlfp8.safetensors
    cd ../..

.. code::

    pretrained/stable_diffusion/
    └── stable-diffusion-3-medium
