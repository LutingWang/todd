Vocos
=====

.. code-block:: bash

    root=pretrained/vocos
    mkdir -p ${root} && cd ${root}

    name=vocos-mel-24khz
    wget https://huggingface.co/charactr/${name}/resolve/main/pytorch_model.bin -C ${name}.pth

    cd ../..
