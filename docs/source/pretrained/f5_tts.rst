F5 TTS
======

.. code-block:: bash

    root=pretrained/f5_tts
    mkdir -p ${root} && cd ${root}

    name=F5TTS_Base
    wget https://huggingface.co/SWivid/F5-TTS/resolve/main/${name}/model_1200000.pt -C ${name}.pth
    wget https://huggingface.co/SWivid/F5-TTS/resolve/main/${name}/vocab.txt -C ${name}.txt

    cd ../..
