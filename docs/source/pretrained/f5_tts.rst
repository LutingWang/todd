F5 TTS
======

.. code-block:: bash

    root=pretrained/f5_tts
    mkdir -p ${root} && cd ${root}

    name=F5TTS_Base
    wget https://huggingface.co/SWivid/F5-TTS/resolve/main/${name}/model_1200000.pt -C ${name}.pth
    wget https://huggingface.co/SWivid/F5-TTS/resolve/main/${name}/vocab.txt -C ${name}.txt

    cd ../..

:download:`f5_tts.py <f5_tts.py>` can be used to access the pretrained models.
The following command will generate a speech file ``f5_tts.wav`` from the text
file :download:`lines.txt <f5_tts/lines.txt>` using the voice file
:download:`voices.py <f5_tts/voices.py>`.

.. code-block:: bash

    python f5_tts.py voices.py lines.txt f5_tts.wav
