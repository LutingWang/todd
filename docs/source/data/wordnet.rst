WordNet
=======

https://wordnet.princeton.edu/

.. code-block:: bash

    root=data/wordnet
    mkdir -p ${root}
    python -c "import nltk; nltk.download('wordnet', download_dir='${root}')"
