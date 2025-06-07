ODinW
=====

https://eval.ai/web/challenges/challenge-page/1839/overview

https://github.com/microsoft/GLIP/tree/main/odinw

.. code-block:: bash

    root=data/odinw
    mkdir -p ${root} && cd ${root}

    url=https://huggingface.co
    curl -s ${url}/api/models/GLIPModel/GLIP/tree/main/odinw_35 |
        jq -r '.[] | select(.type == "file") | .path' |
        sed 's#^odinw_35/##' |
        xargs -I{} -P4 wget ${url}/GLIPModel/GLIP/resolve/main/odinw_35/{}
    for f in *.zip; do unzip ${f}; done

    cd ../..
