ChinaTextbook
=============

.. code-block:: bash

    root=data/chinatextbook
    git clone \
        https://github.com/TapXWorld/ChinaTextbook.git ${root}
    for f in ${root}/**/*.pdf.1; do p=${f%.1}; cat "$p".*(n) > "$p"; done
