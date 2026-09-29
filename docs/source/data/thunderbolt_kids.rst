Thunderbolt Kids
================

http://www.thunderboltkids.co.za/

.. code-block:: bash

    root=data/thunderbolt-kids
    mkdir -p ${root}
    base=http://www.thunderboltkids.co.za/downloads
    for f in Grade{4,5,6}_{A,B}_English_12-11-2012_smaller.pdf \
             Gr_{4,5,6}_{A,B}_Eng_TG_smaller.pdf \
             ScienceAdventuresGrade{4,5,6}_smaller.pdf; do
        curl -L -C - -o ${root}/${f} ${base}/${f}
    done

.. code::

    data/thunderbolt-kids
    ├── download.sh
    ├── download-loop.sh
    ├── Grade{4,5,6}_{A,B}_English_12-11-2012_smaller.pdf
    ├── Gr_{4,5,6}_{A,B}_Eng_TG_smaller.pdf
    └── ScienceAdventuresGrade{4,5,6}_smaller.pdf
