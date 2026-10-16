PureDocBench
============

https://github.com/zhihengli-casia/PureDocBench

.. code-block:: bash

    root=data/puredocbench
    hf download zhihengli-casia/puredocbench --repo-type dataset --local-dir ${root}
    cat ${root}/pdb_full.tar.part-* | tar -xf - -C ${root}
    tar -xzf ${root}/gt/puredocbench_gt_latest.tar.gz -C ${root}/puredocbench-v1.0
