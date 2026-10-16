MPDocBench
==========

.. code-block:: bash

    root=data/mpdocbench
    git clone https://github.com/Tongyi-Zhiwen/Qwen-Doc.git ${root}
    cd ${root}/MPDocBench
    base=https://www.modelscope.cn/datasets/zhoubb/MPDocBench/resolve/feb9d360f4331cf84606e48d3cfe59f159266650
    wget ${base}/MPDocBench_data.zip
    unzip -q MPDocBench_data.zip
    python tools/download_data_from_slidevqa.py
    cd ../../..
