Configs
=======

使用标准库 ``json.dump`` 或 ``json.dumps`` 保存包含 ``Config`` 的结构时，
请在 ``with todd.patches.python_json():`` 内调用，临时关闭可能将配置编码为空字典的
C 编码器（进程级生效，退出时恢复）。

.. automodule:: todd.configs
