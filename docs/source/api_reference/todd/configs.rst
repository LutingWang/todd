Configs
=======

使用标准库 ``json.dump`` 或 ``json.dumps`` 保存包含 ``Config`` 的结构时，
请传入 ``cls=todd.patches.PythonJSONEncoder``，使本次调用使用 Python 编码器，
避免 C 编码器将配置编码为空字典。该方式不修改进程全局状态，可用于多线程调用。

.. automodule:: todd.configs
