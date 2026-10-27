# test-scripts

专项验证脚本：覆盖 `tests/`（pytest）不便模拟的情况，如真实进程崩溃、并发读写、长时间运行、性能，以及需要真实摄像头或人工观察的场景。

- 在仓库根目录用 `demo` 环境运行：`python test-scripts/<脚本名>.py`
- 退出码 0 表示全部通过；每个检查项打印 `PASS` / `FAIL`
- 临时数据写在系统临时目录，不碰 `data/`

| 脚本 | 对应 Issue | 内容 |
|---|---|---|
| `verify_issue3_storage.py` | #3 | 真实分析器场景下的每秒汇总重算、SIGKILL 崩溃恢复、并发读写、90 分钟性能与体积 |
