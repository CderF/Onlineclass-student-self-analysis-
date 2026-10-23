# 决策记录：前端技术栈（交互图表）

## 2026-10-23　报告页交互图表采用「Qt + ECharts」混合方案

**状态**：已采纳

### 背景

- 当前前端：PyQt5 5.15.11 + PyQt-Fluent-Widgets 1.11.1（`qfluentwidgets`，Fluent 风格组件）。界面代码约 400 行，分布在 3 个文件中。
- 报告页 `ReportInterface` 仍是占位：图表区域为空，`load_session_data` / `render_charts` 未实现。
- 主要担忧：图表难做，且未来可能需要交互图表。

### 约束

| 项 | 结论 |
|---|---|
| 用途 | 仅用于论文和实验，不分发、不商用 |
| 平台 | 桌面端，全部在本地运行，视频不离开本机 |
| 网页版 / 手机版 | 暂无计划 |
| 开发者经验 | 有 HTML / CSS / JS 开发经验 |

### 决策

1. **保持** PyQt5 + qfluentwidgets 作为主界面框架。实时视频链路（`CameraThread` → `pyqtSignal` → `QLabel`）不改动。
2. **报告页**用 `QWebEngineView` 加载本地 HTML 页面，图表由 **ECharts**（Apache-2.0，Apache 基金会维护）绘制。Python 端把会话数据序列化为 JSON 传给页面，方式为 `runJavaScript` 或 `QWebChannel`。
3. ECharts 的 JS 文件随项目一起打包，从本地加载，**不走 CDN**，以保证离线可用、数据不外发。

### 理由

起决定作用的理由：需求是交互图表，报告页还规划了状态分布等多种图形。ECharts 自带悬停提示、图例开关、时间轴缩放、图表联动、饼图和事件标注；开发者有网页经验，可以直接上手；实时视频链路完全不受影响。

### 未采纳的方案

| 方案 | 未采纳原因 |
|---|---|
| pyqtgraph（纯 Python 画图库） | 悬停提示、十字准线需要自己写，也没有饼图。更适合实时刷新的曲线；以后监控页需要实时曲线时可再考虑，可与本方案并存 |
| 迁移到 PySide6 / Qt 6 | 主要收益是许可证（LGPL）和 Qt 官方支持。仅用于论文时，这些收益不抵迁移成本 |
| 整个界面改成网页（pywebview 等） | 需要把摄像头画面编码成流传给网页，延迟和复杂度都会增加；没有网页版计划，前端代码也就谈不上复用 |
| Gradio / Streamlit | 适合一次性演示，不适合长时间实时监控加定时弹窗 |

### 后果与注意事项

- **依赖版本必须对齐**：需要安装 `PyQtWebEngine`，其中 `PyQtWebEngine-Qt5` 要锁定为与 `PyQt5-Qt5` 相同的版本（当前为 5.15.17）。PyPI 上最新的是 5.15.19，不锁定会导致版本不一致，程序无法运行。
- **体积**：WebEngine 自带 Chromium 内核，环境和打包体积会明显变大。
- **调试**：图表逻辑在 JS 里，需要配合 WebEngine 的开发者工具调试（设置 `QTWEBENGINE_REMOTE_DEBUGGING` 环境变量）。
- **前置工作**：不管用哪个图表库，都要先实现会话数据的保存，以及从监控页传到报告页的通道。
- **已知遗留**：Qt 5.15 的官方长期支持已于 2025-05-26 结束；PyQt5 和 qfluentwidgets 免费版都是 GPLv3，后者还标明仅限非商业使用。如果以后要分发或商用，需要重新评估迁移到 PySide6 并解决授权问题。

### 何时需要重新评估

- 计划分发给研究室以外的人，或商用、闭源发布。
- 需要网页版或手机版。
- WebEngine 在目标机器上安装或运行有困难。此时退回到 pyqtgraph。

### 参考

- [PyQt-Fluent-Widgets（许可证说明）](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)
- [Riverbank：PyQt 许可证 FAQ](https://www.riverbankcomputing.com/commercial/license-faq)
- [Qt for Python：Commercial Use](https://doc.qt.io/qtforpython-6/commercial/index.html)
- [Qt 博客：Qt 5.15 Extended Security Maintenance](https://www.qt.io/blog/extended-security-maintenance-for-qt-5.15-begins-may-2025)
- [pyqtgraph（GitHub）](https://github.com/pyqtgraph/pyqtgraph)
- [Apache ECharts](https://echarts.apache.org/)
