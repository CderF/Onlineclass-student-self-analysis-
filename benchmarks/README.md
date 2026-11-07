# benchmarks：计算量评估

测量本系统除模型推理之外的计算开销：每帧耗时、CPU 占用和内存。需求见 [proposal.md](proposal.md)，术语「测量口径」「帧间隔底线」见根目录的 `GLOSSARY.md`。

所有脚本都在仓库根目录、用 `demo` 环境运行。`app/` 下的代码一行都不改。

## 文件

| 文件 | 作用 |
|---|---|
| `record_video.py` | 用摄像头录约 5 分钟的测试视频，按提示依次做出各种状态 |
| `build_cache.py` | 对视频跑一遍 MediaPipe 和 YOLO，逐帧缓存关键点与表情概率 |
| `bench_realtime.py` | 上课时持续发生的开销：每帧处理链路、课程记录（SQLite）、界面线程 |
| `bench_report.py` | 课后报告页的一次性开销：读一节课、问题时段、文字反馈、汇总、CSV 导出、页面显示 |
| `summarize.py` | 把一台机器的结果汇总成 `results/<机器>/summary.md` |
| `common.py` | 公共部分：路径、机器信息、资源采样、分步计时、统计 |
| `data/` | 测试视频和缓存。**含人脸，已加入 `.gitignore`，不提交** |
| `results/<机器>/` | 测量数据（JSON、逐帧 CSV.gz、资源时间线 CSV）和 `summary.md`，提交到 git |

## 流程

### 在 Mac 上

```bash
python benchmarks/record_video.py
```

在有摄像头权限的终端里运行。按屏幕和终端的提示依次做：正常、低头、闭眼、打哈欠、低头后脸离开画面、离席。

```bash
python benchmarks/build_cache.py
```

需要 `weights/best.pt`。

```bash
python benchmarks/bench_realtime.py --scope exclude --pace realtime
```

按视频帧率送帧，约 5 分钟 × 5 次。

```bash
python benchmarks/bench_realtime.py --scope exclude --pace max
```

尽可能快地送帧。

```bash
python benchmarks/bench_report.py
```

```bash
python benchmarks/summarize.py
```

正式测量时会弹出真实窗口（监控页或报告页），测量期间不要操作电脑，也不要让显示器休眠，可以用 `caffeinate -d` 运行。

### 在 Windows 低配机上

1. 按 `requirements.txt` 准备 Python 3.10 环境，并放好 `weights/best.pt`。
2. 把 Mac 上的 `benchmarks/data/` 整个复制过去，包括视频和缓存，用 U 盘等方式，**不要经过 git**。两台机器用同一份输入，结果才能直接比较。
3. 从上面的第 3 步（`bench_realtime.py`）开始执行。结果会写到 `results/Windows-<CPU 型号>/`。

### 「都包含」口径

正式实测要等 #26 的 4 类模型就绪。在那之前只能试跑，结果不保存：

```bash
python benchmarks/bench_realtime.py --scope include --smoke 60
```

模型就绪后，加 `--final-model` 正式测量，送帧节奏同样测两种：

```bash
python benchmarks/bench_realtime.py --scope include --pace realtime --final-model
```

```bash
python benchmarks/bench_realtime.py --scope include --pace max --final-model
```

### 常用选项

| 选项 | 说明 |
|---|---|
| `--runs N` | 测量次数，默认 5，每次都在新进程中进行 |
| `--machine 名称` | 结果子目录名，默认按 CPU 型号生成 |
| `--no-mesh` | 关闭面部网格 |
| `--warmup 秒` | 开头不计入的时长，默认 5，包含 3 秒校准 |
| `--offscreen` | 不显示窗口，只用于调试。界面绘制方式不同，正式测量不要用 |

## 测量方法

### 实时链路：`bench_realtime.py`

- **运行的是真实的程序**：`MonitorInterface` 和 `CameraThread` 都真实运行，信号经过真实的跨线程队列到达界面线程，界面真实刷新。只替换以下几处：
  - **摄像头**：换成回放视频的 `ReplayCapture`。
  - **「都排除」口径下的 MediaPipe**：不构建 FaceMesh 的计算图，`face_mesh.process` 直接返回缓存的关键点。
  - **「都排除」口径下的表情分类器**：换成读取缓存的概率，不导入 ultralytics 和 torch（结果里有「已导入 torch」一项可以核对）。
  
  关键点之后的步骤全部是真实代码：几何计算、画网格、注意力规则、课程记录、界面刷新。
- **分步计时**：给这些函数套上计时，记录墙上时间和本线程的 CPU 时间：
  - `face_mesh.process`、`mp_drawing.draw_landmarks`、`estimate_head_pose`、`_calculate_ear` / `_calculate_mar`、`process_frame`；
  - `ExpressionClassifier.process_face`、`AttentionAnalyzer.process_frame`、`SessionRecorder.add_frame`；
  - 监控页的三个槽函数。
  
  以后这些名字改了，脚本会直接报 `AttributeError`，不会悄悄测错。
- **两种送帧节奏**：
  - `realtime` 按视频帧率送帧。处理跟不上时跳到当前时刻对应的帧，计为丢帧，相当于一个没有缓冲的摄像头。
  - `max` 不等待、连续送帧。这时把 `time.time()` 换成视频时间（帧号 ÷ 帧率），校准、3 秒 / 8 秒 / 60 秒窗口和每秒记录里的帧数才与实际使用时一致；否则窗口里的帧数会成倍增加，规则的单帧开销会被放大。
- **测量手段本身的开销会扣除**：
  - 读帧解码：强制单线程解码，开销全部落在采集线程上；
  - 「都排除」口径下的缓存回放。
  
  这两项单独计时，再从每帧处理时间和进程 CPU 中扣掉。

### 报告页：`bench_report.py`

- 在临时数据库里用真实的 `SessionRecorder` 写入 5 节同课程名、各 90 分钟的合成课程。测量对象是最后一节，三类问题时段都会出现，文字反馈里也有「前回との比較」。
- 纯 Python 的操作测本线程 CPU。页面相关的两项把 WebEngine 子进程（QtWebEngineProcess）也计入 CPU 和内存。

## 指标定义

| 指标 | 定义 |
|---|---|
| 每帧处理合计（不含模型推理） | 采集线程从 `read()` 返回到下一次调用 `read()` 的时间，减去模型推理（或回放）的时间。不含等待下一帧和解码。判定理想线用的就是这个值 |
| 帧间隔 | 相邻两帧 `read()` 返回的时间差，按视频帧率送帧时才判定底线（< 250 ms） |
| 进程 CPU（扣除测量开销） | psutil 统计的进程 CPU 时间减去解码和回放的 CPU 时间，再除以墙上时间，以单核 = 100% 表示。包含 OpenCV 内部线程、采样线程等进程内的全部线程 |
| 采集线程 / 界面线程 CPU | 各线程自身的 CPU 时间除以墙上时间。界面线程包括槽函数和窗口绘制 |
| 内存 | 进程常驻内存（RSS）。「开始前」是窗口已显示、还没点「開始」时的值 |

## 已知局限

- **测量代码复刻了程序内部结构**：计时依赖上面列出的函数名，实时链路的步骤划分对应 `CameraThread.run` 当前的处理顺序。以后 `run` 里的步骤或顺序变了，`frame_row` 的派生项（如「采集循环其余」）也要跟着改。
- **视频解码不等于摄像头取帧**：解码开销已经扣除，但真实摄像头由系统框架取帧，它自己的开销没有测到。
- **Windows 上单步 CPU 时间不可靠**：`time.thread_time()` 的分辨率约 15.6 ms，每帧各步骤的 CPU 列没有意义，只看墙上时间和整次测量的 CPU 占用。
- **回放和实时推理的结果略有差异**：「都排除」口径下，按真实帧率送帧时如果丢帧，人脸框平滑后的位置会和生成缓存时略有不同，回放的表情概率就和当场推理的不完全一样。这只影响下游拿到的数据，不影响计算量。
- **内存的口径**：「都排除」口径下，内存仍包含 Python、Qt、OpenCV 以及 mediapipe 包本身（不含 FaceMesh 计算图和模型权重）。
