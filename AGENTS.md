# AGENTS.md

This file provides guidance to Qoder (qoder.com) when working with code in this repository.

## 项目概述

Windows 桌面端 YOLO 标注训练一体化工具（PySide6 + ultralytics）：支持检测/分割/关键点/OBB 四种标注、数据集管理与划分、模型预标注与 SAM 智能分割、训练（实时损失曲线）、导出与推理。面向中文路径数据集用户，Windows 10/11、Python 3.11+。

## 环境与命令

所有命令在项目根目录执行，使用项目内虚拟环境 `.venv`（由 `setup.bat` 创建，包含 CUDA 版 PyTorch，依赖见 `requirements.txt`）。

```powershell
# 安装/重建环境（仅一次；检测 Python → 建 .venv → 装 CUDA 版 torch + 其余依赖）
.\setup.bat

# 启动应用（stderr 追加到 logs\stderr.log）
.\start.bat
.\.venv\Scripts\python.exe -m app.main

# 运行测试 —— 全部是独立脚本直接执行，非 pytest
.\.venv\Scripts\python.exe tests\test_core.py        # 核心逻辑：四种 YOLO txt 读写 + 数据集创建/划分/统计
.\.venv\Scripts\python.exe tests\test_gui.py         # GUI 冒烟：主窗口/主题/数据集，分阶段截图到 tests\_shots
.\.venv\Scripts\python.exe tests\test_ai_smoke.py    # SAM 加载 + 点提示分割（首次需下载 mobile_sam.pt）
.\.venv\Scripts\python.exe tests\test_train_smoke.py # 完整训练链路：建集 → 划分 → 训练 1 epoch → 验证权重
.\.venv\Scripts\python.exe tests\test_chinese_train.py # 中文路径下训练验证
.\.venv\Scripts\python.exe tests\test_user_flow.py   # 复现用户闪退流程（依赖本地数据集 D:\测试新项目）
.\.venv\Scripts\python.exe tests\test_thread_stress.py # 缩略图线程压测（依赖 D:\测试新项目）
```

无 linter/格式化配置。运行日志在 `logs/app.log`（滚动，5MB×3），Qt 内部消息、未捕获异常也写入其中；调试闪退时先查此文件及 `logs/stderr.log`。

## 架构

分层：`app/ui`（PySide6 视图）→ `app/core`（业务逻辑，可直接用 QThread 但保持可测试）→ `app/utils`、`app/config.py`、`app/theme.py`（基础设施）。

### 入口与基础设施

- `app/main.py` — 入口，按此顺序初始化：`setup_logging()` → `install_excepthook()` → `patch_opencv_imread()`。还启用 `faulthandler` 并安装 Qt 消息处理器（qWarning/qFatal → `logs/app.log`），是排查 C++ 层崩溃的关键手段。新增任何图像读取代码前必须先确保 `patch_opencv_imread` 已生效（子线程内需再次调用，见 `prelabel_engine.py`）。
- `app/config.py` — 全局单例 `config`（模块导入即加载）。读写项目根目录 `config.json`，所有 `set()` 立即落盘；新增配置项需同时加入 `DEFAULTS` 字典。
- `app/utils/__init__.py` — `patch_opencv_imread`（cv2.imread 中文路径失败时回退 PIL，ultralytics 内部同样受益）、`download_with_fallback`（GitHub 直连失败自动切换加速镜像、断点续传）、日志与异常钩子。
- `app/theme.py` — 全局单例 `theme`，浅色/深色两套 `ThemeColors` 色板（QSS 与画布/图表自绘共用），运行时热切换，画布类通过 `theme.theme_changed` 信号刷新自绘颜色。

### 核心数据流

1. `app/core/label_data.py` — 数据模型：`LabelType`（DETECT/SEGMENT/POSE/OBB）、`LabelItem`（`points` 为**绝对像素坐标**；POSE 另有 `keypoints` 含可见性 0/1/2）、`ImageAnnotation`、`DatasetMeta`（`dataset.json`：classes + task）。
2. `app/core/label_io.py` — YOLO txt 读写（磁盘为归一化坐标）。读入时按 token 数自动识别格式；OBB 与 4 顶点分割多边形格式完全相同（8 个坐标），必须用数据集声明的 `prefer_type` 消歧。图片尺寸用 PIL 读取（无中文路径问题）。
3. `app/core/dataset_manager.py` — `Dataset` 封装目录：`dataset.json` / `images/` / `labels/`，划分后生成 `images/{train,val,test}` 与对应 labels，`write_data_yaml()` 生成训练配置。
4. `app/core/train_engine.py` / `prelabel_engine.py` — `TrainWorker` / `PrelabelWorker`（QThread），封装 ultralytics；训练通过 ultralytics 回调上报每 epoch 指标（注意 ultralytics 8.4+ 回调注册方式已变更），停止靠 `stop_requested` 标志在 epoch 边界中断。
5. `app/core/sam_engine.py` — `SamEngine` 懒加载 mobile_sam.pt 到 `models/`，点/框提示 → 掩码 → 多边形。
6. `app/core/format_convert.py` — VOC XML / COCO JSON ↔ YOLO 转换、标注可视化导出。

### UI 结构

- `app/ui/main_window.py` — `MainWindow` 持有全部状态：当前 `Dataset`、图片列表、`current_ann`、撤销/重做快照栈（JSON）。中央为 `main_tabs`（QTabWidget，三页：标注画布/训练面板/推理测试）。左侧 `ClassPanel`、右侧 `ImagePanel` 均为 QDockWidget。工具与快捷键定义在模块顶部的 `TOOL_INFO`/`TOOL_KEYS`。
- `app/ui/canvas.py` — `LabelCanvas`（QGraphicsView）+ `ShapeItem`（每个标注一个 item，几何直接读写 `label.points` 绝对像素）。工具交互（绘制/编辑/旋转手柄）在 `app/ui/tools.py`。
- 对话框：`dataset_dialog`、`prelabel_dialog`、`sam_dialog`、`settings_dialog`。`train_panel`（训练面板，含 pyqtgraph 实时曲线）与 `test_panel`（推理测试）不是独立窗口，而是 `main_tabs` 的内嵌页；训练面板通过 `set_dataset()` 随主窗口数据集切换绑定。
- `app/ui/image_panel.py` — 缩略图列表，`_ThumbLoader`（QThread）后台加载。

## 关键约定（违反会直接导致崩溃或功能失效）

1. **PySide6 6.11 枚举**：禁止通过实例访问短名枚举（`dlg.Accepted`），必须用完整作用域（`QDialog.DialogCode.Accepted`）。
2. **Qt 几何运算符**：PySide6 6.11 中 QPointF 等运算符绑定有 bug，禁止使用 `a - b`、`a * f` 等直接运算，必须分量运算（`canvas.py` 顶部已有 `_sub`/`_mul` 辅助函数）。
3. **线程安全**：禁止在后台线程创建 QPixmap（缩略图线程只产数据，GUI 线程转换）；QThread 对象必须被持有引用防止被 GC。程序退出时线程仍运行会触发 C++ 层 abort（0xc0000409），须遵循三层清理：`MainWindow._on_app_about_to_quit`（`aboutToQuit` 全局兜底，覆盖所有退出路径）→ 各面板/对话框关闭时显式停止线程 → 线程替换时安全取消并等待。新增后台任务必须接入这三层。
4. **中文路径**：图像读取一律经 `patch_opencv_imread` 或直接用 PIL；不要依赖 `np.fromfile` 读中文路径文件。训练/预标注/推理都依赖此补丁。
5. **requirements.txt**：不要写中文注释（pip 旧版解析失败）；torch/torchvision 不在其中，由 `setup.bat` 从 PyTorch CUDA 索引单独安装。
6. 用户数据集目录结构由 `Dataset` 类维护（`dataset.json`/`images`/`labels`），不要假设图片名不含中文或路径无空格。
