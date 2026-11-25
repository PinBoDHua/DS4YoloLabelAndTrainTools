"""GUI 冒烟测试：启动主窗口、切换主题、打开数据集，分阶段截图后自动退出。"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 重定向配置到临时目录：theme.apply / 主窗口状态都会写 config.json
os.environ["YOLO_LABEL_CONFIG"] = str(Path(tempfile.mkdtemp()) / "config.json")

from PIL import Image

from app.utils import setup_logging
setup_logging()

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from app.config import config
from app.theme import theme
from app.ui.main_window import MainWindow
from app.core.dataset_manager import create_dataset
from app.core.label_data import LabelType

OUT_DIR = Path(__file__).parent / "_shots"
OUT_DIR.mkdir(exist_ok=True)

app = QApplication(sys.argv)
theme.apply(app, "light")

win = MainWindow()
win.show()

tmp = Path(tempfile.mkdtemp())
ds_root = tmp / "gui_test_dataset"
ds = create_dataset(ds_root, ["cat", "dog", "person"], LabelType.DETECT)
for i in range(4):
    Image.new("RGB", (800, 600), (200 - i * 30, 120, 180)).save(ds_root / "images" / f"img{i}.jpg")


def grab(name: str):
    pm = win.grab()
    pm.save(str(OUT_DIR / name))
    print(f"截图已保存: {OUT_DIR / name}")


def stage1():  # 浅色主题 + 主窗口
    grab("01_light_main.png")


def stage2():  # 打开数据集
    win._open_dataset(str(ds_root), ds)
    grab("02_light_dataset.png")


def stage3():  # 绘制标注并切换深色
    # 用 RectTool 程序化添加一个框
    from app.ui import tools as tools_mod
    win._set_tool("rect")
    # 模拟画一个框：直接添加 LabelItem
    from app.core.label_data import LabelItem
    item = win.canvas.add_shape_item(LabelItem(LabelType.DETECT, 0, points=[[100, 100], [300, 300]]))
    item.setSelected(True)
    grab("03_light_label.png")
    theme.apply(app, "dark")
    QTimer.singleShot(400, stage4)


def stage4():  # 深色主题
    grab("04_dark_dataset.png")


def finish():
    print("GUI 冒烟测试完成")
    app.quit()


QTimer.singleShot(800, stage1)
QTimer.singleShot(1400, stage2)
QTimer.singleShot(2000, stage3)
QTimer.singleShot(2600, stage4)
QTimer.singleShot(3000, finish)
sys.exit(app.exec())
