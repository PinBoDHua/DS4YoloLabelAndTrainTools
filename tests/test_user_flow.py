"""标注序列回归测试：绘制→切图→关闭（使用独立临时数据集，绝不触碰用户数据）。"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 重定向配置到临时目录（必须在导入 app 模块前设置）
os.environ["YOLO_LABEL_CONFIG"] = str(Path(tempfile.mkdtemp()) / "config.json")

from PIL import Image

from app.utils import setup_logging
setup_logging()

from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app.theme import theme
from app.ui.main_window import MainWindow
from app.core.dataset_manager import create_dataset
from app.core.label_data import LabelType

app = QApplication(sys.argv)
theme.apply(app, "light")

# 独立临时数据集（中文路径 + 中文文件名，测试兼容性）
tmp = Path(tempfile.mkdtemp()) / "测试数据集"
ds = create_dataset(tmp, ["qq", "cat"], LabelType.DETECT)
for i in range(4):
    Image.new("RGB", (1000, 700), (80 + i * 30, 100, 160)).save(
        tmp / "images" / f"屏幕截图 测试{i}.png")

win = MainWindow()
win.resize(1440, 900)
win.show()
win._open_dataset(str(tmp), ds)

state = {"idx": 0}


def draw_rect_and_next():
    """当前图绘制一个框 → 通过 _navigate 真正切到下一张。"""
    if state["idx"] >= len(win.images):
        finish()
        return
    vp = win.canvas.viewport()
    win._set_tool("rect")
    scene_rect = win.canvas.scene.sceneRect()
    c = scene_rect.center()
    sp1 = win.canvas.mapFromScene(c.x() - scene_rect.width() / 8, c.y() - scene_rect.height() / 8)
    sp2 = win.canvas.mapFromScene(c.x() + scene_rect.width() / 8, c.y() + scene_rect.height() / 8)
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, sp1)
    for i in range(1, 8):
        t = i / 7.0
        QTest.mouseMove(vp, QPoint(int(sp1.x() + (sp2.x() - sp1.x()) * t),
                                   int(sp1.y() + (sp2.y() - sp1.y()) * t)), 30)
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, sp2)
    QApplication.processEvents()
    print(f"  已标注第 {state['idx'] + 1} 张")
    # 真正切图（修复旧版本测试未切图的逻辑错误）
    win._navigate(1)
    state["idx"] += 1
    QTimer.singleShot(200, draw_rect_and_next)


def finish():
    # 验证：每张图只有 1 个标注
    ok = True
    for i in range(4):
        img = str(tmp / "images" / f"屏幕截图 测试{i}.png")
        ann = ds.read_annotation(img)
        n = len(ann.labels)
        print(f"  第 {i + 1} 张图标注数: {n}（应为 1）")
        if n != 1:
            ok = False
    assert ok, "标注数量异常！"
    print("== 标注序列正确，关闭窗口 ==")
    win.close()
    QTimer.singleShot(2000, app.quit)


QTimer.singleShot(1000, draw_rect_and_next)
QTimer.singleShot(90000, lambda: (print("[FAIL] 超时"), app.exit(1)))
sys.exit(app.exec())
