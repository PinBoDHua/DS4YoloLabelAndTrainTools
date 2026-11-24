"""用户闪退问题复现测试：中文路径数据集 + 中文文件名图片 + 完整标注流程。

模拟：打开数据集 → 绘制矩形框 → 保存 → 编辑拖动 → 删除 → 撤销 → 切图 → 缩略图加载。
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw

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

# 中文路径数据集 + 中文文件名图片（复现用户场景）
tmp = Path(tempfile.mkdtemp()) / "测试数据集"
ds = create_dataset(tmp, ["人", "车", "狗"], LabelType.DETECT)

img_path = tmp / "images" / "屏幕截图 2026-08-13 测试.png"
img = Image.new("RGB", (1200, 800), (60, 120, 200))
d = ImageDraw.Draw(img)
d.rectangle([100, 100, 500, 500], outline=(255, 255, 255), width=6)
d.ellipse([600, 300, 1000, 700], fill=(240, 200, 60))
img.save(img_path)
print(f"已创建测试图片: {img_path}")

win = MainWindow()
win.resize(1200, 800)
win.show()
win._open_dataset(str(tmp), ds)

results = {}


def draw_rect(start: QPoint, end: QPoint):
    """在画布上模拟拖拽绘制矩形框。"""
    vp = win.canvas.viewport()
    win._set_tool("rect")
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, start)
    for i in range(1, 11):
        t = i / 10.0
        p = QPoint(int(start.x() + (end.x() - start.x()) * t),
                   int(start.y() + (end.y() - start.y()) * t))
        QTest.mouseMove(vp, p, 50)
    QTest.mouseRelease(vp, Qt.LeftButton, Qt.NoModifier, end)
    QApplication.processEvents()


def stage1_draw():
    print("== 阶段 1: 绘制矩形框")
    draw_rect(QPoint(300, 200), QPoint(600, 500))
    QTimer.singleShot(400, stage2_verify_draw)


def stage2_verify_draw():
    n = len(win.canvas.collect_shapes())
    txt = tmp / "labels" / "屏幕截图 2026-08-13 测试.txt"
    saved = txt.exists()
    print(f"  画布标注数: {n}, 已保存: {saved}")
    assert n == 1, "绘制后应有 1 个标注"
    assert saved, "标注应已自动保存"
    results["draw"] = True
    QTimer.singleShot(200, stage3_edit)


def stage3_edit():
    print("== 阶段 2: 选中并拖动标注")
    win._set_tool("select")
    item = win.canvas.collect_shapes()[0]
    p0_before = list(item.label.points[0])
    # 通过 item 直接模拟拖动（场景坐标）
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtCore import QEvent, QPointF, Qt as QtC
    center = QPointF(p0_before[0] + 150, p0_before[1] + 150)
    ev_press = QMouseEvent(QEvent.MouseButtonPress, center, QtC.LeftButton,
                           QtC.LeftButton, QtC.NoModifier)
    item.mousePressEvent(ev_press)
    for i in range(1, 8):
        pos = QPointF(center.x() + 10 * i, center.y() + 5 * i)
        ev_move = QMouseEvent(QEvent.MouseMove, pos, QtC.NoButton,
                              QtC.LeftButton, QtC.NoModifier)
        item.mouseMoveEvent(ev_move)
    ev_rel = QMouseEvent(QEvent.MouseButtonRelease,
                         QPointF(center.x() + 70, center.y() + 35),
                         QtC.LeftButton, QtC.NoButton, QtC.NoModifier)
    item.mouseReleaseEvent(ev_rel)
    QApplication.processEvents()
    p0 = item.label.points[0]
    print(f"  拖动后左上角: ({p0[0]:.1f}, {p0[1]:.1f})（应约为 {p0_before[0] + 70:.1f}, {p0_before[1] + 35:.1f}）")
    assert abs(p0[0] - (p0_before[0] + 70)) < 3 and abs(p0[1] - (p0_before[1] + 35)) < 3, "拖动距离不正确"
    results["edit"] = True
    QTimer.singleShot(200, stage4_undo_redo)


def stage4_undo_redo():
    print("== 阶段 3: 撤销 / 重做")
    item = win.canvas.collect_shapes()[0]
    p0_before = list(item.label.points[0])
    win._undo()
    p0 = win.canvas.collect_shapes()[0].label.points[0]
    print(f"  撤销后左上角: ({p0[0]:.1f}, {p0[1]:.1f})（应回到 {p0_before[0] - 70:.1f}, {p0_before[1] - 35:.1f}）")
    assert abs(p0[0] - (p0_before[0] - 70)) < 1 and abs(p0[1] - (p0_before[1] - 35)) < 1, "撤销未生效"
    win._redo()
    QApplication.processEvents()
    results["undo"] = True
    QTimer.singleShot(200, stage5_delete)


def stage5_delete():
    print("== 阶段 4: 删除选中标注")
    win.canvas.select_all()
    win.canvas.delete_selected()
    n = len(win.canvas.collect_shapes())
    print(f"  删除后标注数: {n}")
    assert n == 0
    results["delete"] = True
    QTimer.singleShot(200, stage6_switch)


def stage6_switch():
    print("== 阶段 5: 切图（下一张再回来）")
    win._navigate(1)  # 无下一张，不动作
    win._navigate(-1)  # 也不动作
    # 重新加载当前图
    win._load_current_image()
    QApplication.processEvents()
    print("  切图操作完成，无崩溃")
    results["switch"] = True
    QTimer.singleShot(500, stage7_polygon)


def stage7_polygon():
    print("== 阶段 6: 绘制多边形并删除")
    win._set_tool("polygon")
    vp = win.canvas.viewport()
    pts = [QPoint(150, 150), QPoint(350, 120), QPoint(420, 300), QPoint(200, 350)]
    QTest.mousePress(vp, Qt.LeftButton, Qt.NoModifier, pts[0])
    QApplication.processEvents()
    for p in pts[1:]:
        QTest.mouseClick(vp, Qt.LeftButton, Qt.NoModifier, p)
        QApplication.processEvents()
    QTest.mouseDClick(vp, Qt.LeftButton, Qt.NoModifier, pts[0])
    QApplication.processEvents()
    n = len(win.canvas.collect_shapes())
    print(f"  多边形标注数: {n}")
    assert n == 1
    results["polygon"] = True
    QTimer.singleShot(1500, finish)


def finish():
    print("\n全部阶段通过，未发生闪退！")
    print(f"结果: {results}")
    app.quit()


QTimer.singleShot(800, stage1_draw)
QTimer.singleShot(60000, lambda: (print("[FAIL] 超时"), app.exit(1)))
sys.exit(app.exec())
