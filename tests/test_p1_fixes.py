"""P1 修复回归测试：类别重映射、训练中切换保护、OBB 旋转、POSE 右键、SAM 残留。"""
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

tmp = Path(tempfile.mkdtemp())
os.environ["YOLO_LABEL_CONFIG"] = str(tmp / "config.json")

from PIL import Image

from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox, QPushButton

from app.config import config
from app.core.label_data import ImageAnnotation, LabelItem, LabelType
from app.core.dataset_manager import create_dataset
from app.ui.main_window import MainWindow

app = QApplication([])

failures = []


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception:
        import traceback
        traceback.print_exc()
        failures.append(name)
        print(f"[FAIL] {name}")


def make_ds(name, classes, n_img=2):
    root = tmp / name
    ds = create_dataset(root, classes, LabelType.DETECT)
    for i in range(n_img):
        Image.new("RGB", (100, 100)).save(root / "images" / f"img{i}.jpg")
    ds.scan_images()
    return ds


# ---------------- 类别删除/重排重映射 ----------------
def test_class_delete_remappable():
    ds = make_ds("cls_del", ["cat", "dog", "bird"])
    win = MainWindow()
    win._open_dataset(str(ds.root), ds)
    # 画布与磁盘各有跨类别标注
    win.canvas.add_shape_item(LabelItem(LabelType.DETECT, 2, points=[[10, 10], [40, 40]]))
    win._save_current_annotation(quiet=True)
    lp = ds.label_path(ds.images[1])
    lp.write_text("0 0.1 0.1 0.2 0.2\n1 0.3 0.3 0.2 0.2\n2 0.5 0.5 0.2 0.2\n",
                  encoding="utf-8")
    orig = QMessageBox.question
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    try:
        win._on_class_delete_requested(1)  # 删除 dog
    finally:
        QMessageBox.question = orig
    # 元数据
    assert ds.meta.classes == ["cat", "bird"], ds.meta.classes
    assert win.dataset.meta.classes == ["cat", "bird"]
    assert win.class_panel.classes == ["cat", "bird"]
    # 磁盘重映射：dog 行删除，bird 2→1
    lines = lp.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("0 ") and lines[1].startswith("1 "), lines
    assert len(lines) == 2
    # 画布形状同步重映射
    assert win.canvas.shapes[0].label.class_id == 1
    win.close()


def test_class_move_remappable():
    ds = make_ds("cls_move", ["cat", "dog"])
    lp = ds.label_path(ds.images[0])
    lp.write_text("0 0.1 0.1 0.2 0.2\n1 0.3 0.3 0.2 0.2\n", encoding="utf-8")
    win = MainWindow()
    win._open_dataset(str(ds.root), ds)
    win._on_class_move_requested(0, 1)  # 上移 dog
    assert ds.meta.classes == ["dog", "cat"], ds.meta.classes
    assert win.class_panel.classes == ["dog", "cat"]
    lines = lp.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("1 ") and lines[1].startswith("0 "), lines
    win.close()


def test_delete_last_class_no_crash():
    ds = make_ds("cls_last", ["only"])
    win = MainWindow()
    win._open_dataset(str(ds.root), ds)
    win.canvas.add_shape_item(LabelItem(LabelType.DETECT, 0, points=[[10, 10], [40, 40]]))
    orig = QMessageBox.question
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    try:
        win._on_class_delete_requested(0)  # 删除唯一类别
    finally:
        QMessageBox.question = orig
    assert ds.meta.classes == []
    assert win.class_panel.classes == []
    # 空类别下状态栏刷新不应抛 IndexError
    win._on_class_changed(0)
    win._on_shape_selected(None)
    win.close()


def test_class_panel_requests():
    """面板删除/移动只发请求，不直接改列表。"""
    from app.ui.class_panel import ClassPanel
    panel = ClassPanel()
    panel.set_classes(["a", "b", "c"])
    got = []
    panel.class_delete_requested.connect(lambda i: got.append(("del", i)))
    panel.class_move_requested.connect(lambda i, j: got.append(("move", i, j)))
    panel._current = 1
    panel._on_delete()
    panel._move(-1)
    assert got == [("del", 1), ("move", 1, 0)], got
    assert panel.classes == ["a", "b", "c"], "面板不应在请求阶段修改列表"


# ---------------- 训练中切换保护 ----------------
def test_switch_blocked_while_training():
    ds_a = make_ds("train_a", ["x"])
    ds_b = make_ds("train_b", ["y"])
    win = MainWindow()
    win._open_dataset(str(ds_a.root), ds_a)
    fake_worker = SimpleNamespace(isRunning=lambda: True, stop=lambda: None, wait=lambda: None)
    win.train_panel._worker = fake_worker
    orig = QMessageBox.warning
    QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
    try:
        win._open_dataset(str(ds_b.root), ds_b)
    finally:
        QMessageBox.warning = orig
    assert win.dataset.root == ds_a.root, "训练中不应切换数据集"
    fake_worker.isRunning = lambda: False
    win._open_dataset(str(ds_b.root), ds_b)
    assert win.dataset.root == ds_b.root
    win.close()


# ---------------- 退出兜底保存（自动保存关闭时） ----------------
def test_about_to_quit_saves_edits():
    ds = make_ds("quit_save", ["q"])
    win = MainWindow()
    win._open_dataset(str(ds.root), ds)
    config._data["auto_save"] = False
    win.canvas.add_shape_item(LabelItem(LabelType.DETECT, 0, points=[[10, 10], [40, 40]]))
    win._on_app_about_to_quit()
    config._data["auto_save"] = True
    content = ds.label_path(ds.images[0]).read_text(encoding="utf-8")
    assert content.strip(), "app.quit() 路径退出应保存当前编辑"
    win.close()


# ---------------- 跨数据集保存保护 ----------------
def test_no_cross_dataset_write():
    ds_a = make_ds("cross_a", ["a"])
    ds_b = make_ds("cross_b", ["b"])
    win = MainWindow()
    win._open_dataset(str(ds_a.root), ds_a)
    win.canvas.add_shape_item(LabelItem(LabelType.DETECT, 0, points=[[10, 10], [40, 40]]))
    # 模拟图片加载失败后的错位状态：dataset/images 已切换，current_ann 还是旧数据集
    win.dataset = ds_b
    win.images = ds_b.scan_images()
    win._save_current_annotation(quiet=True)
    strays = list((tmp / "cross_b" / "labels").glob("*.txt"))
    assert not strays, f"旧数据集标注被写入新数据集: {strays}"
    win.close()


# ---------------- OBB 旋转基准 ----------------
def test_obb_rotation_follows_handle():
    import math
    from app.ui.canvas import ShapeItem
    lb = LabelItem(LabelType.OBB, 0, points=[[100, 100], [200, 100], [200, 150], [100, 150]])
    item = ShapeItem(lb)
    cx, cy = 150, 125
    # 旧实现以第一个顶点方向为基准，旋转后手柄方向不指向鼠标
    target = QPointF(cx + 60, cy)  # 期望手柄转到 0 度方向
    item.apply_handle_move("rot", target)
    handle = item.handles()["rot"]
    angle = math.atan2(handle.y() - cy, handle.x() - cx)
    assert abs(angle) < 0.05, f"旋转手柄未对齐鼠标方向: {angle:.3f}"
    # 旋转不应改变边长
    pts = lb.points
    side1 = math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
    side2 = math.hypot(pts[2][0] - pts[1][0], pts[2][1] - pts[1][1])
    assert abs(side1 - 100) < 1e-6 and abs(side2 - 50) < 1e-6, (side1, side2)


# ---------------- POSE 右键转发 + 撤销快照时序 ----------------
class FakeMouseEvent:
    def __init__(self, button, pos):
        self._button = button
        self._pos = pos

    def button(self):
        return self._button

    def position(self):
        return self._pos

    def modifiers(self):
        return Qt.NoModifier

    def accept(self):
        pass

    def ignore(self):
        pass


def test_pose_right_click_cycle_and_snapshot_order():
    from app.ui.canvas import LabelCanvas
    from app.ui import tools as tools_mod
    canvas = LabelCanvas()
    canvas.current_ann = ImageAnnotation(image_path="x")
    kpts = [[150, 150, 2], [160, 150, 2]]
    lb = LabelItem(LabelType.POSE, 0, points=[[100, 100], [200, 200]], keypoints=kpts)
    item = canvas.add_shape_item(lb)
    snapshots = []
    canvas.edit_begin.connect(
        lambda: snapshots.append(list(item.label.keypoints[0])))
    tool = tools_mod.SelectTool(canvas)
    # 直接给场景坐标（视图坐标->场景坐标的映射属于 QGraphicsView 布局细节，
    # 由真实交互冒烟测试覆盖）
    tool._pos = lambda event: QPointF(150, 150)
    ev = FakeMouseEvent(Qt.RightButton, QPointF(150, 150))
    handled = tool.mousePress(ev)
    assert handled, "POSE 关键点右键应被转发到形状"
    assert kpts[0][2] == 1, f"可见性应从 2 循环到 1，实际 {kpts[0][2]}"
    assert kpts[1][2] == 2, "未命中的关键点不应变化"
    assert snapshots and snapshots[0][2] == 2, \
        f"撤销快照应在修改前生成（快照可见性=2），实际 {snapshots}"


def test_pose_right_click_miss_falls_through():
    from app.ui.canvas import LabelCanvas
    from app.ui import tools as tools_mod
    canvas = LabelCanvas()
    canvas.current_ann = ImageAnnotation(image_path="x")
    lb = LabelItem(LabelType.POSE, 0, points=[[100, 100], [200, 200]],
                   keypoints=[[150, 150, 2]])
    canvas.add_shape_item(lb)
    tool = tools_mod.SelectTool(canvas)
    tool._pos = lambda event: QPointF(10, 10)
    ev = FakeMouseEvent(Qt.RightButton, QPointF(10, 10))  # 未命中关键点
    assert not tool.mousePress(ev), "未命中关键点的右键应放行给上下文菜单"


# ---------------- SAM 旧结果残留 ----------------
def test_sam_stale_results():
    import numpy as np
    from app.ui.sam_dialog import SamDialog, SamPreview
    # 不走 __init__（避免加载模型），手工装配被测状态
    dlg = SamDialog.__new__(SamDialog)
    dlg._req_id = 5
    dlg._last_poly = [(10, 10), (50, 10), (50, 50)]
    dlg.preview = SamPreview()
    dlg.btn_add = QPushButton()
    dlg.btn_add.setEnabled(True)
    dlg.lb_status = QLabel()
    # 1) 清除提示作废在途请求
    dlg._clear_hints()
    assert dlg._req_id == 6, "清除提示应作废在途请求"
    assert dlg._last_poly is None and not dlg.btn_add.isEnabled()
    assert dlg.preview.mask_overlay is None, "清除提示应清空掩码"
    # 2) 迟到的旧请求结果应被忽略
    dlg._on_segment_done(5, [(1, 1), (2, 2), (3, 3)])
    assert dlg._last_poly is None, "被作废的请求不应恢复旧结果"
    assert dlg.preview.mask_overlay is None, "被作废的请求不应重绘掩码"
    # 3) 当前请求无结果时清空旧多边形
    dlg._last_poly = [(1, 1)]
    dlg.btn_add.setEnabled(True)
    dlg._on_segment_done(6, None)
    assert dlg._last_poly is None, "新请求无结果应清空旧多边形"
    assert not dlg.btn_add.isEnabled(), "无结果时应禁用添加按钮"


def test_sam_preview_set_mask_rebuilds_pixmap():
    import numpy as np
    from app.ui.sam_dialog import SamPreview
    preview = SamPreview()
    preview.resize(420, 300)
    white = np.full((100, 100, 3), 255, dtype=np.uint8)  # BGR 白
    preview.set_image(white)
    # set_mask 后 _pixmap 必须重建（掩码烘焙进像素图，否则不显示）
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:50, :] = 1  # 左半掩码
    preview.set_mask(mask)
    assert preview._pixmap is not None
    img = preview._pixmap.toImage()
    c_masked = img.pixelColor(10, 25)
    c_clear = img.pixelColor(10, 75)
    assert c_masked.green() > 200 and c_masked.red() < 200, \
        f"掩码区域应显示绿色叠加: {c_masked.red()},{c_masked.green()},{c_masked.blue()}"
    assert c_clear.red() > 240 and c_clear.green() > 240, \
        f"非掩码区域应保持原图: {c_clear.red()},{c_clear.green()},{c_clear.blue()}"


if __name__ == "__main__":
    print("== P1 修复回归测试 ==")
    check("类别删除重映射", test_class_delete_remappable)
    check("类别重排重映射", test_class_move_remappable)
    check("删除最后一个类别不崩溃", test_delete_last_class_no_crash)
    check("类别面板只发请求", test_class_panel_requests)
    check("训练中禁止切换数据集", test_switch_blocked_while_training)
    check("退出兜底保存", test_about_to_quit_saves_edits)
    check("跨数据集写入保护", test_no_cross_dataset_write)
    check("OBB 旋转对齐手柄", test_obb_rotation_follows_handle)
    check("POSE 右键循环与快照时序", test_pose_right_click_cycle_and_snapshot_order)
    check("POSE 右键未命中放行", test_pose_right_click_miss_falls_through)
    check("SAM 旧结果作废", test_sam_stale_results)
    check("SAM 掩码像素图重建", test_sam_preview_set_mask_rebuilds_pixmap)
    if failures:
        print(f"\n{len(failures)} 项失败: {failures}")
        sys.exit(1)
    print("\nP1 修复回归测试全部通过！")
