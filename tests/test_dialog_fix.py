"""对话框交互回归测试：自动填写并确认「新建数据集」对话框，验证无枚举兼容性错误。"""
import os
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

tmp = Path(tempfile.mkdtemp())
# 重定向配置到临时目录（必须在导入 app 模块前设置）
os.environ["YOLO_LABEL_CONFIG"] = str(tmp / "config.json")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog, QLineEdit,
                               QMessageBox, QPushButton)

from app.utils import setup_logging
setup_logging()

from app.config import config
from app.theme import theme
from app.ui.main_window import MainWindow

app = QApplication(sys.argv)
theme.apply(app, "light")

win = MainWindow()
win.show()

result = {"created": False}
failures: list[str] = []


def record_fail(msg: str) -> None:
    """记录失败：任何失败都必须反映到进程退出码。"""
    print(f"[FAIL] {msg}")
    failures.append(msg)


def finish(exit_code: int) -> None:
    code = 1 if failures else exit_code
    if failures:
        print(f"共 {len(failures)} 项失败:")
        for f in failures:
            print(f"  - {f}")
    app.exit(code)


def find_dialog():
    """查找当前活动的对话框。"""
    for w in app.topLevelWidgets():
        if isinstance(w, QDialog) and w.isVisible():
            return w
    return None


def step1_open():
    print("步骤 1: 触发新建数据集")
    win._new_dataset()


def step2_fill():
    dlg = find_dialog()
    if dlg is None:
        record_fail("对话框未打开")
        finish(1)
        return
    print(f"步骤 2: 找到对话框 {type(dlg).__name__}")
    # 填写目录与类别
    edits = dlg.findChildren(QLineEdit)
    edits[0].setText(str(tmp / "my_ds"))
    edits[1].setText("cat, dog, bird")
    print("  已填写目录与类别")


def step3_confirm():
    dlg = find_dialog()
    if dlg is None:
        record_fail("对话框未找到")
        finish(1)
        return
    print("步骤 3: 点击确定")
    for btn in dlg.findChildren(QPushButton):
        if btn.text() in ("OK", "确定", "Ok"):
            btn.click()
            break
    else:
        record_fail("未找到确认按钮，对话框未提交")
        finish(1)


def verify_import_annotation_identity():
    from PIL import Image
    from app.core.dataset_manager import create_dataset
    from app.core.label_data import LabelItem, LabelType

    root = tmp / "import_regression"
    ds = create_dataset(root, ["object"], LabelType.DETECT)
    Image.new("RGB", (20, 20)).save(ds.images_dir / "b.png")
    source = tmp / "a.png"
    Image.new("RGB", (20, 20)).save(source)
    win._open_dataset(str(ds.root), ds)
    win.canvas.add_shape_item(LabelItem(LabelType.DETECT, 0, points=[[1, 1], [8, 8]]))

    original_dialog = QFileDialog.getOpenFileNames
    original_message = QMessageBox.information
    try:
        QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: ([str(source)], ""))
        QMessageBox.information = staticmethod(lambda *a, **k: None)
        win._import_images()
    finally:
        QFileDialog.getOpenFileNames = original_dialog
        QMessageBox.information = original_message
    win._save_current_annotation()

    a_count = len(ds.read_annotation(str(ds.images_dir / "a.png")).labels)
    b_count = len(ds.read_annotation(str(ds.images_dir / "b.png")).labels)
    assert a_count == 0, f"当前标注被写入新导入图片 a.png: {a_count}"
    assert b_count == 1, f"当前图片 b.png 的标注丢失: {b_count}"
    print("[OK] 导入图片后标注仍属于原图片")


def verify_prelabel_close_refreshes():
    from PIL import Image
    import app.ui.prelabel_dialog as prelabel_module
    from app.core.dataset_manager import create_dataset
    from app.core.label_data import ImageAnnotation, LabelItem, LabelType
    from app.core.label_io import write_yolo_txt

    root = tmp / "prelabel_regression"
    ds = create_dataset(root, ["object"], LabelType.DETECT)
    for name in ("a.png", "b.png"):
        Image.new("RGB", (20, 20)).save(ds.images_dir / name)
    win._open_dataset(str(ds.root), ds)
    current = str(ds.images_dir / "a.png")

    class FakePrelabelDialog:
        def __init__(self, *args):
            pass

        def exec(self):
            ann = ImageAnnotation(current, [
                LabelItem(LabelType.DETECT, 0, points=[[1, 1], [8, 8]])])
            write_yolo_txt(ann, ds.label_path(current))
            return QDialog.DialogCode.Rejected

    original_dialog = prelabel_module.PrelabelDialog
    try:
        prelabel_module.PrelabelDialog = FakePrelabelDialog
        win._prelabel()
    finally:
        prelabel_module.PrelabelDialog = original_dialog
    win._navigate(1)
    count = len(ds.read_annotation(current).labels)
    assert count == 1, f"预标注关闭后结果被旧画布覆盖: {count}"
    print("[OK] 预标注关闭后刷新且切图不覆盖结果")


def verify_empty_dataset_clears_annotation_identity():
    from PIL import Image
    from app.core.dataset_manager import create_dataset
    from app.core.label_data import LabelItem, LabelType

    source_ds = create_dataset(tmp / "source_dataset", ["object"], LabelType.DETECT)
    Image.new("RGB", (20, 20)).save(source_ds.images_dir / "a.png")
    empty_ds = create_dataset(tmp / "empty_dataset", ["object"], LabelType.DETECT)
    original_auto_save = config.get("auto_save", True)
    original_information = QMessageBox.information
    try:
        config._data["auto_save"] = False
        win._open_dataset(str(source_ds.root), source_ds)
        win.canvas.add_shape_item(LabelItem(LabelType.DETECT, 0, points=[[1, 1], [8, 8]]))
        QMessageBox.information = staticmethod(lambda *a, **k: None)
        win._open_dataset(str(empty_ds.root), empty_ds)
    finally:
        QMessageBox.information = original_information
        config._data["auto_save"] = original_auto_save

    problems = []
    if win.current_ann is not None or getattr(win.canvas, "current_ann", None) is not None:
        problems.append("空数据集仍保留上一数据集的标注身份")
    try:
        win._save_current_annotation()
    except Exception as exc:
        problems.append(f"空数据集保存抛出异常: {exc}")
    source_count = len(source_ds.read_annotation(str(source_ds.images_dir / "a.png")).labels)
    if source_count != 1:
        problems.append(f"切换前未保存旧数据集标注: {source_count}")
    assert not problems, "；".join(problems)
    print("[OK] 切换空数据集前保存旧图并清空标注身份")


def step4_verify():
    meta = tmp / "my_ds" / "dataset.json"
    if meta.exists():
        print(f"[OK] 数据集创建成功: {meta}")
        result["created"] = True
    else:
        record_fail("数据集未创建")
        finish(1)
        return
    if win.dataset is not None and win.dataset.meta.classes == ["cat", "dog", "bird"]:
        print("[OK] 主窗口已打开数据集，类别正确")
    else:
        record_fail(f"主窗口状态异常: dataset={win.dataset}")
        finish(1)
        return
    QTimer.singleShot(0, step5_regressions)


def step5_regressions():
    for verify in (verify_import_annotation_identity,
                   verify_prelabel_close_refreshes,
                   verify_empty_dataset_clears_annotation_identity):
        try:
            verify()
        except Exception:
            traceback.print_exc()
            record_fail(f"{verify.__name__} 失败")
    print("对话框与标注归属回归测试完成" if not failures else "存在失败项")
    finish(0)


QTimer.singleShot(500, step1_open)
QTimer.singleShot(1200, step2_fill)
QTimer.singleShot(1600, step3_confirm)
QTimer.singleShot(2600, step4_verify)
QTimer.singleShot(15000, lambda: (record_fail("超时"), finish(1)))
sys.exit(app.exec())
