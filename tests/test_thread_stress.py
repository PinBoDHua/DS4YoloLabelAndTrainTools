"""线程压力测试：反复重启缩略图线程 + 过滤切换 + 快速关闭（使用临时数据集）。"""
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

tmp = Path(tempfile.mkdtemp())
# 重定向配置到临时目录（必须在导入 app 模块前设置）
os.environ["YOLO_LABEL_CONFIG"] = str(tmp / "config.json")

from PIL import Image

from app.utils import setup_logging
setup_logging()

from PySide6.QtCore import QThread, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from app.config import config
from app.theme import theme
from app.ui.main_window import MainWindow
from app.core.dataset_manager import create_dataset
from app.core.label_data import LabelType

app = QApplication(sys.argv)
theme.apply(app, "light")

tmp = tmp / "压测数据集"
ds = create_dataset(tmp, ["qq"], LabelType.DETECT)
for i in range(6):
    Image.new("RGB", (800, 600), (120, 90 + i * 20, 180)).save(
        tmp / "images" / f"压测图{i}.png")

win = MainWindow()
win.show()
win._open_dataset(str(tmp), ds)

def verify_task_progress_on_gui_thread():
    threads = []
    original_set_value = QProgressDialog.setValue
    original_information = QMessageBox.information

    def record_thread(dialog, value):
        threads.append(QThread.currentThread())
        return original_set_value(dialog, value)

    def work(callback):
        callback(1, 1)
        return "ok"

    def close_progress_dialogs():
        for widget in app.topLevelWidgets():
            if isinstance(widget, QProgressDialog):
                widget.close()

    try:
        QProgressDialog.setValue = record_thread
        QMessageBox.information = staticmethod(lambda *a, **k: None)
        QTimer.singleShot(200, close_progress_dialogs)
        win._run_task("线程测试", work, "完成")
    finally:
        QProgressDialog.setValue = original_set_value
        QMessageBox.information = original_information
    assert threads, "未执行进度更新"
    assert all(thread == app.thread() for thread in threads), "后台线程直接更新了进度对话框"
    print("[OK] 通用任务进度仅在 GUI 线程更新")


def verify_inference_shutdown_waits():
    class SlowThread(QThread):
        def run(self):
            time.sleep(5.2)

    worker = SlowThread()
    win.test_panel._worker = worker
    worker.start()
    try:
        win.test_panel.shutdown()
        assert not worker.isRunning(), "推理退出等待超时后线程仍在运行"
    finally:
        worker.wait()
        win.test_panel._worker = None
    print("[OK] 推理退出等待线程真正结束")


def verify_sam_results_belong_to_requests():
    from app.core.sam_engine import SamEngine

    class ArrayWrapper:
        def __init__(self, array):
            self.array = array

        def cpu(self):
            return self

        def numpy(self):
            return self.array

    class FakeModel:
        def predict(self, source, **kwargs):
            x = int(kwargs["bboxes"][0][0])
            if x < 10:
                time.sleep(0.1)
            mask = np.zeros(source.shape[:2], dtype=np.uint8)
            mask[2:20, x + 2:x + 20] = 1
            return [SimpleNamespace(masks=SimpleNamespace(data=[ArrayWrapper(mask)]))]

    engine = SamEngine()
    SamEngine._model = FakeModel()
    results = {}
    engine.segment_done.connect(lambda req_id, polygon: results.update({req_id: polygon}))
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    workers = []
    try:
        engine.segment(image, [], [], [1, 1, 25, 25], 1)
        workers.append(engine._worker)
        engine.segment(image, [], [], [32, 1, 56, 25], 2)
        workers.append(engine._worker)
        deadline = time.monotonic() + 3
        while len(results) < 2 and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert results.get(1), f"请求 1 未返回有效结果: {results}"
        assert results.get(2), f"请求 2 未返回有效结果: {results}"
        assert min(p[0] for p in results[1]) < 25, "请求 1 读取了其他 worker 的结果"
        assert min(p[0] for p in results[2]) > 25, "请求 2 多边形位置错误"
    finally:
        engine.shutdown()
        for worker in workers:
            worker.wait()
        SamEngine._model = None
    print("[OK] SAM 并发请求保留各自 worker 与结果")


def verify_gpu_probe_shutdown():
    from app.core import device_probe

    class SlowThread(QThread):
        def run(self):
            time.sleep(0.3)

    worker = SlowThread()
    device_probe._thread = worker
    device_probe._running = True
    worker.start()
    try:
        win._on_app_about_to_quit()
        assert not worker.isRunning(), "GPU 探测线程未接入退出清理"
    finally:
        worker.wait()
        device_probe._thread = None
        device_probe._running = False
    print("[OK] GPU 探测线程接入退出清理")


def run_regressions():
    failed = False
    for verify in (verify_task_progress_on_gui_thread,
                   verify_inference_shutdown_waits,
                   verify_sam_results_belong_to_requests,
                   verify_gpu_probe_shutdown):
        try:
            verify()
        except Exception:
            traceback.print_exc()
            failed = True
    if failed:
        app.exit(1)
        return
    QTimer.singleShot(0, stress)


state = {"i": 0}


def stress():
    i = state["i"]
    # 反复切换过滤模式 → 每次重启缩略图线程
    win.image_panel.filter_combo.setCurrentIndex(i % 3)
    # 反复 set_images（模拟导入）
    if i % 5 == 0:
        win.image_panel.set_images(win.images, win._labeled_cache)
    state["i"] += 1
    if state["i"] < 30:
        QTimer.singleShot(120, stress)
    else:
        print("== 压测完成，立即关闭窗口 ==")
        win.close()  # 立即关闭，检验线程未结束时退出的安全性
        QTimer.singleShot(2000, app.quit)


QTimer.singleShot(800, run_regressions)
QTimer.singleShot(60000, lambda: (print("[FAIL] 超时"), app.exit(1)))
sys.exit(app.exec())
