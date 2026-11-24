"""模型推理测试面板：单图/文件夹/视频推理与结果可视化。"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QProgressBar, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

from app.config import config

logger = logging.getLogger(__name__)


class _TestWorker(QThread):
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, model: str, source: str, out_dir: str, conf: float,
                 iou: float, imgsz: int, device: str, save_txt: bool):
        super().__init__()
        self._model, self._source, self._out = model, source, out_dir
        self._conf, self._iou, self._imgsz = conf, iou, imgsz
        self._device, self._save_txt = device, save_txt

    def run(self) -> None:
        try:
            from ultralytics import YOLO
            model = YOLO(self._model)
            results = model.predict(
                source=self._source,
                conf=self._conf,
                iou=self._iou,
                imgsz=self._imgsz,
                device=self._device,
                save=True,
                save_txt=self._save_txt,
                project=str(Path(self._out) / "runs"),
                name="predict",
                exist_ok=True,
            )
            self.finished_ok.emit(str(results[0].save_dir))
        except Exception as e:
            logger.exception("推理失败")
            self.failed.emit(str(e))


class TestPanel(QWidget):
    """推理测试（主窗口内嵌页）。"""

    def __init__(self, parent):
        super().__init__(parent)
        self._gpu_probed = False

        layout = QVBoxLayout(self)
        form = QFormLayout()

        row = QHBoxLayout()
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("选择模型权重（best.pt），训练完成后自动填入")
        btn = QPushButton("浏览...")
        btn.clicked.connect(lambda: self._browse(self.model_edit, "模型文件 (*.pt)"))
        row.addWidget(self.model_edit)
        row.addWidget(btn)
        form.addRow("模型:", row)

        row2 = QHBoxLayout()
        self.src_edit = QLineEdit()
        self.src_edit.setPlaceholderText("图片 / 文件夹 / 视频路径")
        btn2 = QPushButton("浏览...")
        btn2.clicked.connect(self._browse_source)
        row2.addWidget(self.src_edit)
        row2.addWidget(btn2)
        form.addRow("输入:", row2)

        row3 = QHBoxLayout()
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("结果输出目录")
        btn3 = QPushButton("浏览...")
        btn3.clicked.connect(lambda: self._browse_dir(self.out_edit))
        row3.addWidget(self.out_edit)
        row3.addWidget(btn3)
        form.addRow("输出目录:", row3)

        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 1.0)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.25)
        form.addRow("置信度:", self.spin_conf)

        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(320, 2560)
        self.spin_imgsz.setSingleStep(32)
        self.spin_imgsz.setValue(640)
        form.addRow("推理尺寸:", self.spin_imgsz)

        self.device_combo = QComboBox()
        self.device_combo.addItem("GPU（检测中...）", "0")
        self.device_combo.addItem("CPU", "cpu")
        form.addRow("设备:", self.device_combo)

        self.cb_save_txt = QCheckBox("同时保存 YOLO txt 结果")
        form.addRow("", self.cb_save_txt)

        layout.addLayout(form)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # 忙碌状态
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.btn_run = QPushButton("开始推理")
        self.btn_run.setProperty("primary", True)
        self.btn_run.clicked.connect(self._run)
        layout.addWidget(self.btn_run)

        # 打开结果目录
        self.btn_open_dir = QPushButton("打开结果目录")
        self.btn_open_dir.clicked.connect(self._open_result_dir)
        layout.addWidget(self.btn_open_dir)

    # ---------- 设备探测 / 外部填充 ----------
    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._gpu_probed:
            self._gpu_probed = True
            from app.core.device_probe import ensure_probe
            ensure_probe(self.set_gpu_info)

    def set_gpu_info(self, ok: bool, desc: str) -> None:
        self._gpu_probed = True
        if ok:
            self.device_combo.setItemText(0, f"GPU - {desc}")
        else:
            self.device_combo.setItemText(0, f"GPU（{desc}）")
        self.device_combo.setToolTip(desc)

    def set_model(self, path: str) -> None:
        """训练完成后自动填入最佳权重路径。"""
        self.model_edit.setText(path)

    def _browse(self, edit: QLineEdit, filter_: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", filter_)
        if path:
            edit.setText(path)

    def _browse_dir(self, edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择目录")
        if path:
            edit.setText(path)

    def _browse_source(self) -> None:
        # 图片/视频文件或文件夹
        path, _ = QFileDialog.getOpenFileName(
            self, "选择输入", "", "媒体文件 (*.jpg *.jpeg *.png *.bmp *.mp4 *.avi)")
        if path:
            self.src_edit.setText(path)
        else:
            d = QFileDialog.getExistingDirectory(self, "选择文件夹")
            if d:
                self.src_edit.setText(d)

    def _run(self) -> None:
        model, source = self.model_edit.text().strip(), self.src_edit.text().strip()
        out = self.out_edit.text().strip() or str(Path(source).parent)
        if not model or not Path(model).exists():
            QMessageBox.warning(self, "提示", "请选择有效的模型文件。")
            return
        if not source or not Path(source).exists():
            QMessageBox.warning(self, "提示", "请选择有效的输入路径。")
            return
        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self._worker = _TestWorker(
            model, source, out, self.spin_conf.value(), 0.5,
            self.spin_imgsz.value(), self.device_combo.currentData(),
            self.cb_save_txt.isChecked(),
        )
        self._worker.finished_ok.connect(self._done)
        self._worker.failed.connect(self._fail)
        self._worker.start()

    def _done(self, save_dir: str) -> None:
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)
        self._last_dir = save_dir
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        ret = QMessageBox.information(
            self, "完成", f"推理完成！\n结果目录：\n{save_dir}\n\n是否打开结果目录？",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ret == QMessageBox.Yes:
            QDesktopServices.openUrl(QUrl.fromLocalFile(save_dir))

    def _open_result_dir(self) -> None:
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        d = getattr(self, "_last_dir", None)
        if d and Path(d).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(d))
        else:
            QMessageBox.information(self, "提示", "还没有推理结果目录。")

    def _fail(self, err: str) -> None:
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)
        QMessageBox.critical(self, "错误", f"推理失败：{err}")

    def shutdown(self) -> None:
        """停掉后台推理线程（应用退出前由主窗口调用）。"""
        w = getattr(self, "_worker", None)
        if w is not None and w.isRunning():
            w.requestInterruption()
            w.wait()
