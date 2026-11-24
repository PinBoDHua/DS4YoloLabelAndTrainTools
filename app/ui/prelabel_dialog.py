"""模型预标注对话框：选择模型、参数配置、批量推理进度。"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QProgressBar,
                               QPushButton, QSpinBox, QVBoxLayout)

from app.config import config
from app.core.prelabel_engine import PrelabelWorker

logger = logging.getLogger(__name__)


class PrelabelDialog(QDialog):
    def __init__(self, parent, dataset):
        super().__init__(parent)
        self.dataset = dataset
        self.setWindowTitle("模型预标注")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)

        form = QFormLayout()
        # 模型
        row = QHBoxLayout()
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("选择 YOLO 模型权重（best.pt）")
        btn_browse = QPushButton("浏览...")
        btn_browse.clicked.connect(self._browse_model)
        row.addWidget(self.model_edit)
        row.addWidget(btn_browse)
        form.addRow("模型文件:", row)

        # 置信度 / IoU / 尺寸
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 1.0)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(config.get("prelabel", {}).get("conf", 0.25))
        form.addRow("置信度阈值:", self.spin_conf)

        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setRange(0.1, 1.0)
        self.spin_iou.setSingleStep(0.05)
        self.spin_iou.setValue(config.get("prelabel", {}).get("iou", 0.5))
        form.addRow("NMS IoU 阈值:", self.spin_iou)

        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(320, 2560)
        self.spin_imgsz.setSingleStep(32)
        self.spin_imgsz.setValue(640)
        form.addRow("推理尺寸:", self.spin_imgsz)

        self.device_combo = QComboBox()
        self.device_combo.addItem("GPU（推荐）", "0")
        self.device_combo.addItem("CPU", "cpu")
        form.addRow("推理设备:", self.device_combo)

        self.cb_skip = QCheckBox("跳过已有标注的图片")
        self.cb_skip.setChecked(True)
        form.addRow("", self.cb_skip)

        self.cb_overwrite = QCheckBox("覆盖已有标注（取消勾选跳过则全部重跑）")
        form.addRow("", self.cb_overwrite)

        layout.addLayout(form)

        tip = QLabel("模型任务类型（检测/分割/关键点/OBB）自动识别，"
                     "预标注结果可直接在画布中编辑修正。")
        tip.setStyleSheet("color: gray; font-size: 12px;")
        layout.addWidget(tip)

        # 进度
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.lb_progress = QLabel("")
        layout.addWidget(self.lb_progress)

        # 按钮
        btns = QDialogButtonBox(QDialogButtonBox.Close)
        self.btn_run = QPushButton("开始预标注")
        self.btn_run.setProperty("primary", True)
        self.btn_run.clicked.connect(self._run)
        btns.addButton(self.btn_run, QDialogButtonBox.ActionRole)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self._worker: PrelabelWorker | None = None

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择模型权重", "", "模型文件 (*.pt)")
        if path:
            self.model_edit.setText(path)

    def _run(self) -> None:
        model = self.model_edit.text().strip()
        if not model or not Path(model).exists():
            QMessageBox.warning(self, "提示", "请选择有效的模型权重文件。")
            return
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, "提示", "预标注正在进行中...")
            return

        images = self.dataset.scan_images()
        if not images:
            QMessageBox.warning(self, "提示", "数据集中没有图片。")
            return

        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, len(images))

        self._worker = PrelabelWorker(
            model, images, str(self.dataset.labels_dir),
            conf=self.spin_conf.value(), iou=self.spin_iou.value(),
            imgsz=self.spin_imgsz.value(), device=self.device_combo.currentData(),
            skip_existing=not self.cb_overwrite.isChecked(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, cur: int, total: int) -> None:
        self.progress.setValue(cur)
        self.lb_progress.setText(f"处理中 {cur}/{total}")

    def _on_finished(self, ok: int, skip: int, errors: int) -> None:
        self.btn_run.setEnabled(True)
        self.lb_progress.setText(f"完成：成功 {ok} 张，跳过 {skip} 张，失败 {errors} 张")
        config.set("prelabel", {"conf": self.spin_conf.value(),
                                "iou": self.spin_iou.value()})
        QMessageBox.information(
            self, "完成",
            f"预标注完成！\n成功 {ok} 张，跳过 {skip} 张，失败 {errors} 张。")

    def _on_failed(self, err: str) -> None:
        self.btn_run.setEnabled(True)
        self.lb_progress.setText("失败")
        QMessageBox.critical(self, "错误", f"预标注失败：{err}")

    def done(self, result: int) -> None:
        """关闭前停线程，避免线程运行时被销毁崩溃。"""
        w = getattr(self, "_worker", None)
        if w is not None and w.isRunning():
            w.cancel()
            w.wait()
        super().done(result)
