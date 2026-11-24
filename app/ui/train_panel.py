"""训练面板：参数配置、实时损失曲线、训练日志、结果浏览与模型导出。"""
from __future__ import annotations

import logging
from pathlib import Path

import pyqtgraph as pg
from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QComboBox, QDialogButtonBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
                               QPushButton, QSpinBox, QTabWidget, QVBoxLayout,
                               QWidget)

from app.config import config
from app.core.train_engine import ExportWorker, TrainWorker, list_available_models
from app.theme import theme

logger = logging.getLogger(__name__)


class TrainPanel(QWidget):
    """训练面板（主窗口内嵌页，随数据集切换更新）。"""

    running_changed = Signal(bool)      # 训练开始/结束，主窗口据此更新页签
    best_model_ready = Signal(str)      # 训练完成后的 best.pt 路径（供推理页自动填充）

    def __init__(self, parent, dataset=None):
        super().__init__(parent)
        self.dataset = None

        self._worker: TrainWorker | None = None
        self._epochs_x: list[int] = []
        self._curves: dict[str, list] = {}
        self._gpu_probed = False

        root_layout = QVBoxLayout(self)

        # ---- 左：参数 / 右：曲线日志 ----
        main = QHBoxLayout()

        # 参数表单
        box = QGroupBox("训练参数")
        form = QFormLayout(box)
        self.lb_dataset = QLabel("未打开数据集")
        self.lb_dataset.setWordWrap(True)
        form.addRow("数据集:", self.lb_dataset)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        form.addRow("预训练模型:", self.model_combo)

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 2000)
        self.spin_epochs.setValue(config.get("train", {}).get("epochs", 100))
        form.addRow("训练轮数:", self.spin_epochs)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 256)
        self.spin_batch.setValue(config.get("train", {}).get("batch", 8))
        form.addRow("批量大小:", self.spin_batch)

        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(320, 2560)
        self.spin_imgsz.setSingleStep(32)
        self.spin_imgsz.setValue(config.get("train", {}).get("imgsz", 640))
        form.addRow("输入尺寸:", self.spin_imgsz)

        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(0.00001, 1.0)
        self.spin_lr.setDecimals(5)
        self.spin_lr.setSingleStep(0.001)
        self.spin_lr.setValue(config.get("train", {}).get("lr0", 0.01))
        form.addRow("学习率:", self.spin_lr)

        self.device_combo = QComboBox()
        self.device_combo.addItem("GPU（检测中...）", "0")
        self.device_combo.addItem("CPU（慢）", "cpu")
        form.addRow("训练设备:", self.device_combo)

        self.name_edit = QLineEdit("train")
        form.addRow("实验名称:", self.name_edit)

        main.addWidget(box, 1)

        # 曲线 + 日志
        self.tabs = QTabWidget()
        # 曲线
        curve_widget = QWidget()
        cv = QVBoxLayout(curve_widget)
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.addLegend(offset=(10, 10))
        self._curve_items: dict[str, pg.PlotDataItem] = {}
        cv.addWidget(self.plot)
        self.lb_last = QLabel("尚未开始训练")
        cv.addWidget(self.lb_last)
        self.tabs.addTab(curve_widget, "损失曲线")
        # 日志
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.tabs.addTab(self.log_view, "训练日志")
        main.addWidget(self.tabs, 2)
        root_layout.addLayout(main, 1)

        # ---- 底部按钮 ----
        btns = QHBoxLayout()
        self.btn_train = QPushButton("开始训练")
        self.btn_train.setProperty("primary", True)
        self.btn_train.clicked.connect(self._start_train)
        btns.addWidget(self.btn_train)
        self.btn_stop = QPushButton("停止训练")
        self.btn_stop.setProperty("danger", True)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_train)
        btns.addWidget(self.btn_stop)
        self.btn_open_dir = QPushButton("打开结果目录")
        self.btn_open_dir.clicked.connect(self._open_result_dir)
        btns.addWidget(self.btn_open_dir)
        btns.addStretch()
        self.export_combo = QComboBox()
        for fmt in ("onnx", "openvino", "engine", "tflite", "coreml"):
            self.export_combo.addItem(fmt.upper(), fmt)
        btns.addWidget(QLabel("导出格式:"))
        btns.addWidget(self.export_combo)
        self.btn_export = QPushButton("导出模型")
        self.btn_export.clicked.connect(self._export)
        btns.addWidget(self.btn_export)
        root_layout.addLayout(btns)

        self._apply_plot_theme()
        theme.theme_changed.connect(self._on_theme_changed)
        self.set_dataset(dataset)

    # ---------- 设备探测 ----------
    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._gpu_probed:
            self._gpu_probed = True
            from app.core.device_probe import ensure_probe
            ensure_probe(self.set_gpu_info)

    def set_gpu_info(self, ok: bool, desc: str) -> None:
        """GPU 探测结果回填设备下拉框。"""
        self._gpu_probed = True
        if ok:
            self.device_combo.setItemText(0, f"GPU - {desc}")
        else:
            self.device_combo.setItemText(0, f"GPU（{desc}）")
        self.device_combo.setItemData(0, "0")
        self.device_combo.setToolTip(desc)

    # ---------- 数据集绑定 ----------
    def set_dataset(self, dataset) -> None:
        """绑定数据集（主窗口打开/切换数据集时调用），同步模型列表。"""
        self.dataset = dataset
        if dataset is None:
            self.lb_dataset.setText("未打开数据集")
            return
        n = len(dataset.scan_images())
        self.lb_dataset.setText(f"<b>{Path(dataset.root).name}</b>（{n} 张图片）")
        prev = self.model_combo.currentData()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for name, desc in list_available_models(dataset.meta.task.value):
            self.model_combo.addItem(f"{desc} - {name}", name)
        idx = self.model_combo.findData(prev)
        if idx < 0:
            idx = self.model_combo.findData(config.get("train", {}).get("model", "yolov8n.pt"))
        self.model_combo.setCurrentIndex(max(idx, 0))
        self.model_combo.blockSignals(False)

    # ---------- 主题 ----------
    def _on_theme_changed(self, _name: str) -> None:
        self._apply_plot_theme()

    def _apply_plot_theme(self) -> None:
        c = theme.colors
        pg.setConfigOption("background", c.panel_bg)
        pg.setConfigOption("foreground", c.text)
        self.plot.setBackground(c.panel_bg)

    # ---------- 训练 ----------
    def _check_split(self) -> Path | None:
        """确保数据集已正确划分（train/val 均非空），返回 data.yaml 路径；否则询问。"""
        yaml_path = self.dataset.root / "data.yaml"
        train_dir = self.dataset.images_dir / "train"
        val_dir = self.dataset.images_dir / "val"

        def _has_images(d: Path) -> bool:
            return d.exists() and any(d.iterdir())

        if not _has_images(train_dir) or not _has_images(val_dir):
            ret = QMessageBox.question(
                self, "未划分数据集",
                "训练/验证集未正确划分（验证集为空或图片数量不足），是否现在重新划分（默认 8:1:1）？",
            )
            if ret != QMessageBox.Yes:
                return None
            try:
                self.dataset.split((0.8, 0.1, 0.1), 42, copy=True)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"划分失败：{e}")
                return None
        return self.dataset.write_data_yaml()

    def _start_train(self) -> None:
        if self.dataset is None:
            QMessageBox.information(self, "提示", "请先打开数据集。")
            return
        yaml_path = self._check_split()
        if yaml_path is None:
            return
        self._epochs_x = []
        self._curves = {}
        for item in self._curve_items.values():
            self.plot.removeItem(item)
        self._curve_items = {}
        self.log_view.clear()
        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)

        cfg = {
            "model": self.model_combo.currentData() or self.model_combo.currentText(),
            "imgsz": self.spin_imgsz.value(),
            "epochs": self.spin_epochs.value(),
            "batch": self.spin_batch.value(),
            "lr0": self.spin_lr.value(),
            "device": self.device_combo.currentData(),
        }
        config.set("train", cfg)

        self._worker = TrainWorker(
            model=cfg["model"],
            data_yaml=str(yaml_path),
            epochs=cfg["epochs"],
            batch=cfg["batch"],
            imgsz=cfg["imgsz"],
            lr0=cfg["lr0"],
            device=cfg["device"],
            project=str(self.dataset.root),
            name=self.name_edit.text().strip() or "train",
        )
        self._worker.epoch_done.connect(self._on_epoch)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.stopped.connect(self._on_stopped)
        self._worker.start()
        self.running_changed.emit(True)

    def _stop_train(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._log("正在停止训练（等待当前 epoch 结束）...")

    def _on_epoch(self, epoch: int, total: int, metrics: dict) -> None:
        self._epochs_x.append(epoch)
        for k, v in metrics.items():
            key = k.replace("train/", "").replace("val/", "")
            if key not in self._curves:
                self._curves[key] = []
            self._curves[key].append(v)
        # 更新曲线（保留已有长度对齐）
        colors = ["#ff6b6b", "#4a9eff", "#2da44e", "#ffb100", "#911eb4",
                  "#00c2a8", "#ff6ec7", "#808080"]
        for i, (key, values) in enumerate(self._curves.items()):
            item = self._curve_items.get(key)
            pen = pg.mkPen(colors[i % len(colors)], width=2)
            if item is None:
                item = self.plot.plot(self._epochs_x[:len(values)], values,
                                      pen=pen, name=key)
                self._curve_items[key] = item
            else:
                item.setData(self._epochs_x[:len(values)], values)
        best = ", ".join(f"{k}: {v:.4f}" for k, v in list(metrics.items())[:8])
        self.lb_last.setText(f"Epoch {epoch}/{total}  {best}")

    def _on_finished(self, out_dir: str, final: dict) -> None:
        self.running_changed.emit(False)
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._last_dir = out_dir
        self._last_best = Path(out_dir) / "weights" / "best.pt" if out_dir else None
        final_text = ", ".join(f"{k}: {v:.4f}" for k, v in list(final.items())[:8])
        self._log(f"训练完成！输出目录: {out_dir}")
        self.lb_last.setText(f"训练完成  {final_text}")
        if self._last_best is not None and self._last_best.exists():
            self.best_model_ready.emit(str(self._last_best))  # 推理页自动填充
        QMessageBox.information(self, "完成", f"训练完成！\n\n结果目录：\n{out_dir}\n\n"
                                             f"最佳权重: best.pt\n{final_text}")

    def _on_stopped(self) -> None:
        self.running_changed.emit(False)
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._log("训练已停止（已训练部分的结果仍会保存）")
        self.lb_last.setText("训练已停止")

    def _on_failed(self, err: str) -> None:
        self.running_changed.emit(False)
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._log(f"训练失败：{err}")
        QMessageBox.critical(self, "错误", f"训练失败：\n{err}")

    def _log(self, text: str) -> None:
        self.log_view.appendPlainText(text)

    # ---------- 结果与导出 ----------
    def _open_result_dir(self) -> None:
        d = getattr(self, "_last_dir", None)
        if not d or not Path(d).exists():
            d = self.dataset.root / "train"
        if Path(d).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(d)))
        else:
            QMessageBox.information(self, "提示", "还没有训练结果目录。")

    def _export(self) -> None:
        model_path = getattr(self, "_last_best", None)
        if model_path is None or not Path(model_path).exists():
            path, _ = QFileDialog.getOpenFileName(self, "选择模型权重", "", "模型文件 (*.pt)")
            if not path:
                return
            model_path = path
        fmt = self.export_combo.currentData()
        self.btn_export.setEnabled(False)
        self._worker_export = ExportWorker(str(model_path), fmt, self.spin_imgsz.value())
        self._worker_export.finished_ok.connect(
            lambda p: (QMessageBox.information(self, "完成", f"导出成功：\n{p}"),
                       self.btn_export.setEnabled(True)))
        self._worker_export.failed.connect(
            lambda e: (QMessageBox.critical(self, "错误", f"导出失败：{e}"),
                       self.btn_export.setEnabled(True)))
        self._worker_export.start()

    # 线程清理由主窗口统一负责：closeEvent 阻止训练中退出 + aboutToQuit 全局兜底。
