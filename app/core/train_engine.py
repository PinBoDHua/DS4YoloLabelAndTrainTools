"""训练引擎：封装 ultralytics 训练，实时指标回调、日志捕获、模型导出。"""
from __future__ import annotations

import io
import logging
import sys
from contextlib import redirect_stdout

from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


class _LogCapture(io.TextIOBase):
    """捕获 ultralytics stdout 日志逐行转发。"""

    def __init__(self, on_line):
        self._on_line = on_line
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                try:
                    self._on_line(line)
                except Exception:
                    pass
        return len(s)

    def flush(self) -> None:
        pass


class TrainWorker(QThread):
    """训练线程。

    通过 ultralytics callback 实时上报每个 epoch 的指标，
    停止训练通过 stop_requested 标志在 epoch 边界中断。
    """
    epoch_done = Signal(int, int, dict)   # (epoch, total_epochs, metrics)
    log_line = Signal(str)
    finished_ok = Signal(str, dict)       # (输出目录, 最终指标)
    failed = Signal(str)
    stopped = Signal()

    def __init__(self, model: str, data_yaml: str, epochs: int = 100, batch: int = 8,
                 imgsz: int = 640, lr0: float = 0.01, device: str = "0",
                 project: str = "", name: str = "train"):
        super().__init__()
        self._model = model
        self._data = data_yaml
        self._epochs = epochs
        self._batch = batch
        self._imgsz = imgsz
        self._lr0 = lr0
        self._device = device
        self._project = project
        self._name = name
        self.stop_requested = False

    @staticmethod
    def _ensure_model_file(model_name: str) -> None:
        """预训练模型不存在时预先下载到工作目录（镜像加速，避免 ultralytics 慢速直连）。"""
        from pathlib import Path
        p = Path(model_name)
        if p.exists() or p.is_absolute():
            return  # 已是本地文件
        if Path(p.name).exists():
            return  # 工作目录已有
        url = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{p.name}"
        from app.utils import download_with_fallback
        download_with_fallback(url, Path(p.name))

    def stop(self) -> None:
        self.stop_requested = True

    def run(self) -> None:
        try:
            from app.utils import patch_opencv_imread
            patch_opencv_imread()  # 中文路径数据集支持
            from ultralytics import YOLO

            self._ensure_model_file(self._model)

            def on_fit_epoch_end(trainer):
                if self.stop_requested:
                    trainer.stopping = True
                    raise InterruptedError("训练已被用户停止")
                epoch = trainer.epoch + 1
                metrics = {}
                m = getattr(trainer, "metrics", None)
                if m is not None:
                    for k, v in m.items():
                        try:
                            metrics[k] = float(v)
                        except (TypeError, ValueError):
                            pass
                self.epoch_done.emit(epoch, trainer.epochs, metrics)

            model = YOLO(self._model)
            # 新版 ultralytics 用 add_callback 注册回调（train() 不再接受 callbacks 参数）
            model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
            results = model.train(
                data=self._data,
                epochs=self._epochs,
                batch=self._batch,
                imgsz=self._imgsz,
                lr0=self._lr0,
                device=self._device,
                project=self._project or None,
                name=self._name,
                workers=0,          # Windows 下多进程 dataloader 易崩溃
                plots=True,
                verbose=True,
                exist_ok=True,
            )
            out_dir = str(model.trainer.save_dir) if hasattr(model, "trainer") else ""
            final = {}
            m = getattr(model.trainer, "metrics", None) if hasattr(model, "trainer") else None
            if m is not None:
                final = {k: float(v) for k, v in m.items()}
            if self.stop_requested:
                self.stopped.emit()
            else:
                self.finished_ok.emit(out_dir, final)
        except InterruptedError:
            self.stopped.emit()
        except Exception as e:
            logger.exception("训练失败")
            self.failed.emit(str(e))


def list_available_models(task: str = "") -> list[tuple[str, str]]:
    """可用预训练模型列表：[(模型名, 说明), ...]。

    task: detect / segment / pose / obb
    """
    all_models = [
        ("yolo11n.pt", "YOLO11n 检测-纳米"),
        ("yolo11s.pt", "YOLO11s 检测-小"),
        ("yolo11m.pt", "YOLO11m 检测-中"),
        ("yolov8n.pt", "YOLOv8n 检测-纳米"),
        ("yolov8s.pt", "YOLOv8s 检测-小"),
        ("yolov8m.pt", "YOLOv8m 检测-中"),
        ("yolov8l.pt", "YOLOv8l 检测-大"),
        ("yolo11n-seg.pt", "YOLO11n 分割-纳米"),
        ("yolo11s-seg.pt", "YOLO11s 分割-小"),
        ("yolov8n-seg.pt", "YOLOv8n 分割-纳米"),
        ("yolov8s-seg.pt", "YOLOv8s 分割-小"),
        ("yolo11n-pose.pt", "YOLO11n 关键点-纳米"),
        ("yolo11s-pose.pt", "YOLO11s 关键点-小"),
        ("yolov8n-pose.pt", "YOLOv8n 关键点-纳米"),
        ("yolov8s-pose.pt", "YOLOv8s 关键点-小"),
        ("yolov8n-obb.pt", "YOLOv8n 旋转框-纳米"),
        ("yolov8s-obb.pt", "YOLOv8s 旋转框-小"),
    ]
    if task == "segment":
        return [m for m in all_models if "seg" in m[0]]
    if task == "pose":
        return [m for m in all_models if "pose" in m[0]]
    if task == "obb":
        return [m for m in all_models if "obb" in m[0]]
    if task == "detect":
        return [m for m in all_models if "seg" not in m[0] and "pose" not in m[0] and "obb" not in m[0]]
    return all_models


class ExportWorker(QThread):
    """模型导出线程。"""
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, model_path: str, fmt: str, imgsz: int = 640):
        super().__init__()
        self._model = model_path
        self._fmt = fmt
        self._imgsz = imgsz

    def run(self) -> None:
        try:
            from ultralytics import YOLO
            model = YOLO(self._model)
            path = model.export(format=self._fmt, imgsz=self._imgsz)
            self.finished_ok.emit(str(path))
        except Exception as e:
            logger.exception("导出失败")
            self.failed.emit(str(e))
