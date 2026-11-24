"""SAM 智能分割引擎：框/点提示 → 精确掩码 → 多边形。

基于 ultralytics 内置 SAM 支持（mobile_sam.pt 轻量模型，约 40MB，
8GB 显存实时推理，无需额外编译安装，Windows 开箱即用）。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from app.config import MODELS_DIR, config

logger = logging.getLogger(__name__)

# 模型名 -> 下载 URL（ultralytics 官方资源）
SAM_MODELS = {
    "mobile_sam": ("mobile_sam.pt",
                   "https://github.com/ultralytics/assets/releases/download/v8.3.0/mobile_sam.pt"),
    "sam_b": ("sam_b.pt",
              "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam_b.pt"),
}


def download_sam_model(model_name: str, progress_cb=None) -> bool:
    """下载 SAM 权重到 models 目录（直连失败自动切换加速镜像）。"""
    fname, url = SAM_MODELS[model_name]
    target = MODELS_DIR / fname
    from app.utils import download_with_fallback
    return download_with_fallback(url, target, progress_cb)


class SamEngine(QObject):
    """SAM 分割引擎（懒加载，线程安全）。"""

    segment_done = Signal(int, object)  # (请求 id, 多边形顶点列表 或 None)

    _model = None
    _model_name = None
    _lock = __import__("threading").Lock()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loader: QThread | None = None
        self._worker: QThread | None = None
        self._workers: set[QThread] = set()

    @staticmethod
    def is_loaded() -> bool:
        return SamEngine._model is not None

    def shutdown(self) -> None:
        """停止所有后台线程（对话框关闭/程序退出前调用，避免线程运行时被销毁崩溃）。"""
        for thr in [self._loader, *list(self._workers)]:
            if thr is not None and thr.isRunning():
                thr.requestInterruption()
                thr.wait()

    # ---------- 模型加载 ----------
    def ensure_loaded(self, parent_cb) -> None:
        """确保模型已加载。后台线程加载，完成后 parent_cb(ok: bool, err: str) 回调。"""
        model_name = config.get("sam_model", "mobile_sam")

        class _LoadThread(QThread):
            def __init__(self):
                super().__init__()
                self.success = False
                self.err = ""

            def run(self):
                try:
                    with SamEngine._lock:
                        if SamEngine._model is not None:
                            self.success = True
                            return
                        from ultralytics import SAM

                        fname = SAM_MODELS[model_name][0]
                        if not (MODELS_DIR / fname).exists():
                            ok = download_sam_model(model_name)
                            if not ok:
                                self.err = (
                                    "SAM 模型下载失败，请检查网络后重试，或手动下载到 models 目录：\n"
                                    f"{SAM_MODELS[model_name][1]}"
                                )
                                return
                        # ultralytics 会自动从当前目录查找权重
                        cwd = os.getcwd()
                        os.chdir(str(MODELS_DIR))
                        try:
                            SamEngine._model = SAM(str(MODELS_DIR / fname))
                        finally:
                            os.chdir(cwd)
                        SamEngine._model_name = model_name
                        self.success = True
                except Exception as e:
                    logger.exception("SAM 模型加载失败")
                    self.err = f"加载失败：{e}"

        loader = _LoadThread()
        self._loader = loader
        loader.finished.connect(lambda: parent_cb(loader.success, loader.err))
        loader.start()

    # ---------- 分割 ----------
    def segment(self, image_bgr, points: list, point_labels: list,
                box, request_id: int) -> None:
        """执行分割。points: [[x,y]...]（正点），point_labels: 1/0 与正负对应；
        box: [x1,y1,x2,y2] 或 None。结果经 segment_done(request_id, polygon) 返回。"""
        class _SegThread(QThread):
            def __init__(self):
                super().__init__()
                self.result = None
                self.err = ""

            def run(self):
                try:
                    with SamEngine._lock:
                        model = SamEngine._model
                    if model is None:
                        self.err = "SAM 模型未加载"
                        return
                    # ultralytics SAM predict: points=[[[...]]], labels=[[...]], bboxes=[...]
                    kwargs = {}
                    if box is not None:
                        kwargs["bboxes"] = [box]
                    if points:
                        pos = [p for p, l in zip(points, point_labels) if l == 1]
                        neg = [p for p, l in zip(points, point_labels) if l == 0]
                        if pos or neg:
                            kwargs["points"] = [pos + neg] if (pos and neg) else [pos or neg]
                            kwargs["labels"] = [[1] * len(pos) + [0] * len(neg)]
                    with SamEngine._lock:
                        results = model.predict(source=image_bgr, **kwargs)
                    result = results[0]
                    if result.masks is None or result.masks.data is None or not len(result.masks.data):
                        self.result = None
                        return
                    mask = result.masks.data[0].cpu().numpy()
                    self.result = mask_to_polygon(mask)
                except Exception as e:
                    logger.exception("SAM 分割失败")
                    self.err = f"分割失败：{e}"

        worker = _SegThread()
        self._worker = worker
        self._workers.add(worker)

        def on_finished():
            self.segment_done.emit(request_id, worker.result if not worker.err else None)
            self._workers.discard(worker)

        worker.finished.connect(on_finished)
        worker.start()


def mask_to_polygon(mask, epsilon_ratio: float = 0.002, min_area: int = 64) -> list:
    """二值掩码 → 多边形顶点列表 [[x,y],...]，Douglas-Peucker 简化。"""
    import cv2

    mask_u8 = (mask.astype("uint8")) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return []
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon_ratio * peri, True)
    pts = [[float(p[0][0]), float(p[0][1])] for p in approx]
    if len(pts) < 3:
        return []
    return pts
